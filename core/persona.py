"""
User persona: a structured, evolving profile stored in a Redis Hash.
Updated incrementally as facts are extracted from conversations.
Injected at the top of every recall response as context.

Redis key: mem:persona  (Hash)
Fields: name, work, location, style, preferences, rules, skills

v0.9.7 quality hardening:
  - _should_reject(): blocks AI self-descriptions, tool errors, status logs
  - Word-overlap dedup: rejects paraphrases (not just prefix matches)
  - Max 5 entries per field; newlines stripped
"""

import re
import redis.asyncio as aioredis

PERSONA_KEY = "mem:persona"

_FACT_TO_FIELD: dict[str, str] = {
    "identity":   "name",
    "work":       "work",
    "location":   "location",
    "preference": "preferences",
    "rule":       "rules",
    "skill":      "skills",
    "deadline":   "deadlines",
}

_MAX_ENTRIES_PER_FIELD = 5
_MAX_FIELD_LEN = 500

# ── Reject patterns — things that must never enter the persona ─────────────────

# Known real user name — any identity claim using a different name is rejected.
# Update this constant if the user's name changes.
_KNOWN_USER_NAME = "jace"

# Garbage "name" patterns: the literal phrase "my name is" used as a value,
# placeholder names from test sessions, or clearly fake/template identity strings.
_GARBAGE_NAME_PATTERNS = re.compile(
    r"(^我的名字是\s*$|^my name is\s*$|^name\s*$|^unnamed\s*$|^unknown\s*$|"
    r"^user\s*$|^test\s*$|^example\s*$|^placeholder\s*$)",
    re.I | re.UNICODE,
)

# AI self-reference: sentences about Claude/assistant, not the user
_AI_PATTERNS = re.compile(
    r"(the assistant|claude|i'?m (the|a|an|running|using|ready|sorry|not able|claude)|"
    r"identifies itself as|model made by|made by anthropic|running on claude|"
    r"i am running|i am claude|i'm claude code|as claude|anthropic)",
    re.I,
)

# Tool error / JSON debris
_NOISE_PATTERNS = re.compile(
    r"(tool_use_id|is_error|toolu_|'role':|'content':|\"role\":|\"content\":|"
    r"\{.*?:.*?\}|subagent context|depth \d+/\d+)",
    re.I,
)

# Status log fragments (not personal facts)
_STATUS_PATTERNS = re.compile(
    r"(nodes? (m\d+|jetson|win\d+)|gitee backup|proxy service|proxy is down|"
    r"gateway is live|no new crashes|disconnected from the network|"
    r"status alert|re-pairing with openclaw|has been recovered)",
    re.I,
)

_SECRET_PATTERNS = [
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{12,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\btoken-active-\d+\b", re.I),
    re.compile(
        r"(?i)\b(password|token|api[_ -]?key|secret)\b"
        r"\s*(?:is|=|:|for\s+[^:\n]{1,80}?\s+is)\s*['\"]?[^'\"\s;,\n]+"
    ),
]

# Too short to be meaningful (after stripping punctuation)
_MIN_CONTENT_LEN = 12


def _contains_secret(content: str) -> bool:
    c = content.strip()
    return any(p.search(c) for p in _SECRET_PATTERNS)


def _should_reject(content: str) -> bool:
    """Return True if this content should NOT be added to the persona."""
    c = content.strip()
    if len(c) < _MIN_CONTENT_LEN:
        return True
    if _AI_PATTERNS.search(c):
        return True
    if _NOISE_PATTERNS.search(c):
        return True
    if _STATUS_PATTERNS.search(c):
        return True
    if _contains_secret(c):
        return True
    return False


def _should_reject_name(content: str) -> bool:
    """Return True if this identity/name fact should NOT update the name field.

    Rejects:
    - The literal phrase "我的名字是" / "my name is" used as the name value itself
      (regex captured the wrong group from a Chinese identity pattern).
    - Names that do not match the known real user (_KNOWN_USER_NAME).
      Any name that is clearly not "Jace" (case-insensitive) and is not a
      meaningful description of Jace is blocked.

    We check conservatively: if the content contains the known name we allow it;
    if it contains a GARBAGE_NAME pattern we always reject it.
    """
    c = content.strip()
    # Always reject obvious garbage captures
    if _GARBAGE_NAME_PATTERNS.search(c):
        return True
    # Reject if "我的名字是" appears as content (the regex matched the phrase itself,
    # not an actual name — e.g. "我的名字是" with nothing after it).
    if "我的名字是" in c and _KNOWN_USER_NAME.lower() not in c.lower():
        return True
    # Reject if this looks like an identity statement for someone other than the
    # known user. We look for explicit name introductions with a name that is not
    # the known user. This catches "my name is <foreign-name>" patterns.
    _name_intro = re.compile(
        r"(?:my name is|i(?:'m| am)|我叫|我的名字是)\s+([^\s，。！\n]{1,30})",
        re.I | re.UNICODE,
    )
    for m in _name_intro.finditer(c):
        extracted_name = m.group(1).strip().lower()
        # If the extracted name doesn't contain the known user name, reject
        if extracted_name and _KNOWN_USER_NAME.lower() not in extracted_name:
            return True
    return False


def _word_overlap(a: str, b: str) -> float:
    """Jaccard word overlap: |intersection| / |union|. Fast near-duplicate check."""
    wa = set(re.findall(r"[a-zA-Z\u4e00-\u9fff]{3,}", a.lower()))
    wb = set(re.findall(r"[a-zA-Z\u4e00-\u9fff]{3,}", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _is_duplicate(content: str, existing_entries: list[str], threshold: float = 0.60) -> bool:
    """Return True if content overlaps ≥ threshold with any existing entry."""
    for entry in existing_entries:
        if _word_overlap(content, entry) >= threshold:
            return True
    return False


def _parse_entries(raw: str) -> list[str]:
    """Split a semicolon-delimited persona field into individual entries."""
    return [e.strip() for e in raw.split(";") if e.strip()]


def _best_entries(entries: list[str], max_n: int = _MAX_ENTRIES_PER_FIELD) -> list[str]:
    """
    Deduplicate and select the best (most informative = longest) entries.
    1. Filter rejects
    2. Word-overlap dedup (keep first seen, prefer longer)
    3. Cap at max_n
    """
    filtered = [e for e in entries if not _should_reject(e)]
    seen: list[str] = []
    for entry in sorted(filtered, key=len, reverse=True):  # longest first → keep most informative
        if not _is_duplicate(entry, seen):
            seen.append(entry)
        if len(seen) >= max_n:
            break
    return seen


async def update(r: aioredis.Redis, category: str, content: str) -> None:
    """Merge a new fact into the persona profile, with quality filtering."""
    field = _FACT_TO_FIELD.get(category)
    if not field:
        return

    # Strip newlines — persona entries must be single-line
    content = content.replace("\n", " ").replace("\r", " ").strip()

    if _should_reject(content):
        return

    # Extra guard for the name/identity field: reject fake or garbage user names.
    # This prevents test-session data (e.g. "我的名字是", Firebase test users) from
    # overwriting the real user's identity.
    if field == "name" and _should_reject_name(content):
        import logging as _logging
        _logging.getLogger("mem").warning(
            "[persona] rejected bad name fact: %r", content[:80]
        )
        return

    existing_raw = await r.hget(PERSONA_KEY, field)
    existing_entries = _parse_entries(
        existing_raw.decode() if isinstance(existing_raw, bytes) else (existing_raw or "")
    )

    if _is_duplicate(content, existing_entries):
        return

    # Prepend new (most recent is first) and keep best _MAX_ENTRIES_PER_FIELD
    merged = _best_entries([content] + existing_entries)
    new_val = "; ".join(merged)[:_MAX_FIELD_LEN]
    await r.hset(PERSONA_KEY, field, new_val)


async def get_context(r: aioredis.Redis) -> str:
    """Return persona as a formatted context block, or '' if empty."""
    raw = await r.hgetall(PERSONA_KEY)
    if not raw:
        return ""

    lines = ["## User Profile"]
    field_order = ["name", "work", "location", "preferences", "rules", "skills", "deadlines"]
    seen = set()
    for field in field_order:
        key_b = field.encode()
        val = raw.get(key_b) or raw.get(field)
        if val:
            v = val.decode() if isinstance(val, bytes) else val
            lines.append(f"- {field}: {v}")
            seen.add(field)
    for k, v in raw.items():
        k_str = k.decode() if isinstance(k, bytes) else k
        if k_str not in seen:
            v_str = v.decode() if isinstance(v, bytes) else v
            lines.append(f"- {k_str}: {v_str}")

    return "\n".join(lines)
