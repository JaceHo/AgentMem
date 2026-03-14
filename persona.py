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

# Too short to be meaningful (after stripping punctuation)
_MIN_CONTENT_LEN = 12


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
