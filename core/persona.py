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

2026-10-22 hygiene + PII hardening (after a national-ID-card path was found in
the live persona and was therefore being injected into every agent prompt):
  - _should_reject() also blocks filesystem paths and identity-document refs
  - per-entry length cap + byte-aware field composition (no mid-entry truncation)
  - _best_entries() keeps RECENT entries, not "the longest ever seen"
  - rejects are logged at WARNING — never silent
  - sanitize_stored(): re-applies the rules to what is already in Redis
"""

import logging
import re
import redis.asyncio as aioredis

from .utils import decode_bytes

log = logging.getLogger("mem")

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
_MAX_FIELD_BYTES = 500        # the cap that actually leaves the process (JSON)
_MAX_ENTRY_LEN = 160          # one persona line; longer = a pasted log/report

# ── PII / path guards ─────────────────────────────────────────────────────────
# The persona block is injected into EVERY agent's context on every /recall. A
# single stored fact that points at an identity document therefore leaks a real
# PII pointer into every downstream prompt forever (this actually happened:
# "Both sides of the national ID card are stored in ~/Documents/profile/resume/…"
# landed in the `location` field and shipped on every recall).
# Reject by *shape*, not by exact spelling, so new phrasings can't slip past.

# Absolute / home / tilde paths and Windows drive paths. Requires a real path
# shape (leading / with a second segment, ~/ or X:\) rather than any slash, so
# "and/or", relative "src/utils/x.py" and http:// URLs stay untouched.
_FS_PATH_RE = re.compile(
    r"(?:^|(?<=[\s'\"(（【《,，;；:：]))"
    r"(?:"
    r"~[/\\][^\s'\"()（，。;；]{2,}"
    r"|/[A-Za-z0-9._+\-]{2,}(?:/[^\s'\"()（，。;；]+)+/?"
    r"|[A-Za-z]:[\\/][^\s'\"()（，。;；]{2,}"
    r")",
    re.UNICODE,
)

# Identity / banking / residency documents and numbers, in either language.
_ID_DOC_RE = re.compile(
    r"(身份证|身分证|身份证明|证件照|户口本?|户口簿|购房合同|护照|签证|驾驶证|行驶证|"
    r"社保卡|医保卡|银行卡|储蓄卡|信用卡号|实名|人脸|指纹|"
    r"\bnational\s+id\b|\bid\s+card\b|\bpassport\b|\bdriving\s+licen[cs]e\b|"
    r"\bsocial\s+security\b|\bbank\s+card\b|\bkyc\b|\bbiometric\b)",
    re.I | re.UNICODE,
)

# A 15/18-digit PRC ID number, with or without separators.
_ID_NUMBER_RE = re.compile(r"\b\d{6}[- ]?\d{8}[- ]?\d{3}[\dxX]\b")


def _has_pii_pointer(content: str) -> str:
    """Return the name of the PII rule that fired, or '' when clean."""
    if _ID_NUMBER_RE.search(content):
        return "id_number"
    if _ID_DOC_RE.search(content):
        return "id_document"
    if _FS_PATH_RE.search(content):
        return "fs_path"
    return ""

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


def _reject_reason(content: str) -> str:
    """Return which rule blocks this content ('' = acceptable).

    Named separately from _should_reject so the caller can LOG the reason: an
    invisible filter is how a national-ID-card file path reached the persona
    plane in the first place.
    """
    c = content.strip()
    if len(c) < _MIN_CONTENT_LEN:
        return "too_short"
    if _AI_PATTERNS.search(c):
        return "ai_self_description"
    if _NOISE_PATTERNS.search(c):
        return "tool_noise"
    if _STATUS_PATTERNS.search(c):
        return "status_log"
    if _contains_secret(c):
        return "secret"
    if _has_pii_pointer(c):
        return f"pii:{_has_pii_pointer(c)}"
    return ""


def _should_reject(content: str) -> bool:
    """Return True if this content should NOT be added to the persona.

    Blocks AI self-descriptions, tool errors, status logs, secrets and — since
    the PII leak of 2026-10 — filesystem paths and identity-document references,
    because this block is injected into every agent's context on every recall.
    """
    return bool(_reject_reason(content))


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


def _clamp_entry(entry: str, max_len: int = _MAX_ENTRY_LEN) -> str:
    """Hard-cap a single persona entry, cutting at a boundary (never mid-word).

    A persona entry is one line of a profile that is pasted into every agent's
    context. A 400-char log excerpt is not a profile line, so it is capped here
    rather than stored whole and truncated at the *field* level — the old
    behaviour, which cut the joined string at 500 characters and so could drop a
    profile line in mid-sentence.
    """
    e = " ".join(entry.split()).strip()
    if len(e) <= max_len:
        return e
    cut = e[:max_len]
    for sep in ("。", ". ", "；", "; ", "，", ", ", " "):
        idx = cut.rfind(sep)
        if idx >= int(max_len * 0.6):
            return cut[:idx].rstrip(" ;;.。，,")
    return cut.rstrip(" ;;.。，,")


def _compose_field(entries: list[str],
                   max_chars: int = _MAX_FIELD_LEN,
                   max_bytes: int = _MAX_FIELD_BYTES) -> str:
    """Join persona entries without ever exceeding the char OR byte budget.

    ``[:max_chars]`` alone is not a cap: the persona is served as JSON, so a
    CJK-heavy field blew past its own limit in bytes (live value: 500 chars /
    576 bytes). Whole entries are dropped instead of characters, so the result
    is always valid, complete profile lines.
    """
    out: list[str] = []
    used_chars = 0
    used_bytes = 0
    for e in entries:
        addition = len(e) + (2 if out else 0)          # "; " separator
        addition_bytes = len(e.encode("utf-8")) + (2 if out else 0)
        if used_chars + addition > max_chars or used_bytes + addition_bytes > max_bytes:
            break
        out.append(e)
        used_chars += addition
        used_bytes += addition_bytes
    return "; ".join(out)


def _best_entries(entries: list[str], max_n: int = _MAX_ENTRIES_PER_FIELD) -> list[str]:
    """
    Deduplicate and select the best entries — most recent first.

    1. Filter rejects (quality + PII rules)
    2. Clamp every entry to a real, bounded length
    3. Word-overlap dedup, keeping the FIRST (most recent) occurrence

    The old version sorted ``key=len, reverse=True`` before deduping. Because
    callers pass ``[new] + existing``, that made the longest entry ever seen
    win the slot permanently: a single 400-char log excerpt stored months ago
    outranked every newer, shorter, more accurate identity claim. Recency is the
    correct ordering for a profile that is re-learned from live conversations.
    """
    clamped = [
        _clamp_entry(e) for e in entries
        if not _should_reject(e)
    ]
    seen: list[str] = []
    for entry in clamped:
        if not entry:
            continue
        if not _is_duplicate(entry, seen):
            seen.append(entry)
        if len(seen) >= max_n:
            break
    return seen


def _reject_logged(field: str, content: str) -> bool:
    """Reject `content` for `field`, WARNING-logging the reason.

    A silent filter is indistinguishable from a broken one — this is the same
    mistake that let access_count sit at 0 for weeks. Rejecting a real fact now
    leaves a line in stderr.log that ops can grep.
    """
    reason = _reject_reason(content)
    if not reason:
        return False
    log.warning("[persona] reject field=%s reason=%s content=%r",
                field, reason, content[:120])
    return True


async def update(r: aioredis.Redis, category: str, content: str) -> None:
    """Merge a new fact into the persona profile, with quality filtering."""
    field = _FACT_TO_FIELD.get(category)
    if not field:
        return

    # Strip newlines — persona entries must be single-line
    content = content.replace("\n", " ").replace("\r", " ").strip()

    if _reject_logged(field, content):
        return

    # Extra guard for the name/identity field: reject fake or garbage user names.
    # This prevents test-session data (e.g. "我的名字是", Firebase test users) from
    # overwriting the real user's identity.
    if field == "name" and _should_reject_name(content):
        log.warning("[persona] reject field=name reason=bad_identity content=%r",
                    content[:120])
        return

    existing_raw = await r.hget(PERSONA_KEY, field)
    existing_entries = _parse_entries(
        decode_bytes(existing_raw)
    )

    if _is_duplicate(content, existing_entries):
        return

    # Prepend new (most recent is first) and keep best _MAX_ENTRIES_PER_FIELD
    merged = _best_entries([content] + existing_entries)
    new_val = _compose_field(merged)
    await r.hset(PERSONA_KEY, field, new_val)


async def update_many(r: aioredis.Redis, entries: list[tuple[str, str]]) -> None:
    """Merge many facts into the persona profile in two round trips.

    ``update`` costs one HGET + one HSET *per fact*, and the store path calls
    it once per extracted fact.  This variant reads the whole persona hash
    once, replays every entry against an in-memory view of the fields (so the
    second fact still sees the first one's effect — the same ordering the
    sequential path produced), and writes back only the fields that changed.

    Entries whose category maps to no persona field, or that are rejected by
    the same quality filters as ``update``, are skipped (and logged).
    """
    if not entries:
        return

    # 1. Reject / normalise up front so the field view only sees real content.
    pending: dict[str, list[str]] = {}   # field -> contents, in arrival order
    for category, content in entries:
        field = _FACT_TO_FIELD.get(category)
        if not field:
            continue
        content = content.replace("\n", " ").replace("\r", " ").strip()
        if _reject_logged(field, content):
            continue
        if field == "name" and _should_reject_name(content):
            log.warning("[persona] reject field=name reason=bad_identity content=%r",
                        content[:120])
            continue
        pending.setdefault(field, []).append(content)

    if not pending:
        return

    # 2. One read of the current profile.
    raw_all = await r.hgetall(PERSONA_KEY) or {}
    raw_all = {decode_bytes(k): decode_bytes(v) for k, v in raw_all.items()}

    # 3. Replay each entry against the evolving in-memory view — this is what
    #    keeps the batch semantically identical to calling update() in a loop.
    writes: dict[str, str] = {}
    for field, contents in pending.items():
        current = _parse_entries(raw_all.get(field, ""))
        for content in contents:
            if _is_duplicate(content, current):
                continue
            current = _best_entries([content] + current)
        new_val = _compose_field(current)
        if new_val != raw_all.get(field, ""):
            writes[field] = new_val

    # 4. One write of the changed fields.
    if writes:
        pipe = r.pipeline(transaction=False)
        for field, val in writes.items():
            pipe.hset(PERSONA_KEY, field, val)
        await pipe.execute()


async def sanitize_stored(r: aioredis.Redis) -> dict:
    """Re-apply today's hygiene rules to the persona that is ALREADY stored.

    New filters only guard future writes: a PII entry stored last month keeps
    being injected into every recall until it is overwritten. Called on service
    startup so a stale or hand-edited persona is cleaned without an operator
    having to notice it. Removes rejected entries, clamps long ones and re-caps
    each field; deletes a field that ends up empty.

    Returns {"field": n_entries_removed} for fields that changed.
    """
    raw = await r.hgetall(PERSONA_KEY) or {}
    current = {decode_bytes(k): decode_bytes(v) for k, v in raw.items()}
    removed: dict[str, int] = {}
    writes: dict[str, str] = {}
    deletes: list[str] = []

    for field, val in current.items():
        entries = _parse_entries(val)
        kept = _best_entries(entries)
        rebuilt = _compose_field(kept)
        if rebuilt == val:
            continue
        dropped = max(0, len(entries) - len(kept))
        for entry in entries:
            if entry not in kept:
                log.warning("[persona] sanitize drop field=%s reason=%s content=%r",
                            field, _reject_reason(entry) or "cap", entry[:120])
        if rebuilt:
            writes[field] = rebuilt
        else:
            deletes.append(field)
        removed[field] = dropped

    if writes or deletes:
        pipe = r.pipeline(transaction=False)
        for field, val in writes.items():
            pipe.hset(PERSONA_KEY, field, val)
        for field in deletes:
            pipe.hdel(PERSONA_KEY, field)
        await pipe.execute()
        log.info("[persona] sanitized %d field(s): %s",
                 len(writes) + len(deletes),
                 ", ".join(sorted(list(writes) + deletes)))
    return removed


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
            v = decode_bytes(val)
            lines.append(f"- {field}: {v}")
            seen.add(field)
    for k, v in raw.items():
        k_str = decode_bytes(k)
        if k_str not in seen:
            v_str = decode_bytes(v)
            lines.append(f"- {k_str}: {v_str}")

    return "\n".join(lines)
