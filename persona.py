"""
User persona: a structured, evolving profile stored in a Redis Hash.
Updated incrementally as facts are extracted from conversations.
Injected at the top of every recall response as context.

Redis key: mem:persona  (Hash)
Fields: name, work, location, style, preferences, rules, skills
"""

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


async def update(r: aioredis.Redis, category: str, content: str) -> None:
    """Merge a new fact into the persona profile."""
    field = _FACT_TO_FIELD.get(category)
    if not field:
        return

    existing_raw = await r.hget(PERSONA_KEY, field)
    if existing_raw:
        existing = existing_raw.decode() if isinstance(existing_raw, bytes) else existing_raw
        # Skip near-duplicates (simple substring check, fast)
        short = content[:60]
        if short in existing:
            return
        new_val = f"{existing}; {content}"[:500]  # cap field length
    else:
        new_val = content[:500]

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
    # Any extra fields
    for k, v in raw.items():
        k_str = k.decode() if isinstance(k, bytes) else k
        if k_str not in seen:
            v_str = v.decode() if isinstance(v, bytes) else v
            lines.append(f"- {k_str}: {v_str}")

    return "\n".join(lines)
