"""
LLM summarizer: compress long turns before embedding.

MiniLM-L6 silently truncates at 256 word-pieces (~200 chars of dense text).
Long conversations lose their tail. Summarizing first ensures the full
turn is captured in the 384-dim embedding.

v0.9.8: Switched back to ZAI GLM-4-flash via aiserv (ZAI recovered 2026-03-17).
aiserv routes to zai-gf-1..5 pool (5 keys × 300 RPM). ~1-3s domestic.
Degrades gracefully: returns truncated original if aiserv unavailable or timeout.
"""

import httpx

MIN_TO_SUMMARIZE = 280   # chars — below this, embed directly
MAX_SUMMARY_CHARS = 180  # target summary length
AISERV_URL   = "http://127.0.0.1:4000/v1/messages"
AISERV_KEY   = "sk-aiserv-local-dev"
AISERV_MODEL = "glm-4-flash"
TIMEOUT_S    = 12.0  # GLM-4-flash ~1-3s; 12s covers cold start


async def summarize(text: str) -> str:
    """Return a compressed summary suitable for embedding, or original if short."""
    if len(text) <= MIN_TO_SUMMARIZE:
        return text

    prompt = (
        f"Summarize in 1-2 concise sentences (under {MAX_SUMMARY_CHARS} chars). "
        f"Preserve names, entities, and key facts:\n\n{text[:1200]}"
    )

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            resp = await client.post(
                AISERV_URL,
                json={
                    "model":      AISERV_MODEL,
                    "max_tokens": 80,
                    "messages":   [{"role": "user", "content": prompt}],
                },
                headers={"x-api-key": AISERV_KEY},
            )
            if resp.status_code == 200:
                blocks = resp.json().get("content", [])
                if blocks and blocks[0].get("type") == "text":
                    return blocks[0]["text"].strip()
    except Exception:
        pass

    return text[:MAX_SUMMARY_CHARS]
