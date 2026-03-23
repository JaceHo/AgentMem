"""
LLM summarizer: compress long turns before embedding.

MiniLM-L6 silently truncates at 256 word-pieces (~200 chars of dense text).
Long conversations lose their tail. Summarizing first ensures the full
turn is captured in the 384-dim embedding.

v0.9.9: Role-based routing via /v1/role/nlp. Same model selection as
extractor and retrieval_planner. Falls back to truncation if unavailable.
"""

import httpx
import logging

log = logging.getLogger("mem")

MIN_TO_SUMMARIZE = 280   # chars — below this, embed directly
MAX_SUMMARY_CHARS = 180  # target summary length
AISERV_URL   = "http://127.0.0.1:4000/v1/chat/completions"
AISERV_KEY   = "sk-aiserv-local"
TIMEOUT_S    = 6.0       # summarization generates ~40 tokens; 6s is generous


async def _get_model() -> str:
    """Ask aiserv for the best NLP model."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(
                "http://127.0.0.1:4000/v1/role/nlp",
                headers={"Authorization": f"Bearer {AISERV_KEY}"},
            )
            if resp.status_code == 200:
                return resp.json().get("model", "fast")
    except Exception:
        pass
    return "fast"


async def summarize(text: str) -> str:
    """Return a compressed summary suitable for embedding, or original if short."""
    if len(text) <= MIN_TO_SUMMARIZE:
        return text

    prompt = (
        f"Summarize in 1-2 concise sentences (under {MAX_SUMMARY_CHARS} chars). "
        f"Preserve names, entities, and key facts:\n\n{text[:1200]}"
    )

    model = await _get_model()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            resp = await client.post(
                AISERV_URL,
                json={
                    "model":      model,
                    "max_tokens": 80,
                    "messages":   [{"role": "user", "content": prompt}],
                },
                headers={"Authorization": f"Bearer {AISERV_KEY}"},
            )
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0]["message"]["content"].strip()
    except Exception as e:
        log.debug("[summarizer] %s failed: %s", model, e)

    return text[:MAX_SUMMARY_CHARS]
