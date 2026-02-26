"""
LLM summarizer: compress long turns before embedding.

MiniLM-L6 silently truncates at 256 word-pieces (~200 chars of dense text).
Long conversations lose their tail. Summarizing first ensures the full
turn is captured in the 384-dim embedding.

Uses GLM-4-flash (ZAI) — fast (~800ms) and cheap.
Degrades gracefully: returns truncated original if key unavailable or timeout.
"""

import json
import os
from pathlib import Path

import httpx

MIN_TO_SUMMARIZE = 280   # chars — below this, embed directly
MAX_SUMMARY_CHARS = 180  # target summary length
ZAI_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
TIMEOUT_S = 4.0


def _load_key() -> str:
    """Re-read on every call so key rotation is picked up without restart."""
    # 1. Env var (set in LaunchAgent plist overrides all)
    if k := os.environ.get("ZAI_API_KEY"):
        return k
    # 2. ~/.openclaw/.env  ← rotation script updates this
    env = Path.home() / ".openclaw" / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ZAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    # 3. openclaw.json provider key (fallback)
    cfg = Path.home() / ".openclaw" / "openclaw.json"
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text())
            return data["models"]["providers"]["zai"]["apiKey"]
        except Exception:
            pass
    return ""


async def summarize(text: str) -> str:
    """Return a compressed summary suitable for embedding, or original if short."""
    if len(text) <= MIN_TO_SUMMARIZE:
        return text

    key = _load_key()
    if not key:
        return text[:MAX_SUMMARY_CHARS]

    prompt = (
        f"Summarize in 1-2 concise sentences (under {MAX_SUMMARY_CHARS} chars). "
        f"Preserve names, entities, and key facts:\n\n{text[:1200]}"
    )

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            resp = await client.post(
                ZAI_URL,
                json={"model": "glm-4-flash",
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 80, "temperature": 0.1},
                headers={"Authorization": f"Bearer {key}"},
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        pass

    return text[:MAX_SUMMARY_CHARS]
