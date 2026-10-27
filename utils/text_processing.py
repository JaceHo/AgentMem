"""
Shared text processing utilities.

Extracted from main.py to reduce duplication and improve testability.
"""

import ast
import json
import math
import re
from typing import Any


# ── Token estimation ───────────────────────────────────────────────────────
# CJK-family scripts (Han incl. extensions, kana, Hangul, CJK punctuation,
# fullwidth forms) are unsegmented: word-splitting scores a 1 000-character
# Chinese block as a handful of tokens. Each such character is charged ~1 token.
_CJK_RE = re.compile(
    r"[　-〿㐀-䶿一-鿿豈-﫿"
    r"︰-﹯＀-￯぀-ヿ가-힯]"
)
# Non-CJK text is charged 1 token per this many non-whitespace characters
# (~1.3 tokens per average English word).
_LATIN_CHARS_PER_TOKEN = 3.8
_NON_CJK_WS_RE = re.compile(r"[\s\u3000]+")


# ── Platform Tag Patterns ──────────────────────────────────────────────────
_PLATFORM_TAG_RE = re.compile(
    r"\[\[reply_to_current\]\]|\[\[reply_to:[^\]]*\]\]",
    re.IGNORECASE,
)

_CROSS_SESSION_PREFIX_RE = re.compile(
    r"^\s*<cross_session_memory>.*?</cross_session_memory>\s*",
    re.DOTALL | re.IGNORECASE,
)

_ONLY_CROSS_SESSION_RE = re.compile(
    r"^\s*<cross_session_memory>.*?</cross_session_memory>\s*(HEARTBEAT[^\n]*)?\s*$",
    re.DOTALL | re.IGNORECASE,
)

_OPTION_REPLY_RE = re.compile(
    r"^\s*(?:option|choice|select|pick)?\s*[#(]?\s*(\d{1,2})\s*[)\].:,-]?\s*$",
    re.IGNORECASE,
)

# ── Secret Redaction Patterns ─────────────────────────────────────────────
_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{12,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\btoken-active-\d+\b", re.IGNORECASE), "[REDACTED_TOKEN]"),
]

_SECRET_KEYWORD_VALUE_RE = re.compile(
    r"(?i)\b(password|token|api[_ -]?key|secret)\b"
    r"(\s*(?:is|=|:|for\s+[^:\n]{1,80}?\s+is)\s*)"
    r"([\'\"]?)([^\'\"\s;,\n]+)\3"
)

# ── Trivial Message Filter ────────────────────────────────────────────────
_TRIVIAL_PATTERNS = re.compile(
    r"^(hi+|hello+|hey+|yo+|ok+|okay|sure|thanks?|thx|bye+|good\s*(morning|night|day)|"
    r"好的|嗯|哦|呵|哈|谢谢|再见|早|晚安|你好|您好|ji+|test+|ping)[\s!?.。！？]*$",
    re.IGNORECASE,
)


def flatten_message_content(content: Any) -> str:
    """
    Flatten nested message content into plain text.
    
    Handles strings, dicts (with type/content keys), and lists recursively.
    Filters out tool_use, tool_result, and thinking blocks.
    
    Args:
        content: Message content (str, dict, list, or other)
        
    Returns:
        Flattened text string
    """
    if isinstance(content, str):
        raw = content.strip()
        if not raw:
            return ""
        # Try parsing as JSON/Python literal if it looks structured
        if raw[:1] in "[{":
            parsed = None
            try:
                parsed = json.loads(raw)
            except Exception:
                try:
                    parsed = ast.literal_eval(raw)
                except Exception:
                    parsed = None
            if parsed is not None and parsed is not content:
                return flatten_message_content(parsed)
        return raw
    
    if isinstance(content, dict):
        block_type = str(content.get("type", "")).lower()
        if block_type in {"tool_use", "tool_result", "thinking"}:
            return ""
        if "text" in content:
            return flatten_message_content(content.get("text"))
        if "content" in content:
            return flatten_message_content(content.get("content"))
        return ""
    
    if isinstance(content, list):
        parts = [flatten_message_content(item) for item in content]
        return " ".join(part for part in parts if part).strip()
    
    return ""


def estimate_tokens(text: str) -> int:
    """
    CJK-aware token estimate.

    ``len(text.split())`` (the v1.0 behaviour) scores a 4 000-character Chinese
    block as *one* token because Chinese is written without spaces.  That made
    every ``token_budget`` check in context packing a no-op for the persona /
    fact blocks this deployment actually serves, so a 600-token budget returned
    ~1 600 real tokens.

    Model (deliberately conservative, cheap, no tokenizer dependency):
      * CJK-family characters  -> ~1 token each (cl100k/o200k split Han at
        roughly 1–1.5 chars per token; 1.0 keeps the estimate honest rather
        than optimistic).
      * everything else        -> character count / 3.8 (average English word
        is ~4.75 chars plus a separator, i.e. ~1.25 tokens per word).

    Args:
        text: Text to estimate token count for

    Returns:
        Estimated token count (>= 0)
    """
    if not text:
        return 0
    if not isinstance(text, str):
        text = str(text)
    cjk = len(_CJK_RE.findall(text))
    other = len(_NON_CJK_WS_RE.sub("", text))
    return cjk + math.ceil(other / _LATIN_CHARS_PER_TOKEN)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Clip ``text`` so that ``estimate_tokens(result) <= max_tokens``.

    Packing that *drops* an over-budget block is as useless as packing that
    ignores the budget: the persona block is priority-1 context and must arrive,
    just smaller.  Prefers a line boundary, then a sentence boundary, then hard
    character cutting; a trailing ``…`` is added when anything was removed.
    Returns "" for a non-positive budget.
    """
    if not text or max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text

    # Char budget implied by the token budget. Worst case is all-CJK (1 char
    # per token); latin inflates faster, so iterate rather than guess once.
    limit = max(1, min(len(text), int(max_tokens * _LATIN_CHARS_PER_TOKEN)))
    cut = text[:limit]
    if "\n" in cut:
        nl = cut.rfind("\n")
        if nl >= limit // 2:
            cut = cut[:nl]
    elif len(cut) < len(text):
        for sep in ("。", ". ", "；", "; ", "，", ", "):
            idx = cut.rfind(sep)
            if idx >= int(limit * 0.6):
                cut = cut[: idx + len(sep.rstrip())]
                break
    cut = cut.rstrip()
    while cut and estimate_tokens(cut + "…") > max_tokens:
        cut = cut[: max(1, len(cut) - 16)].rstrip()
    if not cut:
        return ""
    return cut + ("…" if len(cut) < len(text) else "")


def redact_secrets(text: str) -> str:
    """
    Redact API keys, passwords, tokens from text.
    
    Replaces known secret patterns with [REDACTED_*] placeholders.
    Handles GitHub tokens, OpenAI keys, generic API keys, passwords.
    
    Args:
        text: Text that may contain secrets
        
    Returns:
        Text with secrets redacted
    """
    if not text:
        return ""
    
    redacted = text
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    
    # Redact keyword=value patterns (password=xxx, token=yyy, etc.)
    redacted = _SECRET_KEYWORD_VALUE_RE.sub(
        lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]",
        redacted,
    )
    
    return redacted


def contains_secret(text: str) -> bool:
    """
    Check if text contains any secrets.
    
    Args:
        text: Text to check
        
    Returns:
        True if secrets detected, False otherwise
    """
    if not text:
        return False
    if any(pattern.search(text) for pattern, _ in _SECRET_PATTERNS):
        return True
    return bool(_SECRET_KEYWORD_VALUE_RE.search(text))


def strip_platform_noise(text: str) -> str:
    """
    Remove platform tags and cross_session_memory prefixes.
    
    Strips:
    1. Leading <cross_session_memory>…</cross_session_memory> block
    2. [[reply_to_current]] / [[reply_to:<id>]] tags
    
    Args:
        text: Text with potential platform noise
        
    Returns:
        Cleaned text
    """
    # Strip cross_session_memory prefix block
    text = _CROSS_SESSION_PREFIX_RE.sub("", text)
    # Strip platform reply tags
    text = _PLATFORM_TAG_RE.sub("", text)
    return text.strip()


def is_only_platform_noise(text: str) -> bool:
    """
    Check if text is ONLY a cross_session_memory wrapper with no real content.
    
    Args:
        text: Text to check
        
    Returns:
        True if text is only platform noise
    """
    return bool(_ONLY_CROSS_SESSION_RE.match(text or ""))


def is_brief_option_reply(text: str) -> bool:
    """
    Check if text is a brief option selection (e.g., "1", "option 2", "choice 3").
    
    Args:
        text: Text to check
        
    Returns:
        True if text is an option reply
    """
    return bool(_OPTION_REPLY_RE.match(text or ""))


def is_trivial(text: str, min_chars: int = 15) -> bool:
    """
    Check if text is trivial (greeting, single word, etc.).
    
    Skips storing episodes for low-value exchanges like "hi", "thanks", etc.
    
    Args:
        text: Text to check
        min_chars: Minimum length to consider non-trivial
        
    Returns:
        True if text is trivial
    """
    t = text.strip()
    if len(t) < min_chars:
        return True
    return bool(_TRIVIAL_PATTERNS.match(t))


def is_injected_system_content(text: str) -> bool:
    """
    Check if text is system-injected content that should be skipped.
    
    Detects memory context blocks, user profile sections, cron jobs, etc.
    
    Args:
        text: Text to check
        
    Returns:
        True if text is injected system content
    """
    skip_prefixes = (
        "## Long-Term Memory",
        "## Recent Relevant Episodes",
        "## Current Session Context",
        "## User Profile",
        "[cron:",
    )
    t = text.strip()
    return any(t.startswith(p) for p in skip_prefixes)
