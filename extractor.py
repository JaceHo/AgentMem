"""
Hybrid fact extractor: fast regex (Layer 1) + LLM extraction (Layer 2).

Layer 1: Compiled regex patterns — ~1ms, catches explicit first-person declarations.
Layer 2: GLM-4-flash LLM call  — ~200-500ms, catches implicit/Chinese/contextual facts.

Both layers run together; results are merged and deduped before return.
The store path in main.py is async (background), so LLM latency is invisible to the user.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger("mem")


@dataclass
class ExtractedFact:
    content: str
    category: str = "general"
    confidence: float = 0.8
    source: str = "regex"     # "regex" or "llm"
    keywords: list[str] | None = None
    persons: list[str] | None = None
    entities: list[str] | None = None


# ── Layer 1: Regex patterns (fast, ~1ms) ──────────────────────────────────────

_PATTERNS = [
    (re.compile(r"(?:my name is|I am|I'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.I), "identity"),
    (re.compile(r"(?:I (?:work|am working) (?:at|for)|I work as)\s+(.+?)(?:\.|,|$)", re.I), "work"),
    (re.compile(r"(?:I (?:like|love|prefer|enjoy|hate|dislike))\s+(.+?)(?:\.|,|$)", re.I), "preference"),
    (re.compile(r"(?:I (?:live|am|stay) in|I'm from|I'm based in)\s+(.+?)(?:\.|,|$)", re.I), "location"),
    (re.compile(r"(?:my (?:phone|email|address|birthday|goal|project) is)\s+(.+?)(?:\.|,|$)", re.I), "personal"),
    (re.compile(r"(?:remember|note|important):\s*(.+?)(?:\.|$)", re.I), "reminder"),
    (re.compile(r"(?:always|never|always use|never use)\s+(.+?)(?:\.|,|$)", re.I), "rule"),
    (re.compile(r"(?:the (?:api|key|token|secret|password) (?:is|for .+ is))\s+(\S+)", re.I), "credential"),
    (re.compile(r"(?:deadline|due date|by)\s+([\w\s,]+\d{4}|\d{4}-\d{2}-\d{2})", re.I), "deadline"),
]


def _regex_extract(user_text: str) -> list[ExtractedFact]:
    """Layer 1: instant regex extraction."""
    facts: list[ExtractedFact] = []
    seen: set[str] = set()

    for pattern, category in _PATTERNS:
        for match in pattern.finditer(user_text):
            snippet = match.group(0).strip()[:200]
            if snippet and snippet not in seen:
                seen.add(snippet)
                facts.append(ExtractedFact(
                    content=snippet, category=category, confidence=0.8, source="regex"
                ))
    return facts


# ── Layer 2: LLM extraction (GLM-4-flash, ~200-500ms) ────────────────────────

ZAI_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
LLM_TIMEOUT_S = 10.0
LLM_MAX_INPUT = 1500   # chars of conversation to send

_EXTRACT_PROMPT = """从以下对话中提取关于用户的关键事实。只提取明确表述的事实，不要推测。

【重要要求 — 原子化】
1. 每条事实必须是完整、独立、无歧义的句子（Atomic Entry）
2. 禁止使用代词（他/她/它/我/你/this/that/they），必须用具体名字替代
3. 时间必须用绝对格式（2026-03-01），禁止"昨天/今天/上周"等相对时间
4. 每条事实脱离对话上下文后仍可独立理解

返回 JSON 数组，每条记录格式:
[{
  "content": "完整的原子化事实陈述（包含主语、时间、具体内容）",
  "category": "分类",
  "keywords": ["关键词1", "关键词2"],
  "persons": ["人名1"],
  "entities": ["实体名"]
}]

分类只能是以下之一:
- identity: 姓名、身份
- work: 工作、职位、公司
- preference: 偏好、喜好、厌恶
- location: 位置、城市、国家
- personal: 个人信息（邮箱、生日等）
- reminder: 用户要求记住的事项
- rule: 用户设定的规则（总是/永远不）
- decision: 技术决策、方案选择
- context: 项目背景、当前任务
- credential: API密钥、令牌（注意脱敏）

示例:
对话: "user: 我喜欢用bun，不要用npm"
输出: [{"content": "用户偏好使用 bun 作为 JavaScript 包管理器，明确拒绝 npm", "category": "preference", "keywords": ["bun", "npm", "包管理器"], "persons": [], "entities": ["bun", "npm"]}]

如果没有值得提取的事实，返回空数组 []
支持中英文混合提取。

对话内容:
{conversation}"""


def _load_key() -> str:
    """Re-read on every call so key rotation is picked up without restart."""
    if k := os.environ.get("ZAI_API_KEY"):
        return k
    local_env = Path(__file__).parent / ".env"
    if local_env.exists():
        for line in local_env.read_text().splitlines():
            if line.startswith("ZAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    cfg = Path.home() / ".openclaw" / "openclaw.json"
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text())
            return data["models"]["providers"]["zai"]["apiKey"]
        except Exception:
            pass
    return ""


async def _llm_extract(conversation_text: str) -> list[ExtractedFact]:
    """Layer 2: LLM-based extraction via GLM-4-flash."""
    key = _load_key()
    if not key:
        log.debug("[extractor] no ZAI key, skipping LLM extraction")
        return []

    prompt = _EXTRACT_PROMPT.replace("{conversation}", conversation_text[:LLM_MAX_INPUT])

    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT_S) as client:
            resp = await client.post(
                ZAI_URL,
                json={
                    "model": "glm-4-flash",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.1,
                },
                headers={"Authorization": f"Bearer {key}"},
            )
            if resp.status_code != 200:
                log.warning(f"[extractor] LLM returned {resp.status_code}")
                return []

            raw = resp.json()["choices"][0]["message"]["content"].strip()

            # Parse JSON from response (handle markdown code blocks)
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            items = json.loads(raw)
            if not isinstance(items, list):
                return []

            facts = []
            for item in items:
                if not isinstance(item, dict) or not item.get("content"):
                    continue
                category = item.get("category", "general")
                # Skip credential extraction from LLM for safety
                if category == "credential":
                    continue
                facts.append(ExtractedFact(
                    content=str(item["content"]).strip()[:300],
                    category=category,
                    confidence=0.85,
                    source="llm",
                    keywords=item.get("keywords") or None,
                    persons=item.get("persons") or None,
                    entities=item.get("entities") or None,
                ))
            return facts

    except json.JSONDecodeError as e:
        log.warning(f"[extractor] LLM returned non-JSON: {e}")
        return []
    except Exception as e:
        log.warning(f"[extractor] LLM extraction failed: {type(e).__name__}: {e}", exc_info=True)
        return []


# ── Merge + dedup ─────────────────────────────────────────────────────────────

def _dedup(facts: list[ExtractedFact]) -> list[ExtractedFact]:
    """Deduplicate by content similarity (substring check for speed)."""
    result: list[ExtractedFact] = []
    seen_contents: list[str] = []

    for fact in facts:
        content_lower = fact.content.lower()
        is_dup = False
        for existing in seen_contents:
            # Substring match in either direction
            if content_lower in existing or existing in content_lower:
                is_dup = True
                break
        if not is_dup:
            result.append(fact)
            seen_contents.append(content_lower)

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def extract(messages: list[dict]) -> list[ExtractedFact]:
    """Synchronous regex-only extraction (backward compatible)."""
    user_text = " ".join(
        m["content"] if isinstance(m["content"], str)
        else " ".join(b.get("text", "") for b in m["content"] if isinstance(b, dict))
        for m in messages if m.get("role") == "user"
    )
    return _regex_extract(user_text)


async def extract_hybrid(
    messages: list[dict],
    conversation_text: str = "",
) -> list[ExtractedFact]:
    """
    Hybrid extraction: regex (instant) + LLM (async).
    conversation_text: full "role: content" text for LLM context.
    Falls back to regex-only if LLM fails.
    """
    # Layer 1: regex (~1ms)
    user_text = " ".join(
        m["content"] if isinstance(m["content"], str)
        else " ".join(b.get("text", "") for b in m["content"] if isinstance(b, dict))
        for m in messages if m.get("role") == "user"
    )
    regex_facts = _regex_extract(user_text)

    # Layer 2: LLM (~200-500ms)
    llm_input = conversation_text or user_text
    llm_facts = await _llm_extract(llm_input)

    # Merge: LLM first (atomized, richer metadata), then regex additions
    # LLM facts are self-contained with coreference resolution;
    # regex facts are raw snippets kept only if LLM missed them
    merged = llm_facts + regex_facts
    deduped = _dedup(merged)

    log.info(f"[extractor] regex={len(regex_facts)} llm={len(llm_facts)} "
             f"merged={len(deduped)}")

    return deduped
