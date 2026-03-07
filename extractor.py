"""
Hybrid fact extractor — v0.9.1 (SimpleMem-aligned)

Implements SimpleMem's Semantic Structured Compression (Section 3.1):
  Φ_gate(W) → {m_k}  — implicit semantic density gating via LLM
  Φ_coref             — pronoun resolution (no he/she/it/they/this)
  Φ_time              — absolute timestamp disambiguation
  I(m_k) = {s_k, l_k, r_k} — multi-view indexing: semantic + lexical + symbolic

Three-layer extraction:
  Layer 1: Compiled regex (< 1ms) — fast explicit first-person facts
  Layer 2: SimpleMem LLM gate (200-500ms) — structured lossless_restatement
            with topic, location, persons, entities, importance score
  Layer 3: Merge + dedup

The LLM prompt produces pronoun-free, absolute-timestamp memory units
that are self-contained and independently interpretable — exactly as
specified in SimpleMem Section 3.1.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import datetime

import httpx

log = logging.getLogger("mem")


@dataclass
class ExtractedFact:
    content: str                          # lossless_restatement (pronoun-free, abs. time)
    category: str = "general"
    confidence: float = 0.8
    source: str = "regex"                 # "regex" or "llm"
    keywords: list[str] | None = None
    persons: list[str] | None = None
    entities: list[str] | None = None
    topic: str | None = None             # SimpleMem: topic phrase (NEW v0.9.1)
    location: str | None = None          # SimpleMem: symbolic location (NEW v0.9.1)
    importance: float = 0.5             # SimpleMem: importance score 0-1 (NEW v0.9.1)


# ── Layer 1: Regex patterns (fast, ~1ms) ──────────────────────────────────────

_PATTERNS = [
    # ── English patterns ──────────────────────────────────────────────────────
    # Identity
    (re.compile(r"(?:my name is|I am|I'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.I), "identity"),
    # Work / profession — covers "I work at X", "I work as X", "I'm a/an X", "I'm an X at Y"
    (re.compile(r"(?:I (?:work|am working) (?:at|for)|I work as)\s+(.+?)(?:\.|,|$)", re.I), "work"),
    (re.compile(r"I'?m (?:a|an)\s+([^,.]+?)\s+(?:at|for|in)\s+(.+?)(?:\.|,|$)", re.I),           "work"),
    (re.compile(r"I'?m (?:a|an)\s+([A-Za-z][\w\s]{3,30})(?:\.|,|$)", re.I),                       "work"),
    (re.compile(r"I(?:'ve been| have been) (?:working|teaching|practicing|researching)\s+(?:at|for|as|in)\s+(.+?)(?:\.|,|$)", re.I), "work"),
    (re.compile(r"I(?:'ve been| have been) (?:a|an)\s+([^,.]+?)\s+for\s+(\d+\s+\w+)", re.I),      "work"),
    # Where someone works (named institution)
    (re.compile(r"(?:at|for)\s+([A-Z][A-Za-z\s&]{2,40}),?\s+(?:I(?:'m| am)|my role)", re.I),      "work"),
    # Location
    (re.compile(r"(?:I (?:live|am|stay) in|I'm from|I'm based in)\s+(.+?)(?:\.|,|$)", re.I),       "location"),
    (re.compile(r"(?:I (?:bought|rent|own|moved to))\s+(?:a\s+\w+\s+in|to)\s+(.+?)(?:\.|,|$)", re.I), "location"),
    # Broader property/residence pattern: "bought/purchased/renting a [noun] in [Place]"
    (re.compile(r"\b(?:bought|purchased|own|renting|rented|moved to)\s+(?:a|an|my)\s+\w+\s+in\s+([A-Z][A-Za-z\s]{2,30}?)(?:\s+(?:last|this|two|three|four)|[.,]|$)", re.I), "location"),
    # Education
    (re.compile(r"I (?:studied|went to|graduated from|did my (?:PhD|degree|masters?|MBA|MFA|BA|BS) at)\s+(.+?)(?:\.|,|$)", re.I), "personal"),
    (re.compile(r"my (?:PhD|degree|masters?|MFA|MBA|BA|BS)\s+(?:is|was|from)\s+(?:at\s+)?(.+?)(?:\.|,|$)", re.I), "personal"),
    # Duration / experience — "I've been doing X for N years/months"
    (re.compile(r"(?:I've been|I have been)\s+(.+?)\s+for\s+(\w+\s+(?:years?|months?|weeks?))", re.I), "personal"),
    # Preferences
    (re.compile(r"(?:I (?:like|love|prefer|enjoy|hate|dislike))\s+(.+?)(?:\.|,|$)", re.I),         "preference"),
    # Named relations / people
    (re.compile(r"(?:my (?:partner|wife|husband|sister|brother|friend|coworker|colleague|neighbor))\s+([A-Z][a-z]+)", re.I), "personal"),
    # Personal facts
    (re.compile(r"(?:my (?:phone|email|address|birthday|goal|project) is)\s+(.+?)(?:\.|,|$)", re.I), "personal"),
    # Reminders / rules
    (re.compile(r"(?:remember|note|important):\s*(.+?)(?:\.|$)", re.I),                            "reminder"),
    (re.compile(r"(?:always|never|always use|never use)\s+(.+?)(?:\.|,|$)", re.I),                 "rule"),
    # Credentials (regex only, never via LLM for safety)
    (re.compile(r"(?:the (?:api|key|token|secret|password) (?:is|for .+ is))\s+(\S+)", re.I),      "credential"),
    (re.compile(r"(?:deadline|due date|by)\s+([\w\s,]+\d{4}|\d{4}-\d{2}-\d{2})", re.I),           "deadline"),
    # Capability patterns
    (re.compile(r"(?:I (?:used|ran|executed|called|invoked)|using|via)\s+([\w\-]+(?:\s+tool|command)?)\s+to\s+(.+?)(?:\.|,|$)", re.I), "tool_use"),
    (re.compile(r"(?:installed|added|enabled|activated)\s+(.+?)\s+(?:tool|plugin|skill|extension|mcp)", re.I), "capability_gained"),
    (re.compile(r"(?:switched to|now using|changed to|upgrade to)\s+(.+?)(?:\.|,|$)", re.I),       "env_change"),
    (re.compile(r"(?:the (?:current|active) (?:environment|env|dir|directory|branch|project|workspace) is)\s+(.+?)(?:\.|,|$)", re.I), "env_context"),
    # Procedural patterns
    (re.compile(r"(?:to (?:fix|solve|handle|do|implement|run|search|find|create|build))\s+(.+?),?\s+(?:I|you should|we|use|run|try)\s+(.+?)(?:\.|$)", re.I), "procedure"),
    (re.compile(r"(?:the (?:best|correct|right) way to)\s+(.+?)\s+is\s+(?:to\s+)?(.+?)(?:\.|$)", re.I), "procedure"),
    (re.compile(r"(?:workflow|steps?|process) (?:for|to)\s+(.+?):\s*(.+?)(?:\.|$)", re.I),        "procedure"),
    (re.compile(r"(?:always|when)\s+(.+?),\s+(?:use|run|call|execute)\s+(.+?)(?:\.|,|$)", re.I),  "procedure"),

    # ── Chinese / CJK patterns ────────────────────────────────────────────────
    # Chinese has NO spaces between words — patterns use Chinese punctuation
    # (。，！？；：) and end-of-string as delimiters instead of \s+.
    # CJK range: \u4e00-\u9fff (CJK Unified Ideographs)

    # Identity: 我叫X / 我是X / 我的名字是X
    (re.compile(r"(?:我叫|我的名字是)\s*(.{1,20}?)(?:[，。！\n]|$)"),             "identity"),
    # Work: 我在X工作 / 我在X上班 / 我是X（职位）
    (re.compile(r"我在([^\s，。！？]{2,30}?)(?:工作|上班)(?:[，。！\n]|$)"),      "work"),
    (re.compile(r"我(?:是|做)(?:一名|一个)?\s*([^\s，。！？]{2,20}?)(?:[，。！\n]|$)"), "work"),
    (re.compile(r"我的(?:工作|职位|职业|岗位)是\s*(.{1,30}?)(?:[，。！\n]|$)"),  "work"),
    # Location: 我住在X / 我在X住 / 我来自X / 我现在在X
    (re.compile(r"我(?:住在|居住在|定居在|在.{0,3}住|来自|生活在)\s*(.{1,30}?)(?:[，。！\n]|$)"), "location"),
    (re.compile(r"我(?:现在|目前)在\s*(.{1,20}?)(?:[，。！\n]|$)"),              "location"),
    # Education: 我在X读书 / 我毕业于X / 我的学校是X
    (re.compile(r"我(?:在|毕业于|就读于|读书于)\s*(.{2,30}?)(?:读书|学习|毕业)?(?:[，。！\n]|$)"), "personal"),
    (re.compile(r"我的学校(?:是|叫)\s*(.{2,30}?)(?:[，。！\n]|$)"),             "personal"),
    # Duration: 我已经X了Y年/月
    (re.compile(r"我(?:已经|已)?(.{2,20}?)[了]?\s*(\d+\s*(?:年|个月|周))(?:[，。！\n]|$)"), "personal"),
    # Preferences: 我喜欢X / 我不喜欢X / 我偏好X
    (re.compile(r"我(?:喜欢|热爱|爱好|偏好|不喜欢|讨厌)\s*(.{1,40}?)(?:[，。！\n]|$)"), "preference"),
    # Named relations: 我的X(关系词)叫Y / 我的朋友X
    (re.compile(r"我的(?:伴侣|妻子|丈夫|老婆|老公|姐姐|妹妹|哥哥|弟弟|朋友|同事|邻居)\s*(.{1,10}?)(?:[，。！是叫\n]|$)"), "personal"),
    # Reminders / rules (Chinese): 记住X / 注意X / 重要：X
    (re.compile(r"(?:记住|注意|重要)[：:]\s*(.{1,100}?)(?:[。！\n]|$)"),          "reminder"),
    (re.compile(r"(?:总是|始终|永远|绝对不|从不)\s*(.{1,60}?)(?:[，。！\n]|$)"),  "rule"),
    # Deadline: X截止日期是Y / Y之前完成X
    (re.compile(r"(?:截止|截止日期|deadline)[是为：:]\s*(.{1,30}?)(?:[，。！\n]|$)"), "deadline"),
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


# ── Layer 2: SimpleMem LLM gate (GLM-4-flash, ~200-500ms) ────────────────────

ZAI_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
LLM_TIMEOUT_S = 10.0
LLM_MAX_INPUT = 3000   # chars of conversation to send to LLM

# SimpleMem-aligned extraction prompt (bilingual):
# - Φ_coref: forbids all pronouns (he/she/it/they/this/that/我/他/她)
# - Φ_time:  converts relative time → absolute ISO 8601
# - Multi-view indexing: lossless_restatement + keywords + timestamp +
#   location + persons + entities + topic
# - Importance estimation (0.0-1.0): how significant/novel this fact is
_EXTRACT_PROMPT = """Extract all valuable information from the following conversation as structured memory units.

TODAY'S DATE: {today}

[Extraction Rules — SimpleMem Section 3.1]
1. PRONOUN PROHIBITION: Absolutely forbid pronouns (he/she/it/they/this/that/I/we/
   他/她/它/我/你/他们/这/那). Replace with specific names or entities.
2. ABSOLUTE TIME: Convert all relative time (yesterday/today/last week/明天/上周) to
   absolute ISO 8601 dates based on today's date ({today}).
3. LOSSLESS RESTATEMENT: Each fact must be complete, self-contained and independently
   understandable without the original conversation.
4. COMPLETE COVERAGE: Generate enough entries to capture ALL meaningful information.
   One fact per distinct piece of information.
5. IMPORTANCE: Score 0.1-1.0 (1.0 = critical long-term fact, 0.5 = useful context,
   0.1 = minor detail). Preferences/rules/identity score high (0.7-1.0).
   Tool uses/env context score medium (0.4-0.6).

[Output Format — JSON array]
[{{
  "lossless_restatement": "Complete unambiguous statement (no pronouns, no relative time)",
  "category": "one of the categories below",
  "keywords": ["keyword1", "keyword2"],
  "timestamp": "YYYY-MM-DDTHH:MM:SS or null",
  "location": "specific location or null",
  "persons": ["name1"],
  "entities": ["entity1"],
  "topic": "brief topic phrase",
  "importance": 0.75
}}]

[Categories]
identity | work | preference | location | personal | reminder | rule | decision |
context | tool_use | capability_gained | env_change | env_context | procedure

[Example]
Conversation: "user: I prefer using bun over npm. Always use bun for JS projects.
assistant: Got it. I'll use bun."
Output:
[{{
  "lossless_restatement": "The user prefers bun over npm as the JavaScript package manager and requires bun to be used for all JavaScript projects.",
  "category": "preference",
  "keywords": ["bun", "npm", "JavaScript", "package manager"],
  "timestamp": null,
  "location": null,
  "persons": [],
  "entities": ["bun", "npm"],
  "topic": "JavaScript package manager preference",
  "importance": 0.9
}}]

Return ONLY the JSON array. If nothing worth storing, return [].

[Conversation]
{conversation}"""


def _load_env_var(name: str) -> str:
    """Read a var from environment → .env file → return empty string."""
    if v := os.environ.get(name):
        return v
    local_env = Path(__file__).parent / ".env"
    if local_env.exists():
        for line in local_env.read_text().splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return ""


def _load_key() -> str:
    """Re-read on every call so key rotation is picked up without restart."""
    if k := _load_env_var("ZAI_API_KEY"):
        return k
    cfg = Path.home() / ".openclaw" / "openclaw.json"
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text())
            return data["models"]["providers"]["zai"]["apiKey"]
        except Exception:
            pass
    return ""


def _load_proxy() -> str | None:
    """Return proxy URL for ZAI requests, or None to use direct connection."""
    p = _load_env_var("ZAI_PROXY") or _load_env_var("HTTPS_PROXY") or _load_env_var("HTTP_PROXY")
    return p or None


def _parse_llm_json(raw: str) -> list:
    """Robustly parse JSON from LLM response (handles markdown code fences)."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    # Find JSON array
    start = raw.find("[")
    if start == -1:
        return []
    # Try from first '[' to last ']'
    end = raw.rfind("]")
    if end == -1:
        return []
    return json.loads(raw[start:end + 1])


async def _llm_extract(conversation_text: str) -> list[ExtractedFact]:
    """Layer 2: SimpleMem-style LLM extraction via GLM-4-flash.

    Produces pronoun-free, absolute-timestamp lossless_restatement entries
    matching SimpleMem Section 3.1 multi-view indexing format.
    """
    key = _load_key()
    if not key:
        log.debug("[extractor] no ZAI key, skipping LLM extraction")
        return []

    today = datetime.date.today().isoformat()
    prompt = _EXTRACT_PROMPT.format(
        today=today,
        conversation=conversation_text[:LLM_MAX_INPUT],
    )

    proxy = _load_proxy()
    max_retries = 2
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=LLM_TIMEOUT_S, proxy=proxy) as client:
                resp = await client.post(
                    ZAI_URL,
                    json={
                        "model": "glm-4-flash",
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are a professional memory extraction assistant. "
                                    "You extract structured, unambiguous information from conversations. "
                                    "Output valid JSON only."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 1500,
                        "temperature": 0.1,
                    },
                    headers={"Authorization": f"Bearer {key}"},
                )
                if resp.status_code != 200:
                    log.warning(f"[extractor] LLM returned {resp.status_code}")
                    return []

                raw = resp.json()["choices"][0]["message"]["content"].strip()
                items = _parse_llm_json(raw)

                if not isinstance(items, list):
                    if attempt < max_retries - 1:
                        continue
                    return []

                facts: list[ExtractedFact] = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    # SimpleMem uses "lossless_restatement" as the primary content field
                    content = (
                        item.get("lossless_restatement")
                        or item.get("content")
                        or ""
                    )
                    if not content or len(content.strip()) < 10:
                        continue

                    category = item.get("category", "general")
                    # Never store credentials from LLM extraction for safety
                    if category == "credential":
                        continue

                    importance = float(item.get("importance", 0.5))
                    importance = max(0.0, min(1.0, importance))

                    # A-MAC content type prior (arXiv:2603.04549):
                    # Category-based importance floors ensure high-value facts
                    # survive consolidation pruning. The LLM score is the ceiling;
                    # the floor prevents systematic under-scoring of critical facts.
                    _IMPORTANCE_FLOOR = {
                        "identity":   0.80,
                        "rule":       0.80,
                        "preference": 0.75,
                        "work":       0.65,
                        "personal":   0.65,
                        "reminder":   0.65,
                        "decision":   0.60,
                        "procedure":  0.55,
                        "location":   0.55,
                        "capability_gained": 0.55,
                        "env_change": 0.50,
                        "tool_use":   0.45,
                        "env_context":0.40,
                        "context":    0.35,
                        "general":    0.30,
                    }
                    floor = _IMPORTANCE_FLOOR.get(category, 0.35)
                    importance = max(importance, floor)

                    facts.append(ExtractedFact(
                        content=str(content).strip()[:500],
                        category=category,
                        confidence=0.9,   # LLM-extracted = higher confidence
                        source="llm",
                        keywords=item.get("keywords") or None,
                        persons=item.get("persons") or None,
                        entities=item.get("entities") or None,
                        topic=item.get("topic") or None,
                        location=item.get("location") or None,
                        importance=importance,
                    ))

                log.info(
                    f"[extractor] LLM produced {len(facts)} lossless entries "
                    f"(SimpleMem Φ_gate)"
                )
                return facts

        except json.JSONDecodeError as e:
            log.warning(f"[extractor] LLM returned non-JSON (attempt {attempt+1}): {e}")
            if attempt >= max_retries - 1:
                return []
        except Exception as e:
            log.warning(
                f"[extractor] LLM extraction failed (attempt {attempt+1}): "
                f"{type(e).__name__}: {e}"
            )
            if attempt >= max_retries - 1:
                return []

    return []


# ── Merge + dedup ─────────────────────────────────────────────────────────────

def _dedup(facts: list[ExtractedFact]) -> list[ExtractedFact]:
    """Deduplicate by content similarity (substring check for speed).

    LLM facts take priority (higher importance, richer metadata).
    Regex facts are kept only if they add unique information.
    """
    result: list[ExtractedFact] = []
    seen_contents: list[str] = []

    for fact in facts:
        content_lower = fact.content.lower()
        is_dup = False
        for existing in seen_contents:
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
    Hybrid extraction: regex (instant) + SimpleMem LLM gate (async).

    Layer 2 implements SimpleMem's Semantic Structured Compression:
    - Φ_coref: pronoun resolution (no he/she/it/they)
    - Φ_time:  relative → absolute timestamp conversion
    - I(m_k):  multi-view indexing (lossless_restatement + keywords +
                                     timestamp + location + persons + entities + topic)

    Falls back to regex-only if LLM fails.
    """
    # Layer 1: regex (~1ms)
    # Run regex across ALL roles — assistant messages contain key facts too
    # (e.g. "I'm a marine biologist", "I bought a condo in Palo Alto")
    all_text = " ".join(
        m["content"] if isinstance(m["content"], str)
        else " ".join(b.get("text", "") for b in m["content"] if isinstance(b, dict))
        for m in messages
    )
    regex_facts = _regex_extract(all_text)

    # Layer 2: SimpleMem LLM gate (~200-500ms, background)
    llm_input = conversation_text or all_text
    llm_facts = await _llm_extract(llm_input)

    # Merge: LLM first (lossless, pronoun-free, richer metadata),
    # then regex additions only where LLM missed them.
    merged = llm_facts + regex_facts
    deduped = _dedup(merged)

    log.info(
        f"[extractor] regex={len(regex_facts)} llm={len(llm_facts)} "
        f"merged={len(deduped)} (SimpleMem Φ_gate active)"
    )

    return deduped
