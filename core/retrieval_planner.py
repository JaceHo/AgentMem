"""
Retrieval Planner — v0.9.1 (SimpleMem Section 3.3)

Implements SimpleMem's Intent-Aware Retrieval Planning:
  P(q, H) → {q_sem, q_lex, q_sym, d}

Two capabilities:
  1. plan_queries(query) → list[str]
     Analyzes the information requirements of a query and generates
     a minimal set of targeted search queries (usually 1-3) that
     together cover all information gaps. Fallback: [query].

  2. check_sufficiency(query, contexts) → bool
     Reflection loop: checks whether retrieved contexts are sufficient
     to answer the query. Used by the retrieval loop to decide whether
     a second pass of retrieval is needed.

Both functions have hard timeouts (3s) and fail gracefully to ensure
retrieval stays fast. Planning is opt-in via `enable_planning=True` in
RecallRequest.
"""

from __future__ import annotations

import json
import logging

from .http import async_post_json
from .extractor import (
    AISERV_URL, AISERV_KEY, _resolve_nlp_model, _resolve_qa_model,
    AISERV_FALLBACK_MODEL, _parse_llm_json,
)

log = logging.getLogger("mem")

_PLAN_TIMEOUT_S = 8.0               # match extractor timeout for working models
_PLAN_MAX_RETRIES = 3               # try up to 3 distinct models before giving up

# Simple circuit breaker: models that failed recently are skipped for 5 minutes
_model_fail_times: dict[str, float] = {}
_CIRCUIT_BREAKER_TTL = 300  # seconds


def _is_model_broken(model: str) -> bool:
    """Check if a model is in circuit-breaker cooldown."""
    import time as _time
    fail_time = _model_fail_times.get(model)
    if fail_time is None:
        return False
    if _time.time() - fail_time > _CIRCUIT_BREAKER_TTL:
        del _model_fail_times[model]
        return False
    return True


def _mark_model_failed(model: str) -> None:
    """Record that a model failed, tripping its circuit breaker."""
    import time as _time
    _model_fail_times[model] = _time.time()


async def _llm_call(prompt: str, system: str, max_tokens: int = 300) -> str | None:
    """Call LLM via aiserv role-based routing. Returns raw text or None on failure.

    Uses /v1/role/nlp for dynamic model selection with exclude-rotation on failure.
    Skips models that failed recently (circuit breaker).
    """
    import time as _time
    now = _time.time()
    # Prune expired circuit breaker entries
    expired = [m for m, t in _model_fail_times.items() if now - t > _CIRCUIT_BREAKER_TTL]
    for m in expired:
        del _model_fail_times[m]

    exclude = None
    tried = []
    models_to_try = []
    for _ in range(_PLAN_MAX_RETRIES):
        model, _ = await _resolve_nlp_model(exclude=exclude)
        if model not in models_to_try:
            models_to_try.append(model)
        exclude = model
    # Also try QA role as additional fallback
    qa_model, _ = await _resolve_qa_model()
    if qa_model not in models_to_try:
        models_to_try.append(qa_model)
    if AISERV_FALLBACK_MODEL not in models_to_try:
        models_to_try.append(AISERV_FALLBACK_MODEL)

    # Filter out circuit-broken models
    models_to_try = [m for m in models_to_try if not _is_model_broken(m)]

    if not models_to_try:
        log.debug("[planner] all models circuit-broken, skipping planning")
        return None

    for model in models_to_try:
        if model in tried:
            continue
        tried.append(model)
        data = await async_post_json(
            AISERV_URL,
            payload={
                "model":      model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
            headers={"Authorization": f"Bearer {AISERV_KEY}"},
            timeout=_PLAN_TIMEOUT_S,
        )
        if data is not None:
            choices = data.get("choices", [])
            if choices:
                content = choices[0]["message"]["content"].strip()
                if content:
                    return content
        log.warning("[planner] %s returned no usable response", model)
        _mark_model_failed(model)  # trip circuit breaker
    log.warning("[planner] All models exhausted (%s)", ", ".join(tried))
    return None


async def plan_queries(query: str) -> list[str]:
    """
    SimpleMem P(q, H): analyze information requirements and generate targeted queries.

    Steps:
      1. Identify question type, key entities, and required information types.
      2. Generate minimal set of targeted search queries (1-3) that together
         cover all requirements.

    Always returns at least [query] on success or fallback.

    Optimized for speed: single LLM call, 4s timeout.
    """
    prompt = f"""Analyze this question and generate 1-3 minimal targeted search queries
to retrieve exactly the information needed to answer it. The queries should be focused,
non-redundant, and together cover all required information types.

Question: {query}

Return JSON only:
{{"reasoning": "brief strategy", "queries": ["query1", "query2"]}}

If the question is simple, just return the original query as the only item.
Always include the original question. Maximum 3 queries."""

    system = (
        "You are a search query planner. Generate minimal targeted queries. "
        "Output valid JSON only."
    )

    raw = await _llm_call(prompt, system, max_tokens=200)
    if not raw:
        return [query]

    try:
        data = _parse_llm_json(raw)
        if isinstance(data, list):
            # Raw array of strings
            queries = [q for q in data if isinstance(q, str) and q.strip()]
        elif isinstance(data, dict):
            queries = data.get("queries", [query])
        else:
            return [query]

        # Ensure original query is always included
        if query not in queries:
            queries.insert(0, query)

        # Cap at 3 targeted queries to keep retrieval fast
        queries = queries[:3]
        log.info(f"[planner] plan_queries: {len(queries)} targeted queries for: {query[:60]!r}")
        return queries

    except Exception as e:
        log.debug(f"[planner] plan_queries parse failed: {e}")
        return [query]


async def check_sufficiency(query: str, contexts: list[dict]) -> bool:
    """
    SimpleMem reflection loop: assess whether retrieved contexts are sufficient.

    Returns True  → contexts are sufficient, stop retrieval.
    Returns False → information is incomplete, trigger additional retrieval.

    Uses a fast LLM check (GLM-4-flash, 4s timeout). Defaults to True
    on any failure so retrieval doesn't loop indefinitely.
    """
    if not contexts:
        return False

    # Format contexts compactly
    context_lines = []
    for i, ctx in enumerate(contexts[:6], 1):
        content = ctx.get("content", "")[:200]
        context_lines.append(f"[{i}] {content}")
    context_str = "\n".join(context_lines)

    prompt = f"""Does the provided context contain sufficient information to answer the question?

Question: {query}

Context:
{context_str}

Answer JSON only:
{{"sufficient": true/false, "reason": "one sentence"}}"""

    system = "You are an information sufficiency evaluator. Output JSON only."

    raw = await _llm_call(prompt, system, max_tokens=80)
    if not raw:
        return True  # fail-safe: don't loop on LLM failure

    try:
        # Find JSON dict
        raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            return True
        data = json.loads(raw[start:end + 1])
        result = bool(data.get("sufficient", True))
        log.debug(f"[planner] sufficiency={result}: {data.get('reason', '')}")
        return result
    except Exception:
        return True  # fail-safe


async def analyze_query_structure(query: str) -> dict:
    """
    SimpleMem symbolic layer: extract persons, entities, time, location from query
    for structured (symbolic) retrieval.

    Returns dict with keys: keywords, persons, time_expression, location, entities.
    Returns empty defaults on failure.
    """
    prompt = f"""Extract structured information from this query for memory retrieval.

Query: {query}

Return JSON only:
{{"keywords": ["k1"], "persons": ["name1"], "time_expression": null, "location": null, "entities": ["e1"]}}"""

    system = "You are a query analyzer. Output JSON only."

    raw = await _llm_call(prompt, system, max_tokens=150)

    default = {
        "keywords":        [query],
        "persons":         [],
        "time_expression": None,
        "location":        None,
        "entities":        [],
    }

    if not raw:
        return default

    try:
        raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            return default
        data = json.loads(raw[start:end + 1])
        return {**default, **data}
    except Exception:
        return default


async def generate_retrieval_topic(query: str) -> str | None:
    """
    MIRIX Active Retrieval (arXiv:2507.07957 §3.2):
    Generate a focused information-retrieval topic from a vague or conversational query.

    Instead of searching with the raw query, generate a clean factual topic statement
    that the memory system should retrieve. Especially valuable for coding agent queries
    like "that auth bug" → "JWT token validation error in authentication middleware".

    Returns a refined topic string or None on failure.
    Timeout: 2s (faster than HyDE — shorter output).
    """
    prompt = (
        f"Convert this query into a focused information-retrieval topic for a developer memory system.\n"
        f"Output only the topic phrase (5-15 words), no explanation.\n\n"
        f"Query: {query}\n\nTopic:"
    )
    system = (
        "Convert developer queries to focused retrieval topics. "
        "Output only the topic phrase. Be specific and technical."
    )
    result = await _llm_call(prompt, system, max_tokens=40)
    if result:
        topic = result.strip().strip('"').strip("'")
        log.debug(f"[mirix] retrieval topic: {topic!r} for query: {query[:60]!r}")
        return topic if len(topic) > 4 else None
    return None


async def generate_hyde_doc(query: str) -> str | None:
    """
    HyDE (Hypothetical Document Embeddings, Gao et al. 2022):
    Generate a short hypothetical memory entry that would answer the query.
    Embedding the hypothetical doc is closer to actual stored memories than
    embedding the bare question, improving retrieval recall.

    Returns the hypothetical text or None on failure.
    Timeout: 3s. Fails silently so recall stays fast.
    """
    prompt = (
        f"Write a single concise factual statement (1-2 sentences) that would "
        f"be the ideal memory entry to answer this question:\n\n{query}\n\n"
        "Write only the memory statement, no preamble."
    )
    system = (
        "You generate hypothetical memory entries. Be concise and factual. "
        "Write as if this is a stored fact in a personal memory system."
    )
    result = await _llm_call(prompt, system, max_tokens=80)
    if result:
        log.debug(f"[hyde] hypothetical doc: {result[:80]!r}")
    return result
