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

import httpx

from extractor import (
    AISERV_URL, AISERV_KEY, LLM_TIMEOUT_S, LLM_MAX_RETRIES,
    _resolve_nlp_model, AISERV_FALLBACK_MODEL, _parse_llm_json,
)

log = logging.getLogger("mem")

_PLAN_TIMEOUT_S = 3.0   # planning generates fewer tokens; 3s hard cap keeps recall fast
_PLAN_MAX_RETRIES = 1   # single attempt — fail fast; retrying multiplies latency


async def _llm_call(prompt: str, system: str, max_tokens: int = 300) -> str | None:
    """Call LLM via aiserv role-based routing. Returns raw text or None on failure.

    Uses /v1/role/nlp for dynamic model selection with exclude-rotation on failure.
    """
    exclude = None
    tried = []
    for _ in range(_PLAN_MAX_RETRIES):
        model, _ = await _resolve_nlp_model(exclude=exclude)
        if model in tried:
            model = AISERV_FALLBACK_MODEL
        tried.append(model)
        try:
            async with httpx.AsyncClient(timeout=_PLAN_TIMEOUT_S) as client:
                resp = await client.post(
                    AISERV_URL,
                    json={
                        "model":      model,
                        "max_tokens": max_tokens,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ],
                    },
                    headers={"Authorization": f"Bearer {AISERV_KEY}"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0]["message"]["content"].strip()
                log.warning("[planner] %s returned %d", model, resp.status_code)
                exclude = model
        except httpx.TimeoutException:
            log.warning("[planner] %s timed out (%.0fs)", model, _PLAN_TIMEOUT_S)
            exclude = model
        except Exception as e:
            log.debug("[planner] %s failed: %s: %s", model, type(e).__name__, e)
            exclude = model
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
