"""
Retrieval Service - Multi-source memory search and fusion.

Handles hybrid retrieval from:
- Vector similarity search (semantic)
- BM25 keyword search (exact matching)
- Knowledge graph traversal (relational)
- Session context (temporal)

Implements SimpleMem Section 3.3: Intent-Aware Retrieval Planning.
"""

import asyncio
import logging
from typing import Any

from config.settings import settings
from exceptions import RetrievalError, VectorSearchError, BM25SearchError


class RetrievalService:
    """Service for multi-source memory retrieval with token-budgeted fusion."""
    
    def __init__(self, redis_client, embedder, bm25_index, store_module, graph_module=None):
        """Initialize retrieval service with dependencies.
        
        Args:
            redis_client: Redis connection
            embedder: Embedding model wrapper
            bm25_index: BM25 in-memory index
            store_module: Core store module
            graph_module: Optional knowledge graph module
        """
        self._redis = redis_client
        self._embedder = embedder
        self._bm25 = bm25_index
        self._store = store_module
        self._graph = graph_module
        self._log = logging.getLogger("mem")
    
    async def retrieve(
        self,
        query: str,
        session_id: str,
        memory_limit_number: int = 6,
        token_budget: int | None = None,
        include_tools: bool = True,
        include_procedures: bool = False,
        include_graph: bool = False,
        auto_graph: bool = True,
        enable_planning: bool = False,
        enable_reflection: bool = False,
        time_from: int | None = None,
        time_to: int | None = None,
    ) -> dict:
        """Retrieve relevant memories with token-budgeted fusion.
        
        Args:
            query: User query text
            session_id: Current session ID
            memory_limit_number: Number of memories to retrieve
            token_budget: Max tokens in output (defaults to settings.default_token_budget)
            include_tools: Whether to include tool context
            include_procedures: Whether to include procedure context
            include_graph: Whether to include graph neighbors
            auto_graph: Auto-trigger graph expansion for named entities
            enable_planning: Enable query planning (no-op if unsupported)
            enable_reflection: Enable reflection pass (no-op if unsupported)
            time_from: Filter facts created after this timestamp (ms)
            time_to: Filter facts created before this timestamp (ms)
        
        Returns:
            Dict with formatted context sections and candidate memories
        
        Raises:
            RetrievalError: If retrieval fails
        """
        budget = token_budget if token_budget is not None else settings.default_token_budget
        num_facts = max(1, memory_limit_number)
        num_episodes = max(2, num_facts // 2)
        filter_expr = self._build_time_filter(time_from, time_to)
        
        try:
            query_vec = await asyncio.to_thread(self._embedder.embed, query)
            
            # Parallel retrieval from multiple sources
            vector_facts_task = self._vector_search(query_vec, num_facts, filter_expr)
            vector_episodes_task = self._episode_search(query_vec, num_episodes, filter_expr)
            session_ctx_task = self._get_session_context(session_id)
            persona_ctx_task = self._get_persona_context()
            bm25_task = self._bm25_search(query)
            
            vector_results, episode_results, session_ctx, persona_ctx, bm25_results = await asyncio.gather(
                vector_facts_task,
                vector_episodes_task,
                session_ctx_task,
                persona_ctx_task,
                bm25_task,
                return_exceptions=True,
            )
            
            if isinstance(vector_results, Exception):
                self._log.warning(f"[retrieval] vector fact search failed: {vector_results}")
                vector_results = []
            if isinstance(episode_results, Exception):
                self._log.warning(f"[retrieval] vector episode search failed: {episode_results}")
                episode_results = []
            if isinstance(session_ctx, Exception):
                self._log.warning(f"[retrieval] session context failed: {session_ctx}")
                session_ctx = ""
            if isinstance(persona_ctx, Exception):
                self._log.warning(f"[retrieval] persona context failed: {persona_ctx}")
                persona_ctx = ""
            if isinstance(bm25_results, Exception):
                self._log.warning(f"[retrieval] BM25 search failed: {bm25_results}")
                bm25_results = []
            
            fused_facts = self._fuse_and_rank(vector_results, bm25_results)
            
            graph_neighbors = []
            if include_graph or (auto_graph and self._has_named_entities(query)):
                graph_neighbors = await self._expand_graph(query)
            
            formatted = self._format_context(
                facts=fused_facts,
                session_ctx=session_ctx,
                persona_ctx=persona_ctx,
                graph_neighbors=graph_neighbors,
                budget=budget,
            )
            
            return {
                "query": query,
                "session_id": session_id,
                "memory_limit_number": memory_limit_number,
                "token_budget": budget,
                "sections": formatted["sections"],
                "total_tokens": formatted["total_tokens"],
                "budget": formatted["budget"],
                "utilization": formatted["utilization"],
                "fact_count": formatted["fact_count"],
                "facts": fused_facts[:num_facts],
                "episodes": episode_results,
                "graph_neighbors": graph_neighbors,
            }
        except Exception as e:
            raise RetrievalError(
                message=f"Retrieval failed: {e}",
                query=query,
                session_id=session_id,
                cause=e,
            )
    
    def _build_time_filter(
        self,
        time_from: int | None,
        time_to: int | None,
    ) -> str | None:
        filters: list[str] = []
        if time_from is not None:
            filters.append(f".ts >= {time_from}")
        if time_to is not None:
            filters.append(f".ts <= {time_to}")
        return " && ".join(filters) if filters else None
    
    async def _vector_search(
        self,
        query_vec: Any,
        k: int,
        filter_expr: str | None = None,
    ) -> list[dict]:
        """Perform vector similarity search on fact embeddings."""
        try:
            results = await self._store.knn_search(
                self._redis,
                self._store.FACT_KEY,
                query_vec,
                k,
                filter_expr=filter_expr,
                bump_heat=True,
            )
            return results
        except Exception as e:
            raise VectorSearchError(
                message=f"Vector search failed: {e}",
                query=str(query_vec)[:100],
                cause=e,
            )
    
    async def _episode_search(
        self,
        query_vec: Any,
        k: int,
        filter_expr: str | None = None,
    ) -> list[dict]:
        """Perform vector similarity search over episodic memory."""
        try:
            results = await self._store.knn_search(
                self._redis,
                self._store.EPISODE_KEY,
                query_vec,
                k,
                filter_expr=filter_expr,
                bump_heat=False,
            )
            return results
        except Exception as e:
            self._log.warning(f"[retrieval] episode search failed: {e}")
            return []
    
    async def _bm25_search(self, query: str) -> list[dict]:
        """Perform BM25 keyword search for exact term matching."""
        if not self._bm25:
            return []
        try:
            return await self._bm25.search(query, k=settings.bm25_top_k)
        except Exception as e:
            self._log.warning(f"[retrieval] BM25 search failed: {e}")
            return []
    
    async def _get_session_context(self, session_id: str) -> str:
        try:
            ctx = await self._store.get_session_context(self._redis, session_id)
            return ctx or ""
        except Exception as e:
            self._log.warning(f"[retrieval] failed to get session context: {e}")
            return ""
    
    async def _get_persona_context(self) -> str:
        try:
            from core.persona import get_context
            return await get_context(self._redis)
        except Exception as e:
            self._log.warning(f"[retrieval] failed to get persona context: {e}")
            return ""
    
    def _fuse_and_rank(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
    ) -> list[dict]:
        fact_scores: dict[str, float] = {}
        fact_data: dict[str, dict] = {}
        
        for rank, result in enumerate(vector_results, 1):
            fact_id = result.get("_element") or result.get("uid") or result.get("id")
            if not fact_id:
                continue
            rrf_score = 1.0 / (settings.rrf_k_constant + rank)
            fact_scores[fact_id] = fact_scores.get(fact_id, 0.0) + rrf_score
            fact_data[fact_id] = result
        
        for rank, result in enumerate(bm25_results, 1):
            fact_id = result.get("_element") or result.get("uid") or result.get("id")
            if not fact_id:
                continue
            rrf_score = 1.0 / (settings.rrf_k_constant + rank)
            fact_scores[fact_id] = fact_scores.get(fact_id, 0.0) + rrf_score
            if fact_id not in fact_data:
                fact_data[fact_id] = result
        
        ranked_ids = sorted(fact_scores, key=lambda item: fact_scores[item], reverse=True)
        fused = []
        for fact_id in ranked_ids[:settings.retrieval_max_facts]:
            fact = fact_data[fact_id].copy()
            fact["fused_score"] = fact_scores[fact_id]
            fused.append(fact)
        return fused
    
    async def _expand_graph(self, query: str) -> list[dict]:
        if not self._graph:
            return []
        try:
            entities = await self._extract_entities(query)
            if not entities:
                return []
            neighbors = await self._graph.get_neighbors(
                self._redis,
                entities,
                max_depth=settings.graph_expansion_depth,
                max_neighbors=settings.graph_max_neighbors,
            )
            return neighbors
        except Exception as e:
            self._log.warning(f"[retrieval] graph expansion failed: {e}")
            return []
    
    async def _extract_entities(self, query: str) -> list[str]:
        import re
        entities = []
        entities.extend(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query))
        entities.extend(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", query))
        return list(dict.fromkeys(entities))
    
    def _has_named_entities(self, query: str) -> bool:
        import re
        # Match multi-word capitalized sequences (e.g., "New York", "Redis Cache")
        # or dates — skip single common capitalized words like "I", "The"
        if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", query):
            return True
        return bool(re.search(r"\b\d{4}-\d{2}-\d{2}\b", query))
    
    def _format_context(
        self,
        facts: list[dict],
        session_ctx: str,
        persona_ctx: str,
        graph_neighbors: list[dict],
        budget: int,
    ) -> dict:
        from utils.text_processing import estimate_tokens
        sections: dict[str, str] = {}
        total_tokens = 0
        
        if session_ctx:
            session_tokens = estimate_tokens(session_ctx)
            if total_tokens + session_tokens <= budget * 0.3:
                sections["session_context"] = session_ctx
                total_tokens += session_tokens
        
        if persona_ctx:
            persona_tokens = estimate_tokens(persona_ctx)
            if total_tokens + persona_tokens <= budget * 0.2:
                sections["persona_context"] = persona_ctx
                total_tokens += persona_tokens
        
        fact_texts = []
        remaining_budget = budget - total_tokens
        for fact in facts:
            fact_text = fact.get("content", "")
            fact_tokens = estimate_tokens(fact_text)
            if total_tokens + fact_tokens <= remaining_budget:
                fact_texts.append(fact_text)
                total_tokens += fact_tokens
            else:
                break
        if fact_texts:
            sections["relevant_facts"] = "\n".join(fact_texts)
        
        if graph_neighbors:
            graph_texts = [n.get("content", "") for n in graph_neighbors[:5]]
            graph_tokens = sum(estimate_tokens(t) for t in graph_texts)
            if total_tokens + graph_tokens <= budget:
                sections["graph_context"] = "\n".join(graph_texts)
                total_tokens += graph_tokens
        
        return {
            "sections": sections,
            "total_tokens": total_tokens,
            "budget": budget,
            "utilization": total_tokens / budget if budget > 0 else 0,
            "fact_count": len(fact_texts),
        }
