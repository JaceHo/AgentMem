"""
Merge Strategy - Near-duplicate detection and consolidation.

Implements AgentMem memory specification for intelligent merging:
- Cosine similarity × temporal factor ≥ 0.85 triggers LLM consolidation
- Keeps higher importance fact, marks other as superseded
- Preserves temporal context in merged facts
- Supports batch merging for efficiency

Reference: PHASE1_COMPLETE.md, Agent Memory Lifecycle Management Specification
"""

import asyncio
import logging
import time
from typing import Any

from config.settings import settings
from exceptions import MergeError


class MergeStrategy:
    """Intelligent near-duplicate fact merging with LLM consolidation."""
    
    def __init__(
        self,
        redis_client,
        store_module,
        embedder=None,
        llm_client=None,
        log: logging.Logger | None = None,
    ):
        """Initialize merge strategy.
        
        Args:
            redis_client: Redis connection
            store_module: Core store module
            embedder: Optional embedding model wrapper
            llm_client: Optional LLM client for consolidation
            log: Optional logger instance
        """
        self._redis = redis_client
        self._store = store_module
        self._embedder = embedder
        self._llm = llm_client
        self._log = log or logging.getLogger("mem.merge")
    
    async def detect_and_merge(
        self,
        session_id: str | None = None,
        similarity_threshold: float | None = None,
        use_llm_consolidation: bool = True,
    ) -> dict:
        """Detect near-duplicates and merge them.
        
        Args:
            session_id: Optional session filter (None = global)
            similarity_threshold: Similarity threshold (defaults to settings.dedup_similarity_threshold)
            use_llm_consolidation: Whether to use LLM for smart consolidation
            
        Returns:
            Dict with merge statistics
            
        Raises:
            MergeError: If merge operation fails
        """
        threshold = similarity_threshold or settings.dedup_similarity_threshold
        start_time = time.time()
        
        try:
            # Get all active facts
            facts = await self._store.get_all_facts(
                self._redis,
                session_id=session_id,
                exclude_superseded=True,
            )
            
            if len(facts) < 2:
                return {
                    "merges_performed": 0,
                    "facts_scanned": len(facts),
                    "duration_ms": 0,
                    "reason": "insufficient_facts",
                }
            
            # Identify pairs to merge
            pairs_to_merge = await self._identify_merge_candidates(facts, threshold)
            
            if not pairs_to_merge:
                return {
                    "merges_performed": 0,
                    "facts_scanned": len(facts),
                    "duration_ms": int((time.time() - start_time) * 1000),
                    "reason": "no_duplicates_found",
                }
            
            # Perform merges
            merge_count = 0
            for fact_a_id, fact_b_id, similarity in pairs_to_merge:
                success = await self._merge_fact_pair(
                    fact_a_id,
                    fact_b_id,
                    similarity,
                    use_llm_consolidation,
                )
                if success:
                    merge_count += 1
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            self._log.info(
                f"[merge] scanned={len(facts)} candidates={len(pairs_to_merge)} "
                f"merged={merge_count} duration={duration_ms}ms"
            )
            
            return {
                "merges_performed": merge_count,
                "facts_scanned": len(facts),
                "candidates_found": len(pairs_to_merge),
                "similarity_threshold": threshold,
                "duration_ms": duration_ms,
            }
            
        except Exception as e:
            raise MergeError(
                message=f"Merge operation failed: {e}",
                phase="detect_and_merge",
                session_id=session_id,
                cause=e,
            )
    
    async def _identify_merge_candidates(
        self,
        facts: list[dict],
        threshold: float,
    ) -> list[tuple[str, str, float]]:
        """Identify fact pairs that should be merged.
        
        Uses cosine similarity with temporal decay factor.
        
        Args:
            facts: List of fact dicts
            threshold: Similarity threshold
            
        Returns:
            List of (fact_id_a, fact_id_b, similarity) tuples
        """
        candidates = []
        current_time = time.time()
        
        # Compare all pairs
        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                fact_a = facts[i]
                fact_b = facts[j]
                
                # Skip if either already superseded
                if fact_a.get("superseded_by") or fact_b.get("superseded_by"):
                    continue
                
                # Calculate base similarity
                similarity = self._calculate_similarity(fact_a, fact_b)
                
                # Apply temporal factor (recent facts weighted higher)
                created_a = fact_a.get("created_at", current_time)
                created_b = fact_b.get("created_at", current_time)
                age_factor = self._calculate_temporal_factor(created_a, created_b, current_time)
                
                # Combined score
                combined_score = similarity * age_factor
                
                if combined_score >= threshold:
                    id_a = fact_a.get("id") or fact_a.get("uid")
                    id_b = fact_b.get("id") or fact_b.get("uid")
                    if id_a and id_b:
                        candidates.append((id_a, id_b, combined_score))
        
        # Sort by similarity (highest first)
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        return candidates
    
    def _calculate_similarity(self, fact_a: dict, fact_b: dict) -> float:
        """Calculate cosine similarity between two facts.
        
        Args:
            fact_a: First fact dict
            fact_b: Second fact dict
            
        Returns:
            Cosine similarity (0.0 to 1.0)
        """
        # Try embedding-based similarity first
        vec_a = fact_a.get("embedding")
        vec_b = fact_b.get("embedding")
        
        if vec_a and vec_b:
            return self._cosine_similarity(vec_a, vec_b)
        
        # Fallback to text-based Jaccard similarity
        text_a = set(fact_a.get("text", "").lower().split())
        text_b = set(fact_b.get("text", "").lower().split())
        
        if not text_a or not text_b:
            return 0.0
        
        intersection = text_a & text_b
        union = text_a | text_b
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_temporal_factor(
        self,
        created_a: float,
        created_b: float,
        current_time: float,
    ) -> float:
        """Calculate temporal weighting factor.
        
        More recent facts get higher weight to prefer keeping newer information.
        
        Args:
            created_a: Creation timestamp of fact A
            created_b: Creation timestamp of fact B
            current_time: Current timestamp
            
        Returns:
            Temporal factor (0.5 to 1.0)
        """
        age_a = current_time - created_a
        age_b = current_time - created_b
        
        # Prefer newer facts (lower age = higher weight)
        # Normalize to [0.5, 1.0] range
        max_age = max(age_a, age_b, 1)  # Avoid division by zero
        min_age = min(age_a, age_b)
        
        # Factor closer to 1.0 when ages are similar or both recent
        factor = 0.5 + (min_age / max_age) * 0.5
        
        return factor
    
    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(x * x for x in vec_a))
        norm_b = math.sqrt(sum(x * x for x in vec_b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    async def _merge_fact_pair(
        self,
        fact_a_id: str,
        fact_b_id: str,
        similarity: float,
        use_llm: bool = True,
    ) -> bool:
        """Merge a pair of near-duplicate facts.
        
        Strategy:
        1. Keep fact with higher importance
        2. Optionally use LLM to consolidate content
        3. Mark other as superseded with reference to keeper
        
        Args:
            fact_a_id: ID of first fact
            fact_b_id: ID of second fact
            similarity: Similarity score
            use_llm: Whether to use LLM for consolidation
            
        Returns:
            True if merge successful
        """
        try:
            # Fetch both facts
            fact_a = await self._store.get_fact_by_id(self._redis, fact_a_id)
            fact_b = await self._store.get_fact_by_id(self._redis, fact_b_id)
            
            if not fact_a or not fact_b:
                self._log.warning(f"[merge] fact not found: {fact_a_id} or {fact_b_id}")
                return False
            
            # Determine keeper (higher importance wins)
            importance_a = fact_a.get("importance", 0.5)
            importance_b = fact_b.get("importance", 0.5)
            
            if importance_a >= importance_b:
                keeper = fact_a
                superseded = fact_b
                keeper_id = fact_a_id
                superseded_id = fact_b_id
            else:
                keeper = fact_b
                superseded = fact_a
                keeper_id = fact_b_id
                superseded_id = fact_a_id
            
            # Optionally consolidate content with LLM
            if use_llm and self._llm:
                consolidated_text = await self._llm_consolidate(
                    keeper.get("text", ""),
                    superseded.get("text", ""),
                )
                if consolidated_text:
                    # Update keeper with consolidated content
                    await self._store.update_fact_text(
                        self._redis,
                        keeper_id,
                        consolidated_text,
                    )
            
            # Mark superseded fact
            await self._store.soft_delete_fact(
                self._redis,
                superseded_id,
                reason="merged_duplicate",
                superseded_by=keeper_id,
            )
            
            self._log.info(
                f"[merge] {superseded_id} → {keeper_id} "
                f"(similarity={similarity:.3f}, importance_keeper={importance_a if importance_a >= importance_b else importance_b:.3f})"
            )
            
            return True
            
        except Exception as e:
            self._log.error(f"[merge] failed to merge {fact_a_id} and {fact_b_id}: {e}")
            return False
    
    async def _llm_consolidate(self, text_a: str, text_b: str) -> str | None:
        """Use LLM to intelligently consolidate two similar texts.
        
        Args:
            text_a: First text
            text_b: Second text
            
        Returns:
            Consolidated text, or None if consolidation failed
        """
        if not self._llm:
            return None
        
        try:
            prompt = f"""Consolidate these two similar facts into one comprehensive fact.
Preserve all unique information from both. Remove redundancy.

Fact 1: {text_a}

Fact 2: {text_b}

Consolidated fact:"""
            
            response = await self._llm.generate(prompt, max_tokens=200)
            consolidated = response.strip()
            
            # Validate consolidation quality
            if len(consolidated) < 10:
                self._log.warning("[merge] LLM consolidation too short, using original")
                return None
            
            return consolidated
            
        except Exception as e:
            self._log.warning(f"[merge] LLM consolidation failed: {e}")
            return None
    
    async def merge_batch(
        self,
        fact_pairs: list[tuple[str, str, float]],
        use_llm: bool = True,
        max_concurrent: int = 5,
    ) -> dict:
        """Merge multiple fact pairs concurrently.
        
        Args:
            fact_pairs: List of (fact_id_a, fact_id_b, similarity) tuples
            use_llm: Whether to use LLM consolidation
            max_concurrent: Maximum concurrent merge operations
            
        Returns:
            Dict with batch merge statistics
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        
        async def merge_with_semaphore(pair):
            async with semaphore:
                fact_a_id, fact_b_id, similarity = pair
                success = await self._merge_fact_pair(
                    fact_a_id, fact_b_id, similarity, use_llm
                )
                return {
                    "pair": pair,
                    "success": success,
                }
        
        # Execute merges concurrently
        tasks = [merge_with_semaphore(pair) for pair in fact_pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate statistics
        successful = sum(1 for r in results if isinstance(r, dict) and r["success"])
        failed = len(results) - successful
        
        return {
            "total_pairs": len(fact_pairs),
            "successful_merges": successful,
            "failed_merges": failed,
            "success_rate": successful / len(fact_pairs) if fact_pairs else 0,
        }
