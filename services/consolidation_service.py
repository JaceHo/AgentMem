"""
Consolidation Service - Memory lifecycle management.

Handles:
- A-MAC admission gate (fact quality scoring)
- Decay engine (time-based importance decay via Ebbinghaus curve)
- Merge strategy (near-duplicate resolution with LLM consolidation)
- Hard pruning (physical deletion of superseded facts)

Implements AgentMem's memory evolution pipeline.
Delegates to specialized lifecycle modules for core algorithms.
"""

import asyncio
import logging
import time
from typing import Any

from config.settings import settings
from exceptions import ConsolidationError, DecayError, MergeError, PruneError
from lifecycle import DecayEngine, MergeStrategy


class ConsolidationService:
    """Service for memory consolidation and lifecycle management."""
    
    def __init__(self, redis_client, embedder, store_module, llm_client=None):
        """Initialize consolidation service with lifecycle modules.
        
        Args:
            redis_client: Redis connection
            embedder: Embedding model wrapper
            store_module: Core store module
            llm_client: Optional LLM client for smart consolidation
        """
        self._redis = redis_client
        self._embedder = embedder
        self._store = store_module
        self._llm = llm_client
        self._log = logging.getLogger("mem")
        
        # Initialize lifecycle modules
        self._decay_engine = DecayEngine(log=self._log)
        self._merge_strategy = MergeStrategy(
            redis_client=redis_client,
            store_module=store_module,
            embedder=embedder,
            llm_client=llm_client,
            log=self._log,
        )

    async def consolidate(self, session_id: str | None = None) -> dict:
        """Run full consolidation pipeline.
        
        Steps:
        1. Apply time-based decay to all facts
        2. Detect and merge near-duplicates
        3. Remove low-importance facts below threshold
        4. Trigger hard prune if needed
        
        Args:
            session_id: Optional session filter (None = global consolidation)
            
        Returns:
            Dict with consolidation statistics
            
        Raises:
            ConsolidationError: If consolidation fails
        """
        start_time = time.time()
        
        try:
            stats = {
                "session_id": session_id or "global",
                "decay_applied": 0,
                "merges_performed": 0,
                "facts_pruned": 0,
                "duration_ms": 0,
            }
            
            # Step 1: Apply decay
            decay_count = await self._apply_decay(session_id)
            stats["decay_applied"] = decay_count
            
            # Step 2: Merge near-duplicates
            merge_count = await self._merge_near_duplicates(session_id)
            stats["merges_performed"] = merge_count
            
            # Step 3: Prune low-importance facts
            prune_count = await self._prune_low_importance(session_id)
            stats["facts_pruned"] = prune_count
            
            # Step 4: Update statistics
            stats["duration_ms"] = int((time.time() - start_time) * 1000)
            
            self._log.info(
                f"[consolidation] session={stats['session_id']} "
                f"decay={decay_count} merges={merge_count} pruned={prune_count} "
                f"duration={stats['duration_ms']}ms"
            )
            
            return stats
            
        except Exception as e:
            raise ConsolidationError(
                message=f"Consolidation failed: {e}",
                phase="full_consolidation",
                session_id=session_id,
                cause=e,
            )
    
    async def _apply_decay(self, session_id: str | None = None) -> int:
        """Apply time-based importance decay using Ebbinghaus Forgetting Curve.
        
        Delegates to DecayEngine for category-specific decay calculations.
        
        Args:
            session_id: Optional session filter
            
        Returns:
            Number of facts updated
            
        Raises:
            DecayError: If decay application fails
        """
        try:
            # Get all facts (optionally filtered by session)
            facts = await self._store.get_all_facts(self._redis, session_id=session_id)
            
            if not facts:
                return 0
            
            # Calculate decay for all facts using DecayEngine
            decay_results = self._decay_engine.batch_calculate_decay(facts)
            
            # Update facts with decayed importance
            updated_count = 0
            for result in decay_results:
                fact_id = result["fact_id"]
                decayed_importance = result["decayed_importance"]
                
                # Only update if significant change
                original = result["original_importance"]
                if abs(decayed_importance - original) > 0.01:
                    await self._store.update_fact_importance(
                        self._redis,
                        fact_id,
                        decayed_importance,
                    )
                    updated_count += 1
                    
                    # Log pruning candidates
                    if result["should_prune"]:
                        self._log.info(
                            f"[decay] fact {fact_id} below threshold: "
                            f"{decayed_importance:.3f} < {settings.prune_importance_threshold}"
                        )
            
            return updated_count
            
        except Exception as e:
            raise DecayError(
                message=f"Decay application failed: {e}",
                session_id=session_id,
                cause=e,
            )
    
    async def _merge_near_duplicates(self, session_id: str | None = None) -> int:
        """Detect and merge near-duplicate facts using MergeStrategy.
        
        Delegates to MergeStrategy for intelligent merging with temporal factors
        and optional LLM consolidation.
        
        Args:
            session_id: Optional session filter
            
        Returns:
            Number of merges performed
            
        Raises:
            MergeError: If merge operation fails
        """
        try:
            # Use MergeStrategy to detect and merge
            result = await self._merge_strategy.detect_and_merge(
                session_id=session_id,
                similarity_threshold=settings.dedup_similarity_threshold,
                use_llm_consolidation=True,  # Enable smart consolidation
            )
            
            return result["merges_performed"]
            
        except Exception as e:
            raise MergeError(
                message=f"Near-duplicate merge failed: {e}",
                session_id=session_id,
                phase="duplicate_detection",
                cause=e,
            )
    
    async def _prune_low_importance(self, session_id: str | None = None) -> int:
        """Remove facts with importance below threshold.
        
        Args:
            session_id: Optional session filter
            
        Returns:
            Number of facts pruned
            
        Raises:
            PruneError: If pruning fails
        """
        try:
            # Get all facts below importance threshold
            low_importance_facts = await self._store.get_facts_below_importance(
                self._redis,
                threshold=settings.prune_importance_threshold,
                session_id=session_id,
            )
            
            pruned_count = 0
            
            for fact in low_importance_facts:
                fact_id = fact.get("id") or fact.get("uid")
                if fact_id:
                    await self._store.hard_delete_fact(self._redis, fact_id)
                    pruned_count += 1
            
            return pruned_count
            
        except Exception as e:
            raise PruneError(
                message=f"Low-importance pruning failed: {e}",
                session_id=session_id,
                cause=e,
            )
    
    async def hard_prune(self, older_than_days: int | None = None) -> dict:
        """Perform hard physical deletion of old/superseded facts.
        
        This is different from soft delete - it permanently removes entries
        from Redis VLDB to reclaim storage.
        
        Args:
            older_than_days: Only prune facts older than this (None = use settings)
            
        Returns:
            Dict with prune statistics
        """
        cutoff_days = older_than_days or settings.hard_prune_older_than_days
        start_time = time.time()
        
        try:
            pruned_count = await self._store.hard_prune_old_facts(
                self._redis,
                older_than_days=cutoff_days,
            )
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            self._log.info(
                f"[hard_prune] removed {pruned_count} facts older than "
                f"{cutoff_days} days in {duration_ms}ms"
            )
            
            return {
                "pruned_count": pruned_count,
                "cutoff_days": cutoff_days,
                "duration_ms": duration_ms,
            }
            
        except Exception as e:
            self._log.error(f"[hard_prune] failed: {e}", exc_info=True)
            raise PruneError(
                message=f"Hard prune failed: {e}",
                phase="physical_deletion",
                cause=e,
            )
    
    async def evaluate_admission(self, fact_text: str, context: dict | None = None) -> dict:
        """Evaluate whether a fact should be admitted to memory (A-MAC gate).
        
        Implements AgentMem Section 3.2: Adaptive Memory Admission Control.
        
        Scoring factors:
        - Novelty (vs existing facts)
        - Specificity (concrete vs vague)
        - Relevance (to user/session context)
        - Actionability (can be acted upon)
        
        Args:
            fact_text: Fact text to evaluate
            context: Optional context dict (session_id, user_profile, etc.)
            
        Returns:
            Dict with admission decision and scores
        """
        try:
            scores = {}
            
            # Factor 1: Novelty (check against existing facts)
            similar_facts = await self._store.search_facts_by_text(
                self._redis,
                fact_text,
                top_k=5,
            )
            max_similarity = max(
                [f.get("score", 0) for f in similar_facts],
                default=0,
            )
            scores["novelty"] = 1.0 - max_similarity  # Higher = more novel
            
            # Factor 2: Specificity (length, named entities, numbers)
            import re
            word_count = len(fact_text.split())
            has_numbers = bool(re.search(r'\d+', fact_text))
            has_entities = bool(re.search(r'\b[A-Z][a-z]+\b', fact_text))
            scores["specificity"] = min(1.0, (word_count / 20) + (0.2 if has_numbers else 0) + (0.2 if has_entities else 0))
            
            # Factor 3: Relevance (context matching)
            if context:
                session_id = context.get("session_id", "")
                # TODO: Implement session-specific relevance scoring
                scores["relevance"] = 0.7  # Placeholder
            else:
                scores["relevance"] = 0.5  # Neutral without context
            
            # Factor 4: Actionability (contains verbs, imperatives)
            action_words = ['should', 'must', 'need', 'want', 'prefer', 'like', 'dislike']
            has_action = any(word in fact_text.lower() for word in action_words)
            scores["actionability"] = 0.8 if has_action else 0.4
            
            # Weighted composite score
            composite = (
                scores["novelty"] * settings.admission_weight_novelty +
                scores["specificity"] * settings.admission_weight_specificity +
                scores["relevance"] * settings.admission_weight_relevance +
                scores["actionability"] * settings.admission_weight_actionability
            )
            
            # Decision threshold
            admitted = composite >= settings.admission_threshold
            
            return {
                "admitted": admitted,
                "composite_score": composite,
                "threshold": settings.admission_threshold,
                "scores": scores,
                "reason": "above_threshold" if admitted else "below_threshold",
            }
            
        except Exception as e:
            self._log.warning(f"[admission] evaluation failed: {e}")
            # Default to admission on error (fail-open)
            return {
                "admitted": True,
                "composite_score": 0.5,
                "threshold": settings.admission_threshold,
                "scores": {},
                "reason": "error_fail_open",
            }
