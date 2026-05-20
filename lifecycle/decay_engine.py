"""
Decay Engine - Implements Ebbinghaus Forgetting Curve for memory importance.

Based on AgentMem memory specification:
- Importance decays exponentially: importance × 0.9 per 90 days
- Category-specific stability factors (identity/rules: 365d, bugs: 30d)
- Reinforcement resets decay curve
- Confidence tracking with exponential decay

Reference: PHASE1_COMPLETE.md, Agent Memory Lifecycle Management Specification
"""

import logging
import math
import time
from enum import Enum
from typing import Any

from config.settings import settings


class FactCategory(str, Enum):
    """Memory fact categories with different decay rates."""
    IDENTITY = "identity"           # User preferences, identity (365d stability)
    RULES = "rules"                 # Rules, constraints (365d stability)
    BUGS = "bugs"                   # Bug fixes, workarounds (30d stability)
    PROCEDURES = "procedures"       # How-to procedures (90d stability)
    FACTS = "facts"                 # General facts (90d stability)
    PREFERENCES = "preferences"     # User preferences (180d stability)
    CONTEXT = "context"             # Session context (7d stability)


# Category-specific stability periods (days)
CATEGORY_STABILITY = {
    FactCategory.IDENTITY: 365,
    FactCategory.RULES: 365,
    FactCategory.BUGS: 30,
    FactCategory.PROCEDURES: 90,
    FactCategory.FACTS: 90,
    FactCategory.PREFERENCES: 180,
    FactCategory.CONTEXT: 7,
}


class DecayEngine:
    """Implements Ebbinghaus Forgetting Curve for memory importance decay.
    
    The forgetting curve models how information is lost over time when there
    is no attempt to retain it. Each reinforcement event resets the curve.
    
    Formula: confidence(t) = initial_confidence × exp(-t / stability_period)
    where t is time since last access/reinforcement.
    """
    
    def __init__(self, log: logging.Logger | None = None):
        """Initialize decay engine.
        
        Args:
            log: Optional logger instance
        """
        self._log = log or logging.getLogger("mem.decay")
    
    def calculate_decay(
        self,
        initial_importance: float,
        category: FactCategory | str,
        age_days: float,
        last_accessed_days_ago: float | None = None,
        reinforcement_count: int = 0,
    ) -> dict[str, Any]:
        """Calculate decayed importance using Ebbinghaus forgetting curve.
        
        Args:
            initial_importance: Original importance score (0.0 to 1.0)
            category: Fact category (affects decay rate)
            age_days: Days since creation
            last_accessed_days_ago: Days since last access (None = never accessed)
            reinforcement_count: Number of times reinforced
            
        Returns:
            Dict with decay calculation details
        """
        # Normalize category
        if isinstance(category, str):
            try:
                category = FactCategory(category)
            except ValueError:
                category = FactCategory.FACTS  # Default
        
        # Get stability period for category
        stability_period = CATEGORY_STABILITY.get(category, 90)
        
        # Calculate base decay factor (Ebbinghaus formula)
        # confidence(t) = initial × exp(-t / stability)
        if last_accessed_days_ago is not None:
            # Use last access time if available
            decay_factor = math.exp(-last_accessed_days_ago / stability_period)
        else:
            # Otherwise use age
            decay_factor = math.exp(-age_days / stability_period)
        
        # Apply reinforcement boost (each reinforcement extends stability)
        reinforcement_boost = 1.0 + (reinforcement_count * 0.1)  # 10% per reinforcement
        reinforcement_boost = min(reinforcement_boost, 2.0)  # Cap at 2x
        
        # Calculate decayed importance
        decayed_importance = initial_importance * decay_factor * reinforcement_boost
        
        # Clamp to valid range [0.0, 1.0]
        decayed_importance = max(0.0, min(1.0, decayed_importance))
        
        return {
            "original_importance": initial_importance,
            "decayed_importance": round(decayed_importance, 4),
            "decay_factor": round(decay_factor, 4),
            "category": category.value,
            "stability_period_days": stability_period,
            "age_days": round(age_days, 2),
            "last_accessed_days_ago": last_accessed_days_ago,
            "reinforcement_count": reinforcement_count,
            "reinforcement_boost": round(reinforcement_boost, 2),
            "should_prune": decayed_importance < settings.prune_importance_threshold,
        }
    
    def calculate_confidence(
        self,
        initial_confidence: float,
        category: FactCategory | str,
        age_days: float,
        accesses: int = 0,
    ) -> float:
        """Calculate current confidence score with exponential decay.
        
        Args:
            initial_confidence: Initial confidence (0.0 to 1.0)
            category: Fact category
            age_days: Days since creation
            accesses: Number of times accessed (affects confidence)
            
        Returns:
            Current confidence score (0.0 to 1.0)
        """
        # Normalize category
        if isinstance(category, str):
            try:
                category = FactCategory(category)
            except ValueError:
                category = FactCategory.FACTS
        
        # Get stability period
        stability_period = CATEGORY_STABILITY.get(category, 90)
        
        # Base exponential decay
        confidence = initial_confidence * math.exp(-age_days / stability_period)
        
        # Access bonus (each access slightly boosts confidence)
        access_bonus = min(accesses * 0.02, 0.2)  # Max 20% boost
        confidence += access_bonus
        
        # Clamp to valid range
        return max(0.0, min(1.0, confidence))
    
    def apply_user_feedback(
        self,
        current_importance: float,
        feedback_score: float,  # 1-5 rating
        feedback_type: str = "rating",  # "rating", "useful", "not_useful"
    ) -> float:
        """Adjust importance based on user feedback.
        
        Args:
            current_importance: Current importance score
            feedback_score: User feedback (1-5 for rating, or binary)
            feedback_type: Type of feedback
            
        Returns:
            Adjusted importance score
        """
        if feedback_type == "rating":
            # Convert 1-5 rating to multiplier (0.5 to 1.5)
            multiplier = 0.5 + (feedback_score / 5.0)
        elif feedback_type == "useful":
            multiplier = 1.2  # Boost for useful feedback
        elif feedback_type == "not_useful":
            multiplier = 0.7  # Reduce for not useful
        else:
            multiplier = 1.0
        
        adjusted = current_importance * multiplier
        return max(0.0, min(1.0, adjusted))
    
    def get_reinforcement_schedule(self, category: FactCategory | str) -> dict:
        """Get recommended reinforcement schedule for category.
        
        Based on spaced repetition principles: reinforce before significant decay.
        
        Args:
            category: Fact category
            
        Returns:
            Dict with reinforcement recommendations
        """
        # Normalize category
        if isinstance(category, str):
            try:
                category = FactCategory(category)
            except ValueError:
                category = FactCategory.FACTS
        
        stability_period = CATEGORY_STABILITY.get(category, 90)
        
        # Recommend reinforcement at 50% of stability period
        first_reinforcement = stability_period * 0.5
        second_reinforcement = stability_period * 0.75
        third_reinforcement = stability_period * 0.9
        
        return {
            "category": category.value,
            "stability_period_days": stability_period,
            "recommended_reinforcements": [
                {"when_days": round(first_reinforcement, 1), "priority": "high"},
                {"when_days": round(second_reinforcement, 1), "priority": "medium"},
                {"when_days": round(third_reinforcement, 1), "priority": "low"},
            ],
            "half_life_days": round(stability_period * math.log(2), 1),
        }
    
    def batch_calculate_decay(
        self,
        facts: list[dict],
    ) -> list[dict]:
        """Calculate decay for multiple facts efficiently.
        
        Args:
            facts: List of fact dicts with required fields:
                - id/uid: Fact identifier
                - importance: Current importance
                - category: Fact category
                - created_at: Creation timestamp (Unix seconds)
                - last_accessed_at: Optional last access timestamp
                - reinforcement_count: Optional reinforcement count
                
        Returns:
            List of decay calculation results
        """
        current_time = time.time()
        results = []
        
        for fact in facts:
            fact_id = fact.get("id") or fact.get("uid")
            if not fact_id:
                continue
            
            # Extract fact metadata
            importance = fact.get("importance", 0.5)
            category = fact.get("category", FactCategory.FACTS)
            created_at = fact.get("created_at", current_time)
            last_accessed_at = fact.get("last_accessed_at")
            reinforcement_count = fact.get("reinforcement_count", 0)
            
            # Calculate age
            age_days = (current_time - created_at) / 86400.0
            
            # Calculate last accessed
            last_accessed_days_ago = None
            if last_accessed_at:
                last_accessed_days_ago = (current_time - last_accessed_at) / 86400.0
            
            # Calculate decay
            decay_result = self.calculate_decay(
                initial_importance=importance,
                category=category,
                age_days=age_days,
                last_accessed_days_ago=last_accessed_days_ago,
                reinforcement_count=reinforcement_count,
            )
            
            # Add fact ID to result
            decay_result["fact_id"] = fact_id
            
            results.append(decay_result)
        
        return results
    
    def identify_facts_for_pruning(
        self,
        decay_results: list[dict],
        threshold: float | None = None,
    ) -> list[str]:
        """Identify facts that should be pruned based on decay.
        
        Args:
            decay_results: Results from batch_calculate_decay
            threshold: Importance threshold (defaults to settings.prune_importance_threshold)
            
        Returns:
            List of fact IDs to prune
        """
        prune_threshold = threshold or settings.prune_importance_threshold
        
        to_prune = []
        for result in decay_results:
            if result["decayed_importance"] < prune_threshold:
                to_prune.append(result["fact_id"])
        
        return to_prune
    
    def identify_facts_for_merging(
        self,
        facts: list[dict],
        similarity_threshold: float | None = None,
    ) -> list[tuple[str, str, float]]:
        """Identify fact pairs that should be merged based on similarity.
        
        Args:
            facts: List of fact dicts with embeddings
            similarity_threshold: Similarity threshold (defaults to settings.dedup_similarity_threshold)
            
        Returns:
            List of (fact_id_a, fact_id_b, similarity) tuples
        """
        threshold = similarity_threshold or settings.dedup_similarity_threshold
        pairs_to_merge = []
        
        # Compare all pairs (O(n²) but acceptable for consolidation batches)
        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                fact_a = facts[i]
                fact_b = facts[j]
                
                # Skip if either already superseded
                if fact_a.get("superseded_by") or fact_b.get("superseded_by"):
                    continue
                
                # Calculate similarity
                similarity = self._calculate_similarity(fact_a, fact_b)
                
                if similarity >= threshold:
                    id_a = fact_a.get("id") or fact_a.get("uid")
                    id_b = fact_b.get("id") or fact_b.get("uid")
                    if id_a and id_b:
                        pairs_to_merge.append((id_a, id_b, similarity))
        
        return pairs_to_merge
    
    def _calculate_similarity(self, fact_a: dict, fact_b: dict) -> float:
        """Calculate cosine similarity between two facts.
        
        Uses embedding vectors if available, otherwise text-based Jaccard.
        
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
    
    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec_a: First vector
            vec_b: Second vector
            
        Returns:
            Cosine similarity (0.0 to 1.0)
        """
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(x * x for x in vec_a))
        norm_b = math.sqrt(sum(x * x for x in vec_b))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
