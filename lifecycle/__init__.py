"""Lifecycle management package - Memory evolution pipeline."""

from lifecycle.decay_engine import DecayEngine, FactCategory
from lifecycle.merge_strategy import MergeStrategy

__all__ = ["DecayEngine", "FactCategory", "MergeStrategy"]
