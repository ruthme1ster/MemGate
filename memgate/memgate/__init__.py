"""MemGate — an adaptive memory layer for long-running LLM agents.

Capstone project, SVKM's NMIMS Indore.
Simar Singh Khanuja & Yash Ramchandani.
"""
__version__ = "0.1.0"

from .types import Turn, Question, MemoryItem, ContextBundle
from .store import ThreeTierStore
from .scoring import HeuristicScorer, RecencyScorer
from .policies import (FullContextPolicy, OraclePolicy,
                       SlidingWindowPolicy, MemGatePolicy, POLICIES)
from .harness import run_policy, sweep, format_table, to_csv
from .data import build_dataset

__all__ = [
    "Turn", "Question", "MemoryItem", "ContextBundle", "ThreeTierStore",
    "HeuristicScorer", "RecencyScorer", "FullContextPolicy", "OraclePolicy",
    "SlidingWindowPolicy", "MemGatePolicy", "POLICIES", "run_policy",
    "sweep", "format_table", "to_csv", "build_dataset",
]
