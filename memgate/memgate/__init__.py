"""MemGate — an adaptive memory layer for long-running LLM agents.

Capstone project, SVKM's NMIMS Indore.
Simar Singh Khanuja & Yash Ramchandani.
"""
__version__ = "0.1.0"

import os as _os

# Offline by default when the models are vendored.
#
# Every model this project uses is a verified local copy (MiniLM in
# models/all-MiniLM-L6-v2, the judge and reader in models/Qwen2.5-*), and the
# loaders are still Hub-aware: given a local path they will nonetheless reach
# out to check it. On a healthy connection that costs a second. On a stalled
# one it hangs -- a 90-second test suite ran for 14 minutes against an
# ESTABLISHED socket to huggingface.co doing nothing, which is how this was
# found.
#
# The deeper reason is reproducibility, not speed: a run that can silently
# fetch a model is a run whose inputs are not pinned. §16.3 already lost a
# results table to a half-downloaded MiniLM. Set MEMGATE_ALLOW_HUB=1 to permit
# network fetches when deliberately adding a new model.
if _os.environ.get("MEMGATE_ALLOW_HUB", "").lower() not in ("1", "true", "yes"):
    _models = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            "..", "models")
    if _os.path.isdir(_models):
        _os.environ.setdefault("HF_HUB_OFFLINE", "1")
        _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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
