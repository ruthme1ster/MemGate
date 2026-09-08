#!/usr/bin/env python3
"""Score every LoCoMo turn with the local LLM judge, once, to a disk cache.

The judge costs ~12 minutes for 5882 turns on this machine. Every experiment
that uses P2 would otherwise pay it again, so it is paid once here and cached by
text hash. Subsequent runs load the cache and the model is never loaded at all.

Usage:
    python3 precompute_judge.py
"""
import sys
import time

from memgate.data import load_locomo
from memgate.judge import LocalLLMJudge, CACHE_PATH


def main():
    print("\nMemGate — precomputing P2 judge scores")
    print("=" * 70)
    data = load_locomo()
    texts = sorted({f"[{t.speaker}] {t.text}"
                    for _s, turns, _q in data for t in turns})
    print(f"  {len(texts)} distinct turns")
    j = LocalLLMJudge(batch_size=64, dtype='bfloat16')
    have = j.load_cache()
    print(f"  cache: {have} already scored")
    todo = len(texts) - sum(1 for t in texts
                            if __import__("memgate.judge", fromlist=["_key"])._key(t) in j.cache)
    print(f"  to score: {todo}\n")
    t0 = time.time()
    scores = j.score_many(texts, verbose=True, save_every=640)
    j.save_cache()
    el = time.time() - t0
    print(f"\n  scored in {el/60:.1f} min ({len(texts)/max(el,1e-9):.1f} turns/s)")
    print(f"  wrote {CACHE_PATH}")
    lo = sorted(zip(scores, texts))[:3]
    hi = sorted(zip(scores, texts))[-3:]
    print(f"\n  range {min(scores):.3f} – {max(scores):.3f}")
    print("  lowest-scored:")
    for s, t in lo:
        print(f"    {s:.3f}  {t[:66]}")
    print("  highest-scored:")
    for s, t in reversed(hi):
        print(f"    {s:.3f}  {t[:66]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
