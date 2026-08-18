"""Tokenizer and embedder, with pluggable backends.

Step 0 ran on a hashing bag-of-words embedder and a x1.3 token heuristic so it
would work with no downloads. Phase 1 swaps in the real components:

    tokenizer   heuristic  ->  tiktoken (cl100k_base)
    embedder    hashing    ->  sentence-transformers all-MiniLM-L6-v2

Both backends are kept. The fallbacks are not dead code -- they are what lets
the suite run on a machine with no model cache, and they are the ablation that
shows how much of the result depends on embedding quality.

REPRODUCIBILITY: which backend ran is not a detail. The same policy scores
differently under different embedders, so `backend_info()` is recorded into
every results row rather than left implicit.

Selection (env var, default "auto" = prefer real, fall back quietly):
    MEMGATE_EMBEDDER = auto | hash | minilm
    MEMGATE_TOKENIZER = auto | heuristic | tiktoken
"""
import os
import re
import sys as _sys
import zlib
import math
from typing import List, Sequence

# sentence-transformers pulls in transformers, whose TensorFlow path crashes
# when Keras 3 is installed ("Keras 3 not yet supported in Transformers").
# We only ever use the torch path, so disable the TF one before it is imported.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_WORD = re.compile(r"[A-Za-z0-9']+")

try:
    import numpy as _np
except ImportError:
    _np = None


def words(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def answer_tokens(text: str) -> List[str]:
    """Normalised tokens for answer-presence checking.

    Stopwords are dropped so that "the deadline is March 15" is not judged
    present merely because "the" and "is" survived compression. What must
    survive is the content: the dates, names and numbers.
    """
    return [w for w in words(text) if w not in _STOP]


_STOP = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of",
         "in", "on", "at", "for", "and", "or", "it", "its", "he", "she", "they",
         "her", "his", "their", "them", "that", "this", "with", "as", "by",
         "from", "has", "have", "had", "do", "does", "did", "will", "would"}


# ---------------------------------------------------------------- tokenizer

_TOK_MODE = os.environ.get("MEMGATE_TOKENIZER", "auto").lower()
_tiktoken_enc = None
_tok_backend = "heuristic"


def _get_tiktoken():
    """Lazy-load the BPE encoder. Returns None if unavailable."""
    global _tiktoken_enc, _tok_backend
    if _tiktoken_enc is not None:
        return _tiktoken_enc
    if _TOK_MODE == "heuristic":
        return None
    try:
        import tiktoken
        _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        _tok_backend = "tiktoken/cl100k_base"
    except Exception:
        if _TOK_MODE == "tiktoken":
            raise
        _tiktoken_enc = None
    return _tiktoken_enc


def count_tokens(text: str) -> int:
    """Token count for budget accounting.

    This is the unit the entire experiment is denominated in -- every budget,
    every cost claim -- so a biased tokenizer biases every result. The x1.3
    heuristic approximates sub-word splitting for English; tiktoken is exact
    for GPT-family models.
    """
    if not text:
        return 0
    enc = _get_tiktoken()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, int(round(len(words(text)) * 1.3)))


# ---------------------------------------------------------------- embedder

DIM = 256
MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# A vendored copy is preferred over the Hub when present. The Hub download is
# unreliable on this connection -- it failed repeatedly and once produced a
# corrupt model.safetensors that loaded as a silent fallback -- so pinning a
# verified local copy makes runs offline and reproducible.
_LOCAL_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "all-MiniLM-L6-v2")
_MINILM_BYTES = 90868376        # exact size of model.safetensors


def _local_model_ok() -> bool:
    """True only if the vendored weights are COMPLETE.

    Size is checked, not just existence: this connection repeatedly truncated
    the 90 MB download and curl restarted rather than resuming, leaving a
    partial file that loads as `InvalidHeaderDeserialization`. Treating a
    truncated file as present would trip the fallback on every call.
    """
    f = os.path.join(_LOCAL_MODEL, "model.safetensors")
    try:
        return os.path.getsize(f) == _MINILM_BYTES
    except OSError:
        return False

_EMB_MODE = os.environ.get("MEMGATE_EMBEDDER", "auto").lower()
_model = None
_emb_backend = None
_cache = {}


def _get_model():
    """Lazy-load MiniLM once. Returns None to mean 'use the hashing fallback'."""
    global _model, _emb_backend
    if _emb_backend is not None:
        return _model
    if _EMB_MODE == "hash":
        _emb_backend = f"hash-bow/{DIM}d"
        return None
    try:
        from sentence_transformers import SentenceTransformer
        src = _LOCAL_MODEL if _local_model_ok() else MINILM_NAME
        _model = SentenceTransformer(src)
        _model.eval()
        _emb_backend = "all-MiniLM-L6-v2/384d"
    except Exception as exc:
        if _EMB_MODE == "minilm":
            raise
        # Loud, not silent. A quiet fallback here means an entire results table
        # gets labelled as MiniLM while actually being produced by the hashing
        # baseline. This has already happened once: a resumed download left a
        # corrupt model.safetensors, and only backend_info() caught it.
        print(f"  [MemGate] WARNING: MiniLM unavailable ({type(exc).__name__}: "
              f"{str(exc)[:90]}); falling back to the hashing embedder.",
              file=_sys.stderr)
        _model, _emb_backend = None, f"hash-bow/{DIM}d"
    return _model


def _hash_embed(text: str, dim: int = DIM):
    """Deterministic hashing bag-of-words.

    crc32 rather than Python's hash(), because str hashing is salted per
    process and would make runs non-reproducible.
    """
    v = [0.0] * dim
    for w in words(text):
        v[zlib.crc32(w.encode()) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in v))
    if norm > 0:
        v = [x / norm for x in v]
    return _np.asarray(v, dtype="float32") if _np is not None else v


def embed(text: str):
    """Embed one string. Cached: the store re-embeds the same text on every
    dedupe check and every retrieval, so this is a large constant-factor win."""
    hit = _cache.get(text)
    if hit is not None:
        return hit
    model = _get_model()
    if model is None:
        vec = _hash_embed(text)
    else:
        vec = model.encode(text, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True)
    _cache[text] = vec
    return vec


def embed_many(texts: Sequence[str]) -> List:
    """Batch-embed, populating the cache.

    A transformer forward pass per string is the dominant cost on a real
    corpus; batching it is worth an order of magnitude. Call `warm_cache()`
    before a replay so the per-item `embed()` calls all hit the cache.
    """
    missing = [t for t in dict.fromkeys(texts) if t not in _cache]
    if missing:
        model = _get_model()
        if model is None:
            for t in missing:
                _cache[t] = _hash_embed(t)
        else:
            vecs = model.encode(missing, normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True,
                                batch_size=128)
            for t, v in zip(missing, vecs):
                _cache[t] = v
    return [embed(t) for t in texts]


def warm_cache(texts: Sequence[str]) -> int:
    """Pre-embed a corpus in batches. Returns the number newly computed."""
    before = len(_cache)
    embed_many(list(texts))
    return len(_cache) - before


def reset_cache():
    _cache.clear()


def cosine(a, b) -> float:
    if a is None or b is None:
        return 0.0
    if _np is not None and isinstance(a, _np.ndarray) and isinstance(b, _np.ndarray):
        return float(a @ b)          # both sides are L2-normalised
    return sum(x * y for x, y in zip(a, b))


def backend_info() -> dict:
    """What actually ran. Recorded into results so a number can be traced to
    the components that produced it."""
    _get_model()
    _get_tiktoken()
    return {"embedder": _emb_backend or "unset", "tokenizer": _tok_backend}
