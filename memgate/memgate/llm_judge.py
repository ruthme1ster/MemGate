"""P2 — LLM-judged salience, scored by logprob rather than generation.

Fills the build plan's P2 slot with a small open-weights instruct model running
locally on Apple Silicon via MLX. No API key, no network at inference, nothing
leaves the machine.

WHY LOGPROBS AND NOT GENERATION
-------------------------------
The obvious implementation asks the model to emit {"importance": 0.0-1.0} and
parses it. That is worse in every dimension that matters here:

  * a 1.5B model emits malformed JSON often enough to need retry logic
  * sampled numbers cluster on 0.2/0.5/0.8 -- coarse, and poorly calibrated
  * generating even 10 tokens costs ~10 forward passes per turn

Instead the turn is posed as one yes/no question and the score is read straight
off the next-token distribution:

    utility = P("Yes") / (P("Yes") + P("No"))

One forward pass, no sampling, no parsing, and a genuinely continuous score.
Renormalising over the two candidates makes it robust to the model putting mass
elsewhere, and gives a value that is directly comparable to P1's [0,1] utility,
so it drops into the same router with no threshold retuning.

WHAT THIS IS, RELATIVE TO THE OTHER SCORERS
-------------------------------------------
`LearnedScorer` (P3-learned) is supervised by "was this turn ever cited as
evidence" -- information a deployed agent cannot have, which is why it is
documented there as a headroom bound rather than a component.

P2 is **deployable**: at score time it sees one turn's text and nothing else --
no future questions, no evidence set, no fact_id. So it is subject to the same
label-invariance test as P1, and it is the first scorer in this project that is
both strong and honest about what it knows at write time.

Its labels are also what P3 (`DistilledScorer`) was always meant to distil, so
wiring this in is what unblocks the build plan's Step 3 as originally specified.
"""
import hashlib
import json
import os
import re
from typing import List, Sequence, Tuple

# transformers' TF path crashes under Keras 3; mlx_lm pulls in its tokenizer.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from .scoring import Scorer, HeuristicScorer

# Preference order: a vendored local copy first, then the Hub. Same pattern as
# utils.py uses for MiniLM, and for the same reason -- Hub downloads have failed
# repeatedly on this connection, so a verified local copy is what makes a run
# offline and reproducible.
_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
LOCAL_CANDIDATES = ("Qwen2.5-1.5B-Instruct-4bit", "Qwen2.5-0.5B-Instruct")
HUB_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def _weights_ok(d: str) -> bool:
    """True only if every safetensors shard in `d` opens.

    Presence is not enough. A part-downloaded file sits on disk at full name and
    fails at load time with InvalidHeaderDeserialization -- exactly how a
    resumed MiniLM download silently degraded an entire results table (§16.3).
    Reading the header is cheap and catches truncation without needing to know
    the expected byte size.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        return True                      # cannot verify; let the loader decide
    shards = [f for f in os.listdir(d) if f.endswith(".safetensors")]
    if not shards:
        return False
    for f in shards:
        try:
            with safe_open(os.path.join(d, f), framework="numpy"):
                pass
        except Exception:
            return False
    return True


def default_model() -> str:
    """First local model with complete weights, else the Hub id."""
    for name in LOCAL_CANDIDATES:
        d = os.path.join(_MODELS_DIR, name)
        if os.path.isfile(os.path.join(d, "config.json")) and _weights_ok(d):
            return d
    return HUB_MODEL


DEFAULT_MODEL = None   # resolved lazily so a model added later is picked up

SYSTEM = ("You decide what an AI assistant should remember from a conversation. "
          "Durable facts, decisions, preferences, plans, names, dates and "
          "numbers are worth remembering. Greetings, acknowledgements and "
          "small talk are not.")

USER = ('Message:\n"{text}"\n\n'
        "Will this message be needed to answer a question later? "
        "Answer Yes or No.")

_YES = ("Yes", "yes", " Yes", " yes", "YES")
_NO = ("No", "no", " No", " no", "NO")


def _cache_path(model_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", model_name)
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "results")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"p2_cache_{safe}.json")


class MLXJudgeScorer(Scorer):
    """P2 — salience from a local instruct model's yes/no logprob.

    The score is the model's. The *type label* is not: it is taken from the P1
    heuristic, because label detection ("update", "filler") is regex-reliable
    and asking the model for it would double the cost for the one part that
    already works. Documented rather than hidden -- P2's contribution here is
    the salience number, and that is what the ablation measures.

    Results are cached on disk by text hash. A labelling pass over LoCoMo is
    ~5900 forward passes; it must survive a restart, and re-running the sweep
    must not re-run the model.
    """
    name = "P2-llm-judge"

    def __init__(self, model_name: str = None, cache: bool = True,
                 max_chars: int = 600):
        self.model_name = model_name or default_model()
        self.max_chars = max_chars
        self._model = None
        self._tok = None
        self._yes_ids: List[int] = []
        self._no_ids: List[int] = []
        self._labeller = HeuristicScorer()
        self.calls = 0
        self.hits = 0
        self._cache_file = _cache_path(model_name) if cache else None
        self._cache = {}
        if self._cache_file and os.path.exists(self._cache_file):
            try:
                with open(self._cache_file) as f:
                    self._cache = json.load(f)
            except (OSError, ValueError):
                self._cache = {}

    # ------------------------------------------------------------ model
    def _load(self):
        if self._model is not None:
            return
        from mlx_lm import load
        self._model, self._tok = load(self.model_name)
        self._yes_ids = self._first_ids(_YES)
        self._no_ids = self._first_ids(_NO)
        if not self._yes_ids or not self._no_ids:
            raise RuntimeError(
                f"could not resolve Yes/No token ids for {self.model_name}")

    def _first_ids(self, variants: Sequence[str]) -> List[int]:
        """First-token ids for each surface form, deduped.

        Summing over variants rather than picking one: whether the model puts
        its mass on "Yes", "yes" or " Yes" is a tokenizer detail, not a signal
        about the turn.
        """
        out = []
        for v in variants:
            ids = self._tok.encode(v, add_special_tokens=False)
            if ids and ids[0] not in out:
                out.append(ids[0])
        return out

    def _prompt_ids(self, text: str) -> List[int]:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(
                    text=text[:self.max_chars])}]
        try:
            s = self._tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        except Exception:                      # tokenizer without a chat template
            s = f"{SYSTEM}\n\n{msgs[1]['content']}\nAnswer:"
        return self._tok.encode(s)

    # ------------------------------------------------------------ scoring
    def _p_yes(self, text: str) -> float:
        import mlx.core as mx
        self._load()
        ids = self._prompt_ids(text)
        logits = self._model(mx.array([ids]))[0, -1]
        logits = logits.astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits)
        p = mx.exp(logprobs)
        yes = float(sum(p[i].item() for i in self._yes_ids))
        no = float(sum(p[i].item() for i in self._no_ids))
        self.calls += 1
        total = yes + no
        # Both candidates near zero means the model answered neither; fall back
        # to the midpoint rather than inventing a confident score from noise.
        return 0.5 if total <= 1e-9 else yes / total

    def score(self, text: str, speaker: str = "user") -> Tuple[float, str]:
        _, label = self._labeller.score(text, speaker)
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if key in self._cache:
            self.hits += 1
            return float(self._cache[key]), label
        u = self._p_yes(text)
        self._cache[key] = round(u, 6)
        return u, label

    def save_cache(self):
        if not self._cache_file:
            return
        tmp = self._cache_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._cache, f)
        os.replace(tmp, self._cache_file)      # atomic: a killed run cannot
        return self._cache_file                # leave a truncated cache

    def warm(self, texts: Sequence[str], progress_every: int = 200) -> int:
        """Pre-score a corpus, checkpointing as it goes.

        A full LoCoMo pass is thousands of forward passes and takes tens of
        minutes; checkpointing means an interrupted run resumes instead of
        starting over.
        """
        import time
        todo = [t for t in dict.fromkeys(texts)
                if hashlib.sha1(t.encode("utf-8")).hexdigest() not in self._cache]
        t0 = time.perf_counter()
        for n, t in enumerate(todo, 1):
            self.score(t)
            if n % progress_every == 0:
                self.save_cache()
                el = time.perf_counter() - t0
                print(f"    {n}/{len(todo)} scored  "
                      f"({el/n:.2f}s/turn, ~{(len(todo)-n)*el/n/60:.1f} min left)",
                      flush=True)
        self.save_cache()
        return len(todo)
