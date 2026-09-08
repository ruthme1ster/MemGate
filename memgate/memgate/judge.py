"""P2 — LLM-judged salience, run against a local open-weights model.

The build plan's P2 was "a small LLM rates each turn's importance". It stalled
because there was no API key and no local generative model. Both are now
resolved by vendoring Qwen2.5-0.5B-Instruct (~942 MB) and running it on the
machine's own GPU (Apple MPS), so this stays offline, free and reproducible --
the same constraints the rest of the harness is built to.

TWO DESIGN CHOICES WORTH DEFENDING
----------------------------------
1. NO GENERATION. Asking the model to emit a digit and parsing it wastes an
   autoregressive loop on a one-token answer, and a model that replies "I'd say
   7" parses as garbage. Instead this runs ONE forward pass and reads the
   logits at the final position restricted to the ten digit tokens, then takes
   the expectation sum(i * p_i)/9.

   That is faster, cannot fail to parse, and -- the real reason -- it is
   CONTINUOUS. Eviction ranks items by utility, so ties matter: a discrete 0-9
   integer puts hundreds of turns on the same rung and leaves their order to the
   tiebreak, whereas the expectation separates them.

2. RIGHT PADDING, NOT LEFT. Batched scoring first returned NaN for exactly the
   short inputs -- the fillers -- which would have silently mis-scored every
   "ok thanks!" in the corpus. The instinct is to blame precision, and fp32 was
   tried first; it did not fix it. The cause is the MASK. Left padding puts the
   pad tokens FIRST, so a short sequence has leading rows that are entirely
   masked, the attention softmax sees an all -inf row, and Qwen2 on MPS returns
   NaN. Right padding leaves no row fully masked; the last real token is then at
   `attention_mask.sum(1) - 1` instead of at -1.

   With the mask fixed, bfloat16 is safe and 6.8x faster than float32 here
   (2.78 vs 0.41 turns/s) for the same score range. Batches are length-sorted so
   padding stays minimal, and `lm_head` is applied only to the gathered final
   hidden state -- projecting every position to a 151936-token vocabulary
   overflows MPS at 4.78 GB.

   The NaN guard in `score_many` is deliberate and stays: it is what surfaced
   this at all, and a scorer that silently returns NaN for every filler turn
   would have quietly inverted the experiment.

CALIBRATION. The raw scores are compressed into roughly [0.12, 0.27] rather than
spread over [0,1]. That is harmless for the SELECT policy, which only ever ranks
items to choose an eviction victim, but it would break threshold-mode routing
(nothing would ever clear tau_fact = 0.45). `rank_normalise` maps scores onto
their empirical quantiles when a calibrated 0-1 utility is required. The mapping
uses only the turns' own scores -- never the questions or evidence -- so it
cannot leak labels.

NO LABEL LEAKAGE. The judge sees one turn's text. It never sees a question, an
evidence set, or a fact_id, and the invariance test covers this scorer path.
"""
import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

from .scoring import (Scorer, FILLER_PAT, CORRECTION_HINTS, NUM_PAT,
                      MONTH_PAT, words)

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "models", "Qwen2.5-0.5B-Instruct")
CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results", "judge_cache.json")

PROMPT = ("On a scale 0-9, how important is this message to remember for later "
          "questions? Reply with one digit only.\n\n")


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class LocalLLMJudge:
    """Batch salience scoring against a local causal LM. Loads torch lazily."""

    def __init__(self, model_dir: str = DEFAULT_MODEL, device: str = "auto",
                 max_chars: int = 400, batch_size: int = 32,
                 dtype: str = "float32"):
        self.model_dir = model_dir
        self.max_chars = max_chars
        self.batch_size = batch_size
        self.dtype = dtype
        self._device = device
        self._tok = None
        self._model = None
        self._digits = None
        self.cache: Dict[str, float] = {}

    # ------------------------------------------------------------------ cache
    def load_cache(self, path: str = CACHE_PATH) -> int:
        if os.path.exists(path):
            with open(path) as f:
                self.cache.update(json.load(f))
        return len(self.cache)

    def save_cache(self, path: str = CACHE_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.cache, f)

    # ------------------------------------------------------------------ model
    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dev = self._device
        if dev == "auto":
            dev = ("mps" if torch.backends.mps.is_available()
                   else "cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.model_dir)
        # RIGHT padding, with a gather at each sequence's true last token.
        #
        # Left padding is the usual choice for batched decoding, and it is what
        # broke here: with left padding the first rows of a short sequence are
        # fully masked, the attention softmax sees an all -inf row, and Qwen2 on
        # MPS returns NaN. It struck exactly the short turns -- the fillers --
        # so it would have scored every "ok thanks!" as NaN. float32 did not fix
        # it; the mask is the cause, not the precision.
        #
        # Right padding puts the pad tokens AFTER the real ones, so no row is
        # ever fully masked. The last real token is then at
        # attention_mask.sum(1) - 1 rather than at -1.
        self._tok.padding_side = "right"
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        # float32: fp16 + left padding produced NaN on MPS for short inputs.
        dt = {"float32": torch.float32, "bfloat16": torch.bfloat16,
              "float16": torch.float16}[self.dtype]
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_dir, torch_dtype=dt).to(dev).eval()
        self._dev = dev
        self._digits = [self._tok.encode(str(i), add_special_tokens=False)[0]
                        for i in range(10)]

    def _prompt(self, text: str) -> str:
        return self._tok.apply_chat_template(
            [{"role": "user", "content": PROMPT + text[:self.max_chars]}],
            add_generation_prompt=True, tokenize=False)

    def score_many(self, texts: Sequence[str], verbose: bool = False,
                   save_every: int = 0) -> List[float]:
        """Score every text, using and filling the cache."""
        todo = [t for t in dict.fromkeys(texts) if _key(t) not in self.cache]
        if todo:
            self._ensure()
            torch = self._torch
            # Length-sorted so each batch pads to a similar width.
            todo.sort(key=len)
            for i in range(0, len(todo), self.batch_size):
                chunk = todo[i:i + self.batch_size]
                enc = self._tok([self._prompt(t) for t in chunk],
                                return_tensors="pt", padding=True).to(self._dev)
                # Run the base transformer and apply lm_head ONLY at the last
                # real token of each sequence. Calling the full model would
                # project every position to the 151936-token vocabulary --
                # batch x seq x vocab x 4 B, which overflowed at 4.78 GB. We
                # need exactly one distribution per sequence, so projecting one
                # hidden vector each is both correct and ~seq_len times cheaper.
                last = enc["attention_mask"].sum(1) - 1          # true final token
                with torch.no_grad():
                    h = self._model.model(**enc).last_hidden_state
                    h_last = h[torch.arange(h.shape[0], device=h.device), last, :]
                    lg = self._model.lm_head(h_last).float()
                p = torch.softmax(lg[:, self._digits], dim=-1)
                sc = (p * torch.arange(10, device=p.device,
                                       dtype=p.dtype)).sum(-1) / 9.0
                for t, v in zip(chunk, sc.tolist()):
                    # A NaN here means the numerics broke; fail loudly rather
                    # than silently scoring a turn as 0 and calling it filler.
                    if v != v:
                        raise RuntimeError(f"judge produced NaN for: {t[:60]!r}")
                    self.cache[_key(t)] = float(v)
                done = i + len(chunk)
                if verbose:
                    print(f"    judged {done}/{len(todo)}", flush=True)
                # This run takes ~90 minutes. Checkpointing means a crash costs
                # one batch, not the whole thing.
                if save_every and done % save_every < self.batch_size:
                    self.save_cache()
        return [self.cache[_key(t)] for t in texts]

    def score_one(self, text: str) -> float:
        return self.score_many([text])[0]


def rank_normalise(scores: Sequence[float]) -> Dict[float, float]:
    """Map raw judge scores onto their empirical quantiles in [0,1].

    The judge's raw range is narrow, which is fine for ranking but breaks any
    absolute threshold. This spreads it without changing the ORDER, so a policy
    that only ranks is unaffected and one that thresholds becomes usable.
    """
    uniq = sorted(set(scores))
    if len(uniq) < 2:
        return {s: 0.5 for s in uniq}
    return {s: i / (len(uniq) - 1) for i, s in enumerate(uniq)}


class LLMJudgeScorer(Scorer):
    """P2 — utility from the local judge; labels from P1's vocabulary.

    Only the utility number changes relative to P1, so routing, supersession and
    the type counters behave identically and this is a clean one-component swap.
    Pass `calibration` (from `rank_normalise`) to get spread-out utilities.
    """
    name = "P2-llm-judge"

    def __init__(self, judge: LocalLLMJudge,
                 calibration: Optional[Dict[float, float]] = None):
        self.judge = judge
        self.calibration = calibration

    def score(self, text: str, speaker: str = "user") -> Tuple[float, str]:
        u = self.judge.score_one(text)
        if self.calibration is not None:
            u = self.calibration.get(u, u)
        t = text.strip()
        w = set(words(t))
        if FILLER_PAT.match(t) or len(words(t)) <= 2:
            label = "filler"
        elif (w & CORRECTION_HINTS) and (NUM_PAT.search(t) or MONTH_PAT.search(t)):
            label = "update"
        elif t.endswith("?"):
            label = "query"
        else:
            label = "fact" if u >= 0.5 else "context"
        return max(0.0, min(1.0, float(u))), label
