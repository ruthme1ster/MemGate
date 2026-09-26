"""Abstractive compression — rewriting an evicted turn instead of selecting words.

`compress.py` established the EXTRACTIVE baseline: keep the most informative
words of a turn, in their original order. The project's negative result about
compression is stated for that case only. §23 found that demoting an evicted
turn to a gist loses 6.71 answer-recall points against dropping it outright,
and every deliverable since has listed abstractive re-summarisation as the
experiment that would either overturn that finding or strengthen it.

There is a specific reason to expect a different outcome. An extractive gist is
constrained to a *subsequence* of the original, so it keeps words but not
grammar, and what it drops is gone:

    "[Caroline] Hey Mel! Good to see you! My sister Priya moved to Pune in March."
    extractive ->  "[Caroline] sister Priya moved Pune March."

A rewriting summariser is not constrained that way and can carry the same
content in fewer tokens while remaining a sentence. If compression ever pays
for itself, it should pay here. If it still loses, the negative result stops
being about extractive methods and starts being about compression.

WHAT IS HELD FIXED
------------------
This is a single-component swap, the discipline §28 used for the judge. Policy,
storage budget, retrieval, eviction order and thresholds are untouched; the only
thing that changes is the function turning a stored item into a shorter stored
item. `ThreeTierStore._gist` is that single site, so demotion under pressure and
Tier-2 consolidation both move together — a swap applied to only one of them
would measure a mixture of the two compressors.

THE BUDGET IS ENFORCED, NOT REQUESTED
-------------------------------------
A generated gist is passed through the extractive compressor before it is
returned, so it obeys the same token bound the store's accounting assumes. A
model that ignores "at most N words" cannot silently inflate the store and
invalidate the comparison: the budget is a postcondition here, not a prompt.

COST
----
~20 generated tokens per compressed item against microseconds for the
extractive rule. That asymmetry is what the result has to justify, and it is
the same shape as the P2 judge's: the interesting outcome is not whether the
expensive component wins, but by how much, and whether it wins at all.
Generations are cached on disk by (model, budget, text), so re-running a sweep
costs nothing, and `warm()` checkpoints so an interrupted pass resumes.
"""
import hashlib
import json
import os
import re
from typing import Optional, Sequence

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from .compress import compress, informative_head

_SPEAKER = re.compile(r"^\s*\[([^\]]{1,40})\]\s*")
_WORDY = re.compile(r"[A-Za-z0-9]")

SYSTEM = ("You compress messages for an assistant's memory. Rewrite the "
          "message as a single short statement that preserves every name, "
          "number, date, place and decision. Drop greetings and small talk. "
          "Output only the rewritten statement, with no preamble and no "
          "quotation marks.")

USER = 'Rewrite in at most {n} words:\n"{text}"'


def _cache_path(model_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", os.path.basename(model_name))
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "results")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"abstractive_cache_{safe}.json")


class AbstractiveCompressor:
    """A rewriting summariser with the extractive compressor as its floor.

    Callable as `(text, max_words) -> str`, which is the signature
    `ThreeTierStore._gist` and `MemGatePolicy._summarise` expect, so it drops in
    wherever `informative_head` is used today.

    Degenerate output falls back to extraction rather than to nothing. A model
    that returns an empty string, a refusal, or something with no alphanumeric
    content would otherwise delete the item's content while the store still pays
    for the item — the worst of both policies, and a failure that would show up
    as "compression is even worse than we thought" rather than as a bug.
    """

    name = "abstractive"

    def __init__(self, model_name: Optional[str] = None, cache: bool = True,
                 max_chars: int = 600, max_new: int = 48):
        from .llm_judge import default_model
        self.model_name = model_name or default_model()
        self.max_chars = max_chars
        self.max_new = max_new
        self._model = self._tok = None
        self.calls = 0
        self.hits = 0
        self.fallbacks = 0
        self._cache_file = _cache_path(self.model_name) if cache else None
        self._cache = {}
        if self._cache_file and os.path.exists(self._cache_file):
            try:
                with open(self._cache_file) as f:
                    self._cache = json.load(f)
            except (OSError, ValueError):
                self._cache = {}

    # ------------------------------------------------------------- model
    def _load(self):
        if self._model is None:
            from mlx_lm import load
            self._model, self._tok = load(self.model_name)

    def _key(self, text: str, max_words: int) -> str:
        raw = f"{os.path.basename(self.model_name)}\x00{max_words}\x00{text}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _generate(self, body: str, max_words: int) -> str:
        from mlx_lm import generate
        self._load()
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER.format(
                    n=max_words, text=body[:self.max_chars])}]
        try:
            prompt = self._tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        except Exception:                      # tokenizer without a chat template
            prompt = f"{SYSTEM}\n\n{msgs[1]['content']}\nRewritten:"
        out = generate(self._model, self._tok, prompt=prompt,
                       max_tokens=self.max_new, verbose=False)
        self.calls += 1
        return out.strip().split("\n")[0].strip().strip('"').strip()

    # ------------------------------------------------------------- public
    def __call__(self, text: str, max_words: int) -> str:
        """Rewrite `text` to at most ~`max_words` words, speaker tag preserved.

        The "[Speaker]" prefix is carried through untouched rather than being
        left to the model. Attribution is what makes a gist interpretable later,
        it costs 2-3 tokens, and a rewrite is exactly the operation most likely
        to drop it.
        """
        if not text or max_words <= 0:
            return ""
        budget = max(1, int(max_words * 1.4))          # same as informative_head

        m = _SPEAKER.match(text)
        prefix, body = (m.group(0).strip(), text[m.end():]) if m else ("", text)
        if not body.strip():
            return text

        key = self._key(body, max_words)
        if key in self._cache:
            self.hits += 1
            gist = self._cache[key]
        else:
            try:
                gist = self._generate(body, max_words)
            except Exception:
                gist = ""
            self._cache[key] = gist

        if not gist or not _WORDY.search(gist):
            self.fallbacks += 1
            return informative_head(text, max_words)

        out = f"{prefix} {gist}".strip() if prefix else gist
        # Budget as postcondition. `compress` is a no-op when the text already
        # fits, so a well-behaved rewrite passes through untouched and only an
        # over-long one is trimmed -- and it is trimmed by the extractive rule,
        # which is the baseline this is being compared against.
        return compress(out, max_tokens=budget)

    # ------------------------------------------------------------- cache
    def save_cache(self):
        if not self._cache_file:
            return None
        tmp = self._cache_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._cache, f)
        os.replace(tmp, self._cache_file)      # atomic: a killed run cannot
        return self._cache_file                # leave a truncated cache

    def warm(self, texts: Sequence[str], max_words: int,
             progress_every: int = 100) -> int:
        """Pre-generate gists for a corpus, checkpointing as it goes.

        Compression only happens on eviction, so a sweep touches far fewer turns
        than the corpus holds -- but which ones depends on the storage budget,
        and re-running at a second budget would otherwise pay again for every
        turn the first run already compressed. Warming the whole corpus once
        makes every later configuration a cache read.
        """
        import time
        todo = [t for t in dict.fromkeys(texts)
                if self._key(t, max_words) not in self._cache]
        t0 = time.perf_counter()
        for n, t in enumerate(todo, 1):
            self(t, max_words)
            if n % progress_every == 0:
                self.save_cache()
                el = time.perf_counter() - t0
                print(f"    {n}/{len(todo)} rewritten  ({el/n:.2f}s each, "
                      f"~{(len(todo)-n)*el/n/60:.1f} min left)", flush=True)
        self.save_cache()
        return len(todo)

    def stats(self):
        return {"generated": self.calls, "cache_hits": self.hits,
                "fallbacks": self.fallbacks, "cached": len(self._cache),
                "model": os.path.basename(self.model_name)}
