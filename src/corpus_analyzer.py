"""
Post-PII Corpus Analysis Module
===============================

Consumes a cleaned corpus (output of `pii_remover.remove_pii`) and produces
a structured analysis describing token statistics, context-window fit,
chunking guidance, vocabulary diversity, and domain indicators.

The tokenizer is configurable. The default is `bert-base-uncased`, which is
a reasonable proxy for token counts across most encoder-style embedding
models. If the user knows which embedding model they intend to use, they can
pass the matching tokenizer name and get exact counts.

Output schema
-------------
    {
      "doc_count": int,
      "token_stats": {
          "total", "mean", "median", "min", "max", "std"
      },
      "context_window_recommendations": {
          "512":  { "fits": int, "fit_percentage": float },
          "1024": ...,
          "2048": ...,
          "4096": ...,
          "8192": ...
      },
      "chunking_needed": bool,
      "suggested_chunk_size": int | null,
      "suggested_overlap": int | null,
      "vocabulary_metrics": {
          "unique_tokens", "total_tokens", "type_token_ratio"
      },
      "domain_indicators": [ {"term": str, "frequency": int}, ... ]
    }
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

# Common English stopwords. Kept inline so the module has no extra runtime
# dependency for a small static list.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "would",
    "should", "could", "ought", "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "this", "that", "these", "those", "as", "so", "not", "no",
    "nor", "any", "some", "all", "each", "every", "more", "most", "other",
    "such", "than", "too", "very", "just", "also", "only", "own", "same",
    "can", "will", "shall", "may", "might", "must", "one", "two", "three",
}

CONTEXT_WINDOW_THRESHOLDS: List[int] = [512, 1024, 2048, 4096, 8192]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
class _Tokenizer:
    """Wraps a Hugging Face AutoTokenizer with a graceful fallback chain.

    The fallback chain is: HF tokenizer -> tiktoken -> whitespace split.
    Each stage is tried at construction time so the runtime hot path is just
    one method call.
    """

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        self.model_name = model_name
        self._encode: Callable[[str], List[Any]]
        self._kind: str

        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model_name)
            self._encode = lambda t: tok.encode(t, add_special_tokens=False)
            self._kind = f"hf:{model_name}"
            return
        except Exception as e:
            print(f"[corpus_analyzer] HF tokenizer unavailable ({e}); trying tiktoken.")

        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            self._encode = lambda t: enc.encode(t)
            self._kind = "tiktoken:cl100k_base"
            return
        except Exception as e:
            print(f"[corpus_analyzer] tiktoken unavailable ({e}); falling back to whitespace.")

        self._encode = lambda t: t.split()
        self._kind = "whitespace"

    def count(self, text: str) -> int:
        return len(self._encode(text))

    def kind(self) -> str:
        return self._kind


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class CorpusAnalysis:
    doc_count: int
    token_stats: Dict[str, float]
    context_window_recommendations: Dict[str, Dict[str, float]]
    chunking_needed: bool
    suggested_chunk_size: Optional[int]
    suggested_overlap: Optional[int]
    vocabulary_metrics: Dict[str, float]
    domain_indicators: List[Dict[str, Any]]
    tokenizer_used: str = ""
    notes: List[str] = field(default_factory=list)
    # Language profile: dominant language(s), and a multilingual flag the
    # recommender uses to prefer multilingual embedding models when set.
    language_profile: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _split_corpus(corpus: Union[str, List[str]]) -> List[str]:
    """Accept either a list of documents or a single string. A single string
    is split on blank lines, which works well for paragraph- or
    record-oriented corpora."""
    if isinstance(corpus, list):
        return [d for d in corpus if d and d.strip()]
    if isinstance(corpus, str):
        chunks = [c.strip() for c in corpus.split("\n\n") if c.strip()]
        return chunks if chunks else [corpus]
    raise TypeError(f"Corpus must be str or list[str], got {type(corpus).__name__}")


def _percentile(sorted_counts: List[int], p: float) -> float:
    """Linear-interpolated percentile on a pre-sorted list. p in [0, 100]."""
    if not sorted_counts:
        return 0.0
    if len(sorted_counts) == 1:
        return float(sorted_counts[0])
    rank = (p / 100.0) * (len(sorted_counts) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_counts[lo])
    frac = rank - lo
    return sorted_counts[lo] + (sorted_counts[hi] - sorted_counts[lo]) * frac


def _token_stats(counts: List[int]) -> Dict[str, float]:
    if not counts:
        return {"total": 0, "mean": 0.0, "median": 0.0, "min": 0, "max": 0, "std": 0.0,
                "p50": 0.0, "p95": 0.0, "p99": 0.0}
    sorted_counts = sorted(counts)
    return {
        "total": sum(counts),
        "mean": round(statistics.mean(counts), 2),
        "median": round(statistics.median(counts), 2),
        "min": min(counts),
        "max": max(counts),
        "std": round(statistics.pstdev(counts), 2) if len(counts) > 1 else 0.0,
        # Percentiles drive chunking decisions much more reliably than the
        # mean — a corpus with mean=100 but p95=4000 still needs chunking.
        "p50": round(_percentile(sorted_counts, 50), 2),
        "p95": round(_percentile(sorted_counts, 95), 2),
        "p99": round(_percentile(sorted_counts, 99), 2),
    }


def _context_window_fit(counts: List[int]) -> Dict[str, Dict[str, float]]:
    """For each threshold, report how many docs fit and what percentage that
    represents. Output keys are stringified ints for JSON-friendliness."""
    n = len(counts) or 1
    out: Dict[str, Dict[str, float]] = {}
    for w in CONTEXT_WINDOW_THRESHOLDS:
        fits = sum(1 for c in counts if c <= w)
        out[str(w)] = {
            "fits": fits,
            "fit_percentage": round(100.0 * fits / n, 2),
        }
    return out


def _vocabulary_metrics(documents: List[str]) -> Dict[str, float]:
    """Word-level (not subword) vocabulary metrics — lowercase, alphabetic
    tokens only, so digits and punctuation don't dilute the count.

    Type-token ratio (TTR) is unique tokens / total tokens. Higher TTR means
    more lexical diversity. TTR is sensitive to corpus length, so it's most
    useful as a relative metric across similar-sized corpora.
    """
    word_counts: Counter[str] = Counter()
    for doc in documents:
        for w in re.findall(r"[A-Za-z]+", doc.lower()):
            word_counts[w] += 1
    total = sum(word_counts.values())
    unique = len(word_counts)
    return {
        "unique_tokens": unique,
        "total_tokens": total,
        "type_token_ratio": round(unique / total, 4) if total else 0.0,
    }


def _domain_indicators(
    documents: List[str],
    top_k: int = 15,
    min_length: int = 4,
) -> List[Dict[str, Any]]:
    """Approximate domain-specific terminology by surfacing high-frequency,
    low-stopword-overlap tokens. Not a substitute for TF-IDF over a reference
    corpus, but a useful first-pass signal."""
    counts: Counter[str] = Counter()
    for doc in documents:
        for w in re.findall(r"[A-Za-z][A-Za-z\-]+", doc.lower()):
            if len(w) < min_length or w in STOPWORDS:
                continue
            counts[w] += 1
    return [{"term": term, "frequency": freq} for term, freq in counts.most_common(top_k)]


def _detect_language_profile(documents: List[str], sample_n: int = 60) -> Dict[str, Any]:
    """Detect dominant language(s) over a sample of documents.

    Uses `langdetect` if available; falls back to a script-based heuristic
    (Devanagari / Tamil / Bengali / etc.) so we still flag obvious
    non-Latin-script content even without the optional dep.

    Returns:
        {
          "languages": [{"code": "en", "share": 0.82}, {"code": "hi", "share": 0.12}, ...],
          "multilingual": bool,   # True if >1 language >5% share
          "non_latin_present": bool,
          "detector": "langdetect" | "script-heuristic" | "none"
        }
    """
    if not documents:
        return {"languages": [], "multilingual": False, "non_latin_present": False, "detector": "none"}

    # Sample evenly across the corpus rather than the first N docs (which on
    # multi-source corpora would all come from the first source).
    n = len(documents)
    step = max(1, n // sample_n)
    sample = documents[::step][:sample_n]

    # Quick script check is always cheap and useful even when langdetect is
    # available — caches the "non-Latin-present" signal independently.
    non_latin_present = any(
        any(_is_non_latin_script(ch) for ch in doc[:500])
        for doc in sample
    )

    detector = "none"
    counts: Counter[str] = Counter()
    try:
        from langdetect import detect_langs, DetectorFactory
        DetectorFactory.seed = 0
        detector = "langdetect"
        for doc in sample:
            text = doc.strip()
            if len(text) < 30:
                continue
            try:
                langs = detect_langs(text)
                if langs:
                    # Take only the top language per doc; weighting by
                    # confidence inflates noise on short or mixed docs.
                    counts[langs[0].lang] += 1
            except Exception:
                continue
    except ImportError:
        # Script-only fallback: Latin / Devanagari / etc. classifications.
        detector = "script-heuristic"
        for doc in sample:
            counts[_dominant_script(doc[:500])] += 1

    total = sum(counts.values()) or 1
    languages = [
        {"code": code, "share": round(c / total, 3)}
        for code, c in counts.most_common()
    ]
    multilingual = sum(1 for L in languages if L["share"] >= 0.05) > 1

    return {
        "languages": languages,
        "multilingual": multilingual,
        "non_latin_present": non_latin_present,
        "detector": detector,
    }


def _is_non_latin_script(ch: str) -> bool:
    o = ord(ch)
    # Devanagari (Hindi, Marathi), Tamil, Telugu, Kannada, Bengali, Gurmukhi,
    # Gujarati, Malayalam, Oriya, plus CJK and Arabic — broad enough for the
    # multilingual flag without enumerating every block.
    return (
        0x0900 <= o <= 0x097F or  # Devanagari
        0x0980 <= o <= 0x09FF or  # Bengali
        0x0A00 <= o <= 0x0A7F or  # Gurmukhi
        0x0A80 <= o <= 0x0AFF or  # Gujarati
        0x0B00 <= o <= 0x0B7F or  # Oriya
        0x0B80 <= o <= 0x0BFF or  # Tamil
        0x0C00 <= o <= 0x0C7F or  # Telugu
        0x0C80 <= o <= 0x0CFF or  # Kannada
        0x0D00 <= o <= 0x0D7F or  # Malayalam
        0x0600 <= o <= 0x06FF or  # Arabic
        0x4E00 <= o <= 0x9FFF or  # CJK Unified
        0x3040 <= o <= 0x30FF      # Hiragana / Katakana
    )


def _dominant_script(text: str) -> str:
    """Crude script classifier for the langdetect-less fallback path."""
    latin = devanagari = tamil = bengali = cjk = arabic = 0
    for ch in text:
        o = ord(ch)
        if 0x0041 <= o <= 0x007A:
            latin += 1
        elif 0x0900 <= o <= 0x097F:
            devanagari += 1
        elif 0x0B80 <= o <= 0x0BFF:
            tamil += 1
        elif 0x0980 <= o <= 0x09FF:
            bengali += 1
        elif 0x4E00 <= o <= 0x9FFF:
            cjk += 1
        elif 0x0600 <= o <= 0x06FF:
            arabic += 1
    counts = {"latin": latin, "devanagari": devanagari, "tamil": tamil,
              "bengali": bengali, "cjk": cjk, "arabic": arabic}
    return max(counts, key=counts.get) if any(counts.values()) else "unknown"


def _decide_chunking(
    counts: List[int],
    threshold: int = 512,
    fit_target_percentage: float = 90.0,
) -> bool:
    """Chunking is recommended if fewer than `fit_target_percentage` of docs
    fit within the smallest practical context window (`threshold`).

    Why 90%: if more than 10% of documents need truncation under a compact
    embedding model, the user is meaningfully losing information unless they
    chunk or upgrade to a long-context model.
    """
    if not counts:
        return False
    fits = sum(1 for c in counts if c <= threshold)
    return (100.0 * fits / len(counts)) < fit_target_percentage


def _suggest_chunk_size(counts: List[int], threshold: int = 512) -> int:
    """Aim for chunks comfortably below the threshold. We use the median of
    docs that already fit, capped at threshold * 0.85, so embeddings don't
    bump up against the absolute limit."""
    fitting = [c for c in counts if c <= threshold]
    if not fitting:
        return int(threshold * 0.75)
    median_fit = int(statistics.median(fitting))
    return min(median_fit, int(threshold * 0.85)) or int(threshold * 0.5)


def _suggest_overlap(chunk_size: int, percentage: float = 0.15) -> int:
    """Overlap percentage in the typical 10–20% range; default 15%."""
    return max(16, int(round(chunk_size * percentage)))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def analyze_corpus(
    corpus: Union[str, List[str]],
    tokenizer_name: str = "bert-base-uncased",
    overlap_percentage: float = 0.15,
) -> CorpusAnalysis:
    """
    Analyze a cleaned corpus and produce structured statistics + chunking
    guidance.

    Parameters
    ----------
    corpus : str or list[str]
        The cleaned corpus.
    tokenizer_name : str
        Hugging Face tokenizer to use. Default `bert-base-uncased`. Pass the
        tokenizer matching your target embedding model for exact counts.
    overlap_percentage : float
        Desired overlap as a fraction of chunk size, in [0.10, 0.20] is
        typical.

    Returns
    -------
    CorpusAnalysis
    """
    if not 0.0 <= overlap_percentage <= 0.5:
        raise ValueError("overlap_percentage must be in [0.0, 0.5].")

    documents = _split_corpus(corpus)
    notes: List[str] = []

    if not documents:
        notes.append("Corpus is empty after splitting.")
        empty_window = {
            str(w): {"fits": 0, "fit_percentage": 0.0}
            for w in CONTEXT_WINDOW_THRESHOLDS
        }
        return CorpusAnalysis(
            doc_count=0,
            token_stats=_token_stats([]),
            context_window_recommendations=empty_window,
            chunking_needed=False,
            suggested_chunk_size=None,
            suggested_overlap=None,
            vocabulary_metrics={"unique_tokens": 0, "total_tokens": 0, "type_token_ratio": 0.0},
            domain_indicators=[],
            tokenizer_used="",
            notes=notes,
            language_profile={"languages": [], "multilingual": False,
                              "non_latin_present": False, "detector": "none"},
        )

    tokenizer = _Tokenizer(tokenizer_name)
    counts = [tokenizer.count(doc) for doc in documents]

    chunking_needed = _decide_chunking(counts, threshold=512)
    suggested_chunk_size: Optional[int] = None
    suggested_overlap: Optional[int] = None
    if chunking_needed:
        suggested_chunk_size = _suggest_chunk_size(counts, threshold=512)
        suggested_overlap = _suggest_overlap(suggested_chunk_size, overlap_percentage)
        notes.append(
            f"Chunking recommended: {sum(1 for c in counts if c > 512)}/{len(counts)} "
            f"documents exceed 512 tokens."
        )

    if max(counts) > 4 * (statistics.median(counts) or 1):
        notes.append(
            f"Long-tail outliers present (max={max(counts)} tokens). Consider per-document "
            "chunking decisions or truncation."
        )

    language_profile = _detect_language_profile(documents)
    if language_profile.get("multilingual"):
        notes.append(
            f"Multilingual corpus detected ({language_profile['languages']}). "
            "Recommender will prefer multilingual embedding models."
        )
    elif language_profile.get("non_latin_present"):
        notes.append("Non-Latin script content detected; multilingual model recommended.")

    return CorpusAnalysis(
        doc_count=len(documents),
        token_stats=_token_stats(counts),
        context_window_recommendations=_context_window_fit(counts),
        chunking_needed=chunking_needed,
        suggested_chunk_size=suggested_chunk_size,
        suggested_overlap=suggested_overlap,
        vocabulary_metrics=_vocabulary_metrics(documents),
        domain_indicators=_domain_indicators(documents),
        tokenizer_used=tokenizer.kind(),
        notes=notes,
        language_profile=language_profile,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "The quick brown fox jumps over the lazy dog.\n\n"
        + "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 50
        + "\n\nA short note about onboarding and embeddings."
    )
    result = analyze_corpus(sample)
    print(json.dumps(result.to_dict(), indent=2))
