"""
Empirical evaluation of embedding-model candidates on the user's actual corpus.

The heuristic recommender (`synthesize_heuristic_recommendation`) tells you
which model SHOULD work given hardware / corpus shape / task. This module
tells you which model DOES work by:

  1. Sampling N documents from the corpus.
  2. Asking the local LLM to write a realistic search query for each.
  3. Embedding all docs + queries with each candidate model.
  4. Computing MRR, nDCG@10, recall@5, recall@10 per model.
  5. Returning a comparison table the caller can render.

Why this matters: two corpora with identical token stats can have wildly
different retrieval difficulty (technical jargon vs everyday language,
near-duplicates, ambiguous queries). The heuristic is a strong prior but
a weak posterior. Empirical eval converts "MiniLM probably works" into
"mpnet beats MiniLM by 19 MRR points on YOUR data — use mpnet."

This is opt-in (slow: 1-5 minutes for typical corpora + cached models).
Exposed via:
  • POST /evaluate (API)
  • main.py --evaluate flag (CLI)
  • UI "Run empirical evaluation" button

Honest constraints:
  • Synthetic queries are only as good as the LLM that wrote them. Default
    is whatever Ollama model is configured; better LLMs → better queries
    → less noisy comparisons.
  • 30 queries is enough to separate models that differ by ≥10 MRR points
    but won't reliably distinguish 1-2 point gaps. Bump to 100+ if you
    need sharper estimates (~3x slower).
  • The relevance signal is "this query was written from this doc" — a
    proxy for true human-judged relevance. Good enough for relative
    ranking; not a substitute for a labeled eval set.
"""

from __future__ import annotations

import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("evaluator")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[evaluator] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    """Per-model metrics from one evaluation run."""
    model: str
    mrr: float
    ndcg_at_10: float
    recall_at_5: float
    recall_at_10: float
    elapsed_seconds: float
    n_docs_embedded: int
    n_queries: int
    error: Optional[str] = None  # populated if this model failed; metrics will be 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "mrr": round(self.mrr, 4),
            "ndcg_at_10": round(self.ndcg_at_10, 4),
            "recall_at_5": round(self.recall_at_5, 4),
            "recall_at_10": round(self.recall_at_10, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "n_docs_embedded": self.n_docs_embedded,
            "n_queries": self.n_queries,
            "error": self.error,
        }


@dataclass
class EvalReport:
    """All-up evaluation output."""
    results: List[EvalResult] = field(default_factory=list)
    n_queries_generated: int = 0
    query_source: str = "llm"  # "llm" or "keyword-fallback"
    llm_model_used: Optional[str] = None
    heuristic_top: Optional[str] = None  # model the heuristic picked first
    empirical_top: Optional[str] = None  # model with highest MRR
    diverged: bool = False               # heuristic top != empirical top
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "n_queries_generated": self.n_queries_generated,
            "query_source": self.query_source,
            "llm_model_used": self.llm_model_used,
            "heuristic_top": self.heuristic_top,
            "empirical_top": self.empirical_top,
            "diverged": self.diverged,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Corpus splitting (reuse the same convention corpus_analyzer uses)
# ---------------------------------------------------------------------------
def _split_corpus(corpus: str) -> List[str]:
    """Same blank-line split corpus_analyzer uses. Empty entries dropped."""
    if not corpus:
        return []
    chunks = [c.strip() for c in corpus.split("\n\n") if c.strip()]
    return chunks if chunks else [corpus]


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------
_QUERY_GEN_PROMPT = (
    "You are generating a single realistic search query that someone would "
    "type into a search box to find the following passage. Output ONLY the "
    "query — no quotes, no preamble, no explanation. Keep it under 15 words "
    "and conversational, not a verbatim sentence from the passage.\n\n"
    "Passage:\n{passage}\n\nQuery:"
)


def _keyword_query_from_doc(doc: str, n_keywords: int = 4) -> str:
    """Fallback query generator when no LLM is available. Picks the N
    most distinctive (longest non-stopword) tokens. Lossy — the queries
    are not natural-language but they're still useful relative-rank signals."""
    stopwords = {
        "the", "and", "for", "with", "that", "this", "have", "from", "they",
        "been", "were", "their", "would", "there", "what", "which", "when",
        "where", "while", "your", "about", "into", "such", "than",
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{4,}", doc.lower())
    tokens = [t for t in tokens if t not in stopwords]
    # Pick the longest distinct tokens (proxy for "distinctive").
    seen, picked = set(), []
    for t in sorted(set(tokens), key=len, reverse=True):
        if t not in seen:
            picked.append(t)
            seen.add(t)
        if len(picked) >= n_keywords:
            break
    return " ".join(picked)


def generate_eval_queries(
    docs: List[str],
    n: int = 30,
    llm: Optional[Any] = None,
    seed: int = 0,
    max_passage_chars: int = 1200,
) -> Tuple[List[Tuple[str, int]], str]:
    """Sample N docs and produce one query per doc. Returns:

        ([(query, doc_index), ...], query_source)

    where query_source is "llm" if the LLM was used, "keyword-fallback"
    if we fell back to keyword extraction.

    Sampling is deterministic given `seed` so identical inputs produce
    identical eval sets (matters for reproducibility / comparison runs).
    """
    if not docs:
        return [], "none"
    rng = random.Random(seed)
    n = min(n, len(docs))
    indices = rng.sample(range(len(docs)), n)

    if llm is None:
        logger.info("No LLM provided; using keyword-extraction fallback for queries.")
        pairs = [(_keyword_query_from_doc(docs[i]), i) for i in indices]
        return pairs, "keyword-fallback"

    pairs: List[Tuple[str, int]] = []
    for idx in indices:
        passage = docs[idx][:max_passage_chars]
        prompt = _QUERY_GEN_PROMPT.format(passage=passage)
        try:
            resp = llm.invoke(prompt)
            text = getattr(resp, "content", None) or str(resp)
            query = text.strip().strip('"').strip("'").splitlines()[0].strip()
            if query:
                pairs.append((query, idx))
        except Exception as e:
            logger.warning("LLM query generation failed for doc %d (%s); using keyword fallback for this doc.", idx, e)
            pairs.append((_keyword_query_from_doc(docs[idx]), idx))
    return pairs, "llm"


# ---------------------------------------------------------------------------
# Embedding + metrics
# ---------------------------------------------------------------------------
def _embed_texts(
    model_name: str,
    texts: List[str],
    prefix: str = "",
    batch_size: int = 64,
) -> Any:
    """Load `model_name` via sentence-transformers, encode `texts` with
    optional prefix. Returns a numpy array (M, d) of L2-normalized vectors.
    The model is freed after encoding so memory doesn't accumulate across
    candidate models."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    try:
        inputs = [prefix + t for t in texts] if prefix else texts
        return model.encode(
            inputs,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
    finally:
        del model
        # No explicit gc.collect — Python will reclaim, and we want the
        # next iteration to be fast. If memory is tight, the OS will page.


def _compute_metrics(
    query_embs: Any,
    doc_embs: Any,
    target_indices: List[int],
    k_recall: List[int] = (5, 10),
    k_ndcg: int = 10,
) -> Dict[str, float]:
    """For each query (row i in query_embs), compute the rank of
    target_indices[i] in the cosine-similarity-sorted list of docs.
    Return MRR, nDCG@10, recall@5, recall@10."""
    import numpy as np

    # cosine similarity (vectors are already L2-normalized so dot = cosine).
    sims = query_embs @ doc_embs.T  # (M, N)
    ranks: List[int] = []
    for i, target in enumerate(target_indices):
        # argsort descending. rank is 1-indexed.
        order = np.argsort(-sims[i])
        try:
            rank = int(np.where(order == target)[0][0]) + 1
        except IndexError:
            rank = len(order) + 1   # target somehow not in the corpus
        ranks.append(rank)

    n = len(ranks)
    mrr = sum(1.0 / r for r in ranks) / n
    ndcg_at_k = sum(1.0 / math.log2(r + 1) if r <= k_ndcg else 0.0 for r in ranks) / n
    metrics = {"mrr": mrr, f"ndcg_at_{k_ndcg}": ndcg_at_k}
    for k in k_recall:
        metrics[f"recall_at_{k}"] = sum(1 for r in ranks if r <= k) / n
    return metrics


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------
def evaluate_candidates(
    corpus: str,
    candidates: List[Dict[str, Any]],
    n_queries: int = 30,
    llm: Optional[Any] = None,
    seed: int = 0,
    heuristic_top_model: Optional[str] = None,
) -> EvalReport:
    """Empirically rank `candidates` on `corpus`.

    Parameters
    ----------
    corpus : str
        The (already-PII-redacted, ideally) text corpus.
    candidates : list of dict
        Each entry needs: "name" (str), and optionally "embed_prefix" /
        "query_prefix" if the model uses asymmetric prefixes (BGE / nomic /
        e5). Format matches the entries in `recommended_models`.
    n_queries : int
        How many synthetic queries to generate. 30 is a good speed/accuracy
        balance; bump to 100 for sharper estimates.
    llm : LangChain BaseChatModel or None
        If provided, used to generate natural-language queries. If None,
        falls back to keyword extraction (lossier but no LLM dependency).
    seed : int
        Random seed for query-doc sampling. Default 0 → reproducible runs.
    heuristic_top_model : str or None
        The model the heuristic recommended first. Used to flag divergence
        in the report.

    Returns
    -------
    EvalReport with per-model metrics + a `diverged` flag.
    """
    docs = _split_corpus(corpus)
    if not docs:
        return EvalReport(notes=["Corpus is empty after splitting."])
    if not candidates:
        return EvalReport(notes=["No candidates provided."])

    # 1. Generate queries
    query_pairs, query_source = generate_eval_queries(
        docs, n=n_queries, llm=llm, seed=seed
    )
    if not query_pairs:
        return EvalReport(
            notes=["Query generation produced 0 queries. Check LLM availability."],
            query_source=query_source,
        )

    query_texts = [q for q, _ in query_pairs]
    target_indices = [i for _, i in query_pairs]
    llm_name = None
    if llm is not None:
        # Pull a usable model identifier off whatever LLM was passed in.
        llm_name = (
            getattr(llm, "model", None)
            or getattr(llm, "model_name", None)
            or type(llm).__name__
        )

    # 2. Per-candidate evaluation
    results: List[EvalResult] = []
    # Pre-canonicalize to catch obvious LLM paraphrases ('BGE-large' →
    # 'BAAI/bge-large-en-v1.5'). Without this, sentence-transformers will
    # try to fetch e.g. 'sentence-transformers/BGE-large' from HF and
    # 404. This is belt-and-suspenders — the API merge step should have
    # already canonicalized, but we re-check here so direct callers of
    # evaluate_candidates() are also defended.
    try:
        from src.agent_orchestrator import canonicalize_model_name
    except ImportError:
        canonicalize_model_name = None  # type: ignore

    for cand in candidates:
        name = cand.get("name")
        if not name:
            continue
        embed_prefix = cand.get("embed_prefix", "") or ""
        query_prefix = cand.get("query_prefix", "") or ""
        # Try to canonicalize a non-Hub-shaped name. Models in the
        # catalogue always include a '/' org prefix; LLM paraphrases
        # often don't.
        if canonicalize_model_name and "/" not in name:
            canonical = canonicalize_model_name(name)
            if canonical:
                logger.info("Canonicalized model name %r -> %r", name, canonical)
                name = canonical
        t0 = time.time()
        try:
            doc_embs = _embed_texts(name, docs, prefix=embed_prefix)
            query_embs = _embed_texts(name, query_texts, prefix=query_prefix)
            metrics = _compute_metrics(query_embs, doc_embs, target_indices)
            results.append(EvalResult(
                model=name,  # canonical form, may differ from cand["name"]
                mrr=metrics["mrr"],
                ndcg_at_10=metrics["ndcg_at_10"],
                recall_at_5=metrics["recall_at_5"],
                recall_at_10=metrics["recall_at_10"],
                elapsed_seconds=time.time() - t0,
                n_docs_embedded=len(docs),
                n_queries=len(query_pairs),
            ))
            logger.info(
                "%s: MRR=%.3f nDCG@10=%.3f recall@10=%.3f (%.1fs)",
                name, metrics["mrr"], metrics["ndcg_at_10"], metrics["recall_at_10"],
                time.time() - t0,
            )
        except Exception as e:
            logger.warning("Eval failed for %s: %s", name, e)
            results.append(EvalResult(
                model=name,
                mrr=0.0, ndcg_at_10=0.0, recall_at_5=0.0, recall_at_10=0.0,
                elapsed_seconds=time.time() - t0,
                n_docs_embedded=0,
                n_queries=len(query_pairs),
                error=f"{type(e).__name__}: {e}",
            ))

    # 3. Determine empirical winner + divergence flag
    successful = [r for r in results if not r.error]
    empirical_top = max(successful, key=lambda r: r.mrr).model if successful else None
    diverged = (
        bool(heuristic_top_model and empirical_top)
        and heuristic_top_model != empirical_top
    )

    notes: List[str] = []
    if query_source == "keyword-fallback":
        notes.append(
            "Queries were generated by keyword extraction (no LLM available). "
            "Metrics still rank candidates correctly relative to each other but "
            "are not directly comparable to MTEB-style benchmarks."
        )
    if any(r.error for r in results):
        failed = [r.model for r in results if r.error]
        notes.append(f"Some candidates failed to load: {failed}. Their metrics are 0.")
    if diverged:
        notes.append(
            f"Empirical winner ({empirical_top}) differs from heuristic top "
            f"({heuristic_top_model}). Consider using the empirical winner."
        )

    return EvalReport(
        results=results,
        n_queries_generated=len(query_pairs),
        query_source=query_source,
        llm_model_used=llm_name,
        heuristic_top=heuristic_top_model,
        empirical_top=empirical_top,
        diverged=diverged,
        notes=notes,
    )
