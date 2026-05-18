"""
Empirical evaluation of embedding-model candidates on the user's actual corpus.

The heuristic recommender (`synthesize_heuristic_recommendation`) tells you
which model SHOULD work given hardware / corpus shape / task. This module
tells you which model DOES work by actually running each candidate on
synthetic supervision derived from the corpus itself.

Task-aware evaluation
---------------------
Different downstream tasks need different metrics. Comparing models on
retrieval MRR when the user said `task=classification` is meaningless —
classifiers need separation in feature space, not query/passage similarity.

Per task:
  • retrieval      — LLM writes one query per sampled doc; metric: MRR /
                     nDCG@10 / recall@k against the source doc.
  • classification — LLM assigns one of K pseudo-labels per sampled doc;
                     5-fold stratified CV with logistic regression on the
                     embeddings; metric: macro-F1.
  • clustering     — LLM groups sampled docs into K topics; KMeans(K) on
                     the embeddings; metric: V-measure vs LLM labels.
  • deduplication  — LLM paraphrases each sampled doc (positive); random
                     non-pairs (negative); metric: ROC-AUC of cosine
                     similarity discriminating the two.
  • similarity     — LLM rates (doc, derived-query) pairs on a 1-5 scale;
                     metric: Spearman correlation between LLM scores and
                     embedding cosine similarity.

Every per-model result populates `primary_metric_name` /
`primary_metric_value`, so downstream consumers (Decider, UI) can rank
candidates uniformly regardless of which task ran. Retrieval still
populates `mrr` / `ndcg_at_10` / `recall_at_k` for back-compat.

This is opt-in (slow: 1-5 minutes for typical corpora + cached models).
Exposed via:
  • POST /evaluate (API) and /evaluate/upload (UI multipart)
  • main.py --evaluate flag (CLI)
  • UI "Run empirical evaluation" button

Honest constraints:
  • Synthetic supervision is only as good as the LLM that produced it.
    Better LLMs → less noisy comparisons. With no LLM available, retrieval
    falls back to keyword-extracted queries; other tasks degrade harder
    (see `_no_llm_<task>_fallback`).
  • 30 samples is enough to separate models that differ meaningfully
    (≥0.05 on whatever primary metric the task uses). For sharper
    estimates bump n_queries to 100+ (~3x slower).
  • The relevance / label / paraphrase signal is LLM-generated. Good
    enough for relative ranking; not a substitute for a hand-labeled
    eval set.
"""

from __future__ import annotations

import json
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
    """Per-model metrics from one evaluation run.

    `primary_metric_name` / `primary_metric_value` is what the Decider
    and UI rank on — uniform across tasks (mrr for retrieval, f1_macro
    for classification, v_measure for clustering, auc for dedup,
    spearman for similarity). The legacy retrieval-specific fields
    (mrr / ndcg_at_10 / recall_at_k) stay populated for task=retrieval
    and are zero for other tasks.
    """
    model: str
    mrr: float
    ndcg_at_10: float
    recall_at_5: float
    recall_at_10: float
    elapsed_seconds: float
    n_docs_embedded: int
    n_queries: int
    error: Optional[str] = None  # populated if this model failed; metrics will be 0
    # Task-aware primary metric. Default values keep retrieval behavior
    # unchanged: primary_metric_name="mrr", value=self.mrr after the
    # retrieval evaluator finishes (set explicitly so callers reading the
    # dict always get a non-empty primary metric).
    task: str = "retrieval"
    primary_metric_name: str = "mrr"
    primary_metric_value: float = 0.0
    # Free-form extras the task evaluator wants to surface (e.g. number of
    # label classes for classification, K for clustering). Kept open so
    # we don't have to grow the schema for every task tweak.
    extra: Dict[str, Any] = field(default_factory=dict)

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
            "task": self.task,
            "primary_metric_name": self.primary_metric_name,
            "primary_metric_value": round(self.primary_metric_value, 4),
            "extra": self.extra,
        }


@dataclass
class EvalReport:
    """All-up evaluation output."""
    results: List[EvalResult] = field(default_factory=list)
    n_queries_generated: int = 0
    query_source: str = "llm"  # "llm" or "keyword-fallback"
    llm_model_used: Optional[str] = None
    heuristic_top: Optional[str] = None  # model the heuristic picked first
    empirical_top: Optional[str] = None  # model with highest primary metric
    diverged: bool = False               # heuristic top != empirical top
    notes: List[str] = field(default_factory=list)
    # Task this report was produced for. Drives which metric the UI /
    # Decider rank on. Defaults to "retrieval" for back-compat.
    task: str = "retrieval"
    primary_metric_name: str = "mrr"

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
            "task": self.task,
            "primary_metric_name": self.primary_metric_name,
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
def _lookup_eval_batch_size(model_name: str, default: int = 64) -> int:
    """Some models (BGE-M3 in particular) crash on MPS with the default
    batch size because the attention tile (batch * seqlen * heads * d)
    overflows a 2**32-element tensor. The catalogue records a per-model
    `eval_batch_size` for those cases; we look it up here so the evaluator
    doesn't need to know the catalog schema."""
    try:
        from src.agent_orchestrator import EMBEDDING_CATALOGUE
    except ImportError:
        return default
    for entry in EMBEDDING_CATALOGUE:
        if entry.get("name") == model_name:
            return int(entry.get("eval_batch_size", default))
    return default


def _embed_texts(
    model_name: str,
    texts: List[str],
    prefix: str = "",
    batch_size: Optional[int] = None,
) -> Any:
    """Load `model_name` via sentence-transformers, encode `texts` with
    optional prefix. Returns a numpy array (M, d) of L2-normalized vectors.
    The model is freed after encoding so memory doesn't accumulate across
    candidate models.

    `batch_size` defaults to whatever the catalogue says for this model
    (most models: 64; BGE-M3: 8 to avoid the MPS 2**32-element crash on
    long-context inputs). Pass an explicit int to override."""
    from sentence_transformers import SentenceTransformer

    if batch_size is None:
        batch_size = _lookup_eval_batch_size(model_name)

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
# Task-aware evaluation
# ---------------------------------------------------------------------------
#
# Each downstream task needs its own metric. We dispatch from
# `evaluate_candidates` based on the `task` kwarg. Each per-task
# evaluator returns a list of EvalResult — one per candidate — with the
# `primary_metric_name` / `primary_metric_value` fields populated, so the
# Decider and UI can compare candidates uniformly regardless of task.

# What each task's primary metric is called (drives UI column header and
# Decider's swap logic).
TASK_PRIMARY_METRIC: Dict[str, str] = {
    "retrieval": "mrr",
    "classification": "f1_macro",
    "clustering": "v_measure",
    "deduplication": "auc",
    "similarity": "spearman",
}

# The minimum "real" margin a non-heuristic winner must beat the
# heuristic top by, per task. Below this is sampling noise. These
# mirror the Decider's swap thresholds — the two agents must agree on
# what counts as "real" so the UI's "diverged" hint and the Decider's
# decision basis don't contradict each other.
TASK_DIVERGENCE_MARGIN: Dict[str, float] = {
    "retrieval": 0.05,        # MRR
    "classification": 0.05,   # F1 macro
    "clustering": 0.05,       # V-measure
    "deduplication": 0.02,    # AUC (small absolute moves are real)
    "similarity": 0.05,       # Spearman ρ
}


def _resolve_canonical_name(name: str) -> str:
    """Try to map a (possibly LLM-paraphrased) candidate name to its
    canonical Hugging Face identifier. No-ops if agent_orchestrator
    isn't importable (defensive — direct callers of this module shouldn't
    have to depend on the agent stack)."""
    try:
        from src.agent_orchestrator import canonicalize_model_name
    except ImportError:
        return name
    if canonicalize_model_name and "/" not in name:
        canonical = canonicalize_model_name(name)
        if canonical:
            logger.info("Canonicalized model name %r -> %r", name, canonical)
            return canonical
    return name


# ---------------------------------------------------------------------------
# Retrieval (existing behavior — unchanged metric, now wrapped uniformly)
# ---------------------------------------------------------------------------
def _eval_retrieval_one(
    name: str, embed_prefix: str, query_prefix: str,
    docs: List[str], query_texts: List[str],
    target_indices: List[int], n_queries: int,
) -> EvalResult:
    t0 = time.time()
    try:
        doc_embs = _embed_texts(name, docs, prefix=embed_prefix)
        query_embs = _embed_texts(name, query_texts, prefix=query_prefix)
        metrics = _compute_metrics(query_embs, doc_embs, target_indices)
        elapsed = time.time() - t0
        logger.info(
            "%s [retrieval]: MRR=%.3f nDCG@10=%.3f recall@10=%.3f (%.1fs)",
            name, metrics["mrr"], metrics["ndcg_at_10"], metrics["recall_at_10"], elapsed,
        )
        return EvalResult(
            model=name,
            mrr=metrics["mrr"],
            ndcg_at_10=metrics["ndcg_at_10"],
            recall_at_5=metrics["recall_at_5"],
            recall_at_10=metrics["recall_at_10"],
            elapsed_seconds=elapsed,
            n_docs_embedded=len(docs),
            n_queries=n_queries,
            task="retrieval",
            primary_metric_name="mrr",
            primary_metric_value=metrics["mrr"],
        )
    except Exception as e:
        logger.warning("Eval failed for %s: %s", name, e)
        return EvalResult(
            model=name,
            mrr=0.0, ndcg_at_10=0.0, recall_at_5=0.0, recall_at_10=0.0,
            elapsed_seconds=time.time() - t0,
            n_docs_embedded=0,
            n_queries=n_queries,
            error=f"{type(e).__name__}: {e}",
            task="retrieval",
            primary_metric_name="mrr",
            primary_metric_value=0.0,
        )


# ---------------------------------------------------------------------------
# Classification — LLM produces categorical pseudo-labels, then we score
# embedding linear separability via 5-fold stratified CV macro-F1.
# ---------------------------------------------------------------------------
_CLASSIFY_PROMPT = (
    "You will be given a list of short text snippets, numbered 0..N-1. "
    "Group them into 2 to 5 meaningful topical categories. Output ONLY a "
    "JSON object mapping each snippet index (as a string) to a short "
    "lowercase category label (1-2 words). No preamble, no explanation.\n\n"
    "Snippets:\n{snippets}\n\nJSON:"
)


def _llm_assign_labels(
    docs: List[str], llm: Any, seed: int, max_n: int = 40, max_chars: int = 200,
) -> Tuple[List[int], List[str]]:
    """Returns (sample_indices, labels). One label per sample_index, in the
    same order. Empty lists if the LLM call or parsing fails."""
    rng = random.Random(seed)
    n = min(max_n, len(docs))
    sample_idx = sorted(rng.sample(range(len(docs)), n))
    snippets = "\n".join(f"[{i}] {docs[idx][:max_chars]}" for i, idx in enumerate(sample_idx))
    try:
        resp = llm.invoke(_CLASSIFY_PROMPT.format(snippets=snippets))
        text = getattr(resp, "content", None) or str(resp)
    except Exception as e:
        logger.warning("LLM label assignment failed: %s", e)
        return [], []

    # Extract JSON object — first { through last } is good enough for our use.
    first, last = text.find("{"), text.rfind("}")
    if first < 0 or last <= first:
        return [], []
    try:
        parsed = json.loads(text[first:last + 1])
    except (ValueError, json.JSONDecodeError) as e:
        logger.warning("Couldn't parse LLM label JSON: %s", e)
        return [], []

    if not isinstance(parsed, dict):
        return [], []
    # Index strings → labels; missing entries get the most-common label later.
    labels: List[str] = []
    kept_indices: List[int] = []
    for i, doc_idx in enumerate(sample_idx):
        lab = parsed.get(str(i)) or parsed.get(i)
        if lab is None:
            continue
        if not isinstance(lab, str) or not lab.strip():
            continue
        labels.append(lab.strip().lower())
        kept_indices.append(doc_idx)
    return kept_indices, labels


def _eval_classification_one(
    name: str, embed_prefix: str,
    sample_docs: List[str], labels: List[str], seed: int,
) -> EvalResult:
    """5-fold stratified CV with logistic regression on the embeddings.
    Primary metric: macro-F1 (insensitive to class imbalance)."""
    t0 = time.time()
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import LabelEncoder
        import numpy as np

        X = _embed_texts(name, sample_docs, prefix=embed_prefix)
        y = LabelEncoder().fit_transform(labels)
        # Stratified-K needs at least k samples per class; downshift K
        # if any class is too small.
        min_class_count = int(np.bincount(y).min())
        k = max(2, min(5, min_class_count))
        if min_class_count < 2:
            # Can't stratify-CV with singleton classes — fall back to
            # 80/20 holdout.
            split = int(len(y) * 0.8)
            order = np.arange(len(y))
            rng = np.random.default_rng(seed)
            rng.shuffle(order)
            train_idx, test_idx = order[:split], order[split:]
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X[train_idx], y[train_idx])
            from sklearn.metrics import f1_score
            preds = clf.predict(X[test_idx])
            f1 = float(f1_score(y[test_idx], preds, average="macro"))
        else:
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            scores = cross_val_score(
                LogisticRegression(max_iter=1000), X, y,
                cv=cv, scoring="f1_macro",
            )
            f1 = float(scores.mean())
        elapsed = time.time() - t0
        n_classes = int(len(set(labels)))
        logger.info(
            "%s [classification]: macro-F1=%.3f (k=%d classes, %d samples, %.1fs)",
            name, f1, n_classes, len(sample_docs), elapsed,
        )
        return EvalResult(
            model=name, mrr=0.0, ndcg_at_10=0.0, recall_at_5=0.0, recall_at_10=0.0,
            elapsed_seconds=elapsed,
            n_docs_embedded=len(sample_docs),
            n_queries=len(sample_docs),
            task="classification",
            primary_metric_name="f1_macro",
            primary_metric_value=f1,
            extra={"n_classes": n_classes, "n_samples": len(sample_docs)},
        )
    except Exception as e:
        logger.warning("Classification eval failed for %s: %s", name, e)
        return EvalResult(
            model=name, mrr=0.0, ndcg_at_10=0.0, recall_at_5=0.0, recall_at_10=0.0,
            elapsed_seconds=time.time() - t0,
            n_docs_embedded=0, n_queries=0,
            error=f"{type(e).__name__}: {e}",
            task="classification",
            primary_metric_name="f1_macro",
            primary_metric_value=0.0,
        )


# ---------------------------------------------------------------------------
# Clustering — LLM groups docs into K topics; KMeans(K) on embeddings;
# V-measure between LLM labels (ground truth) and KMeans labels.
# ---------------------------------------------------------------------------
def _eval_clustering_one(
    name: str, embed_prefix: str,
    sample_docs: List[str], labels: List[str], seed: int,
) -> EvalResult:
    t0 = time.time()
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import v_measure_score
        from sklearn.preprocessing import LabelEncoder
        import numpy as np

        X = _embed_texts(name, sample_docs, prefix=embed_prefix)
        y = LabelEncoder().fit_transform(labels)
        k = int(len(set(labels)))
        if k < 2:
            raise ValueError(f"Need at least 2 clusters, got {k}")
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
        v = float(v_measure_score(y, km.labels_))
        elapsed = time.time() - t0
        logger.info(
            "%s [clustering]: V-measure=%.3f (k=%d, %d samples, %.1fs)",
            name, v, k, len(sample_docs), elapsed,
        )
        return EvalResult(
            model=name, mrr=0.0, ndcg_at_10=0.0, recall_at_5=0.0, recall_at_10=0.0,
            elapsed_seconds=elapsed,
            n_docs_embedded=len(sample_docs),
            n_queries=len(sample_docs),
            task="clustering",
            primary_metric_name="v_measure",
            primary_metric_value=v,
            extra={"k": k, "n_samples": len(sample_docs)},
        )
    except Exception as e:
        logger.warning("Clustering eval failed for %s: %s", name, e)
        return EvalResult(
            model=name, mrr=0.0, ndcg_at_10=0.0, recall_at_5=0.0, recall_at_10=0.0,
            elapsed_seconds=time.time() - t0,
            n_docs_embedded=0, n_queries=0,
            error=f"{type(e).__name__}: {e}",
            task="clustering",
            primary_metric_name="v_measure",
            primary_metric_value=0.0,
        )


# ---------------------------------------------------------------------------
# Deduplication — LLM paraphrases each sampled doc (positive pair); we pair
# each sample with a different random doc (negative pair); metric: AUC of
# cosine similarity discriminating positives from negatives.
# ---------------------------------------------------------------------------
_PARAPHRASE_PROMPT = (
    "Rewrite the following passage in completely different words while "
    "preserving the meaning. Output ONLY the rewritten passage — no preamble.\n\n"
    "Passage:\n{passage}\n\nRewrite:"
)


def _llm_paraphrase_docs(
    docs: List[str], llm: Any, seed: int, max_n: int = 30, max_chars: int = 800,
) -> Tuple[List[int], List[str]]:
    """Returns (sample_indices, paraphrases). Empty lists if no LLM available."""
    if llm is None:
        return [], []
    rng = random.Random(seed)
    n = min(max_n, len(docs))
    sample_idx = sorted(rng.sample(range(len(docs)), n))
    paraphrases: List[str] = []
    kept: List[int] = []
    for idx in sample_idx:
        try:
            resp = llm.invoke(_PARAPHRASE_PROMPT.format(passage=docs[idx][:max_chars]))
            text = getattr(resp, "content", None) or str(resp)
            p = text.strip().strip('"').strip("'")
            if p and len(p) >= 10:
                paraphrases.append(p)
                kept.append(idx)
        except Exception as e:
            logger.warning("Paraphrase failed for doc %d (%s); skipping.", idx, e)
    return kept, paraphrases


def _eval_deduplication_one(
    name: str, embed_prefix: str,
    sample_docs: List[str], paraphrases: List[str], seed: int,
) -> EvalResult:
    t0 = time.time()
    try:
        from sklearn.metrics import roc_auc_score
        import numpy as np

        rng = np.random.default_rng(seed)
        anchors = _embed_texts(name, sample_docs, prefix=embed_prefix)
        positives = _embed_texts(name, paraphrases, prefix=embed_prefix)
        # Positive pair scores: cosine(anchor_i, paraphrase_i)
        pos_scores = np.sum(anchors * positives, axis=1)
        # Negative pair scores: each anchor paired with a random *different*
        # anchor's paraphrase. Use a derangement-ish permutation.
        perm = np.arange(len(anchors))
        rng.shuffle(perm)
        # Repair any fixed points (i==perm[i]) by swapping into the next slot.
        for i in range(len(perm)):
            if perm[i] == i:
                j = (i + 1) % len(perm)
                perm[i], perm[j] = perm[j], perm[i]
        neg_scores = np.sum(anchors * positives[perm], axis=1)
        y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
        y_score = np.concatenate([pos_scores, neg_scores])
        auc = float(roc_auc_score(y_true, y_score))
        elapsed = time.time() - t0
        logger.info(
            "%s [deduplication]: AUC=%.3f (%d positive/negative pairs each, %.1fs)",
            name, auc, len(anchors), elapsed,
        )
        return EvalResult(
            model=name, mrr=0.0, ndcg_at_10=0.0, recall_at_5=0.0, recall_at_10=0.0,
            elapsed_seconds=elapsed,
            n_docs_embedded=len(sample_docs) + len(paraphrases),
            n_queries=len(sample_docs),
            task="deduplication",
            primary_metric_name="auc",
            primary_metric_value=auc,
            extra={"n_pairs": len(anchors)},
        )
    except Exception as e:
        logger.warning("Deduplication eval failed for %s: %s", name, e)
        return EvalResult(
            model=name, mrr=0.0, ndcg_at_10=0.0, recall_at_5=0.0, recall_at_10=0.0,
            elapsed_seconds=time.time() - t0,
            n_docs_embedded=0, n_queries=0,
            error=f"{type(e).__name__}: {e}",
            task="deduplication",
            primary_metric_name="auc",
            primary_metric_value=0.0,
        )


# ---------------------------------------------------------------------------
# Similarity — LLM rates (query, doc) pair similarity on a 1-5 Likert scale;
# Spearman ρ between LLM scores and cosine similarities.
# Uses retrieval's query generator to derive queries, then mixes correct
# (query_i, doc_i) pairs with random (query_i, doc_j) pairs and asks the
# LLM to rate each.
# ---------------------------------------------------------------------------
_SIMILARITY_RATE_PROMPT = (
    "Rate how related the following query is to the passage on a scale of "
    "1 to 5, where:\n"
    "  1 = unrelated\n"
    "  2 = barely related\n"
    "  3 = partially related\n"
    "  4 = strongly related\n"
    "  5 = the passage directly answers the query\n\n"
    "Output ONLY a single integer 1-5. No preamble, no explanation.\n\n"
    "Query: {query}\nPassage: {passage}\n\nScore:"
)


def _llm_rate_similarity_pairs(
    query_doc_pairs: List[Tuple[str, str]], llm: Any, max_chars: int = 600,
) -> List[Optional[int]]:
    """Returns a list of int ratings (or None if unparseable) — same
    length / order as the input pairs."""
    ratings: List[Optional[int]] = []
    for q, d in query_doc_pairs:
        try:
            resp = llm.invoke(
                _SIMILARITY_RATE_PROMPT.format(query=q, passage=d[:max_chars])
            )
            text = (getattr(resp, "content", None) or str(resp)).strip()
            # First digit 1-5 wins.
            m = re.search(r"[1-5]", text)
            ratings.append(int(m.group(0)) if m else None)
        except Exception as e:
            logger.warning("Similarity rating failed: %s", e)
            ratings.append(None)
    return ratings


def _eval_similarity_one(
    name: str, embed_prefix: str, query_prefix: str,
    queries: List[str], docs_for_pairs: List[str],
    llm_scores: List[float],
) -> EvalResult:
    """Spearman correlation between LLM Likert scores and cosine
    similarity of (query, doc) pairs in the same order."""
    t0 = time.time()
    try:
        from scipy.stats import spearmanr
        import numpy as np

        q_embs = _embed_texts(name, queries, prefix=query_prefix or embed_prefix)
        d_embs = _embed_texts(name, docs_for_pairs, prefix=embed_prefix)
        cos = np.sum(q_embs * d_embs, axis=1)
        rho, _p = spearmanr(cos, llm_scores)
        rho = float(rho) if rho == rho else 0.0  # NaN guard
        elapsed = time.time() - t0
        logger.info(
            "%s [similarity]: Spearman=%.3f (%d pairs, %.1fs)",
            name, rho, len(queries), elapsed,
        )
        return EvalResult(
            model=name, mrr=0.0, ndcg_at_10=0.0, recall_at_5=0.0, recall_at_10=0.0,
            elapsed_seconds=elapsed,
            n_docs_embedded=len(queries) + len(docs_for_pairs),
            n_queries=len(queries),
            task="similarity",
            primary_metric_name="spearman",
            primary_metric_value=rho,
            extra={"n_pairs": len(queries)},
        )
    except Exception as e:
        logger.warning("Similarity eval failed for %s: %s", name, e)
        return EvalResult(
            model=name, mrr=0.0, ndcg_at_10=0.0, recall_at_5=0.0, recall_at_10=0.0,
            elapsed_seconds=time.time() - t0,
            n_docs_embedded=0, n_queries=0,
            error=f"{type(e).__name__}: {e}",
            task="similarity",
            primary_metric_name="spearman",
            primary_metric_value=0.0,
        )


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------
def _llm_name(llm: Optional[Any]) -> Optional[str]:
    if llm is None:
        return None
    return (
        getattr(llm, "model", None)
        or getattr(llm, "model_name", None)
        or type(llm).__name__
    )


def evaluate_candidates(
    corpus: str,
    candidates: List[Dict[str, Any]],
    n_queries: int = 30,
    llm: Optional[Any] = None,
    seed: int = 0,
    heuristic_top_model: Optional[str] = None,
    task: str = "retrieval",
) -> EvalReport:
    """Empirically rank `candidates` on `corpus` for the given `task`.

    Parameters
    ----------
    corpus : str
        The (already-PII-redacted, ideally) text corpus.
    candidates : list of dict
        Each entry needs: "name" (str), and optionally "embed_prefix" /
        "query_prefix" for asymmetric models. Format matches the entries
        in `recommended_models`.
    n_queries : int
        How many synthetic samples to generate. For retrieval this is
        query count; for classification/clustering it's the number of
        sampled docs that get LLM-assigned labels; for dedup it's the
        number of (anchor, paraphrase) pairs; for similarity it's the
        number of (query, doc) pairs (half correct, half random).
    llm : LangChain BaseChatModel or None
        If provided, drives LLM-side supervision (queries for retrieval,
        labels for classification/clustering, paraphrases for dedup,
        ratings for similarity). For retrieval, falls back to keyword
        extraction if None. Other tasks return empty/error reports
        without an LLM — the supervision is the LLM.
    seed : int
        Random seed for sampling. Default 0 → reproducible runs.
    heuristic_top_model : str or None
        The model the heuristic recommended first. Used to compute the
        `diverged` flag against the empirical winner using the
        task-appropriate margin (see TASK_DIVERGENCE_MARGIN).
    task : str
        retrieval | classification | clustering | deduplication | similarity.
        Falls back to retrieval if unknown.

    Returns
    -------
    EvalReport with per-model EvalResult — every result has
    primary_metric_name / primary_metric_value populated so the Decider
    and UI can rank uniformly regardless of which task ran.
    """
    if task not in TASK_PRIMARY_METRIC:
        task = "retrieval"
    docs = _split_corpus(corpus)
    if not docs:
        return EvalReport(notes=["Corpus is empty after splitting."], task=task,
                          primary_metric_name=TASK_PRIMARY_METRIC[task])
    if not candidates:
        return EvalReport(notes=["No candidates provided."], task=task,
                          primary_metric_name=TASK_PRIMARY_METRIC[task])

    llm_name = _llm_name(llm)

    if task == "retrieval":
        return _run_retrieval(
            docs, candidates, n_queries, llm, seed,
            heuristic_top_model, llm_name,
        )
    if task == "classification":
        return _run_classification(
            docs, candidates, n_queries, llm, seed,
            heuristic_top_model, llm_name,
        )
    if task == "clustering":
        return _run_clustering(
            docs, candidates, n_queries, llm, seed,
            heuristic_top_model, llm_name,
        )
    if task == "deduplication":
        return _run_deduplication(
            docs, candidates, n_queries, llm, seed,
            heuristic_top_model, llm_name,
        )
    if task == "similarity":
        return _run_similarity(
            docs, candidates, n_queries, llm, seed,
            heuristic_top_model, llm_name,
        )
    # Unreachable due to the guard above, but keep the explicit branch
    # for static analysis.
    return EvalReport(notes=[f"Unsupported task '{task}'"], task=task)


def _finalize_report(
    results: List[EvalResult],
    task: str,
    n_supervision_samples: int,
    query_source: str,
    llm_name: Optional[str],
    heuristic_top_model: Optional[str],
    notes_extra: Optional[List[str]] = None,
) -> EvalReport:
    """Compute empirical_top + diverged flag + user-visible notes using
    the task-appropriate primary metric and divergence margin. Shared
    across all task evaluators so the comparison logic stays consistent."""
    primary = TASK_PRIMARY_METRIC[task]
    margin_threshold = TASK_DIVERGENCE_MARGIN[task]

    successful = [r for r in results if not r.error]
    empirical_top_result = (
        max(successful, key=lambda r: r.primary_metric_value) if successful else None
    )
    empirical_top = empirical_top_result.model if empirical_top_result else None

    heuristic_metric: Optional[float] = None
    if heuristic_top_model and successful:
        heuristic_metric = next(
            (r.primary_metric_value for r in successful if r.model == heuristic_top_model),
            None,
        )

    diverged = False
    margin = 0.0
    if (heuristic_top_model and empirical_top
            and heuristic_top_model != empirical_top
            and empirical_top_result is not None
            and heuristic_metric is not None):
        margin = empirical_top_result.primary_metric_value - heuristic_metric
        diverged = margin >= margin_threshold

    notes: List[str] = list(notes_extra or [])
    if any(r.error for r in results):
        failed = [r.model for r in results if r.error]
        notes.append(f"Some candidates failed: {failed}. Their metrics are 0.")
    if diverged:
        notes.append(
            f"Empirical winner ({empirical_top}) beats heuristic top "
            f"({heuristic_top_model}) by {margin:+.3f} {primary} — large enough "
            "to be real signal, not noise. Consider using the empirical winner."
        )
    elif (heuristic_top_model and empirical_top
            and heuristic_top_model != empirical_top
            and heuristic_metric is not None):
        notes.append(
            f"Empirical winner ({empirical_top}) edged out heuristic top "
            f"({heuristic_top_model}) by only {margin:+.3f} {primary} — within "
            f"noise on a {n_supervision_samples}-sample {task} eval. The "
            "heuristic pick is the safer call."
        )

    return EvalReport(
        results=results,
        n_queries_generated=n_supervision_samples,
        query_source=query_source,
        llm_model_used=llm_name,
        heuristic_top=heuristic_top_model,
        empirical_top=empirical_top,
        diverged=diverged,
        notes=notes,
        task=task,
        primary_metric_name=primary,
    )


def _run_retrieval(
    docs, candidates, n_queries, llm, seed, heuristic_top_model, llm_name,
) -> EvalReport:
    query_pairs, query_source = generate_eval_queries(
        docs, n=n_queries, llm=llm, seed=seed
    )
    if not query_pairs:
        return EvalReport(
            notes=["Query generation produced 0 queries. Check LLM availability."],
            query_source=query_source, task="retrieval", primary_metric_name="mrr",
        )
    query_texts = [q for q, _ in query_pairs]
    target_indices = [i for _, i in query_pairs]

    results: List[EvalResult] = []
    for cand in candidates:
        name = cand.get("name")
        if not name:
            continue
        name = _resolve_canonical_name(name)
        results.append(_eval_retrieval_one(
            name=name,
            embed_prefix=cand.get("embed_prefix", "") or "",
            query_prefix=cand.get("query_prefix", "") or "",
            docs=docs, query_texts=query_texts,
            target_indices=target_indices,
            n_queries=len(query_pairs),
        ))

    notes_extra: List[str] = []
    if query_source == "keyword-fallback":
        notes_extra.append(
            "Queries were generated by keyword extraction (no LLM available). "
            "Metrics still rank candidates correctly relative to each other but "
            "are not directly comparable to MTEB-style benchmarks."
        )
    return _finalize_report(
        results, task="retrieval", n_supervision_samples=len(query_pairs),
        query_source=query_source, llm_name=llm_name,
        heuristic_top_model=heuristic_top_model, notes_extra=notes_extra,
    )


def _run_classification(
    docs, candidates, n_queries, llm, seed, heuristic_top_model, llm_name,
) -> EvalReport:
    if llm is None:
        return EvalReport(
            notes=["Classification eval needs an LLM to assign pseudo-labels. "
                   "Provide one or switch task to retrieval (keyword fallback exists)."],
            task="classification", primary_metric_name="f1_macro",
            query_source="none", heuristic_top=heuristic_top_model,
        )
    sample_indices, labels = _llm_assign_labels(docs, llm, seed, max_n=n_queries)
    if len(set(labels)) < 2:
        return EvalReport(
            notes=[f"LLM assigned {len(set(labels))} distinct labels — need ≥2 to "
                   "score classification. Try a stronger LLM or longer corpus."],
            task="classification", primary_metric_name="f1_macro",
            query_source="llm" if labels else "none",
            llm_model_used=llm_name,
            heuristic_top=heuristic_top_model,
        )
    sample_docs = [docs[i] for i in sample_indices]
    results = [
        _eval_classification_one(
            name=_resolve_canonical_name(c["name"]),
            embed_prefix=c.get("embed_prefix", "") or "",
            sample_docs=sample_docs, labels=labels, seed=seed,
        )
        for c in candidates if c.get("name")
    ]
    return _finalize_report(
        results, task="classification", n_supervision_samples=len(sample_docs),
        query_source="llm", llm_name=llm_name,
        heuristic_top_model=heuristic_top_model,
    )


def _run_clustering(
    docs, candidates, n_queries, llm, seed, heuristic_top_model, llm_name,
) -> EvalReport:
    if llm is None:
        return EvalReport(
            notes=["Clustering eval needs an LLM to assign ground-truth topic labels. "
                   "Provide one or switch task to retrieval."],
            task="clustering", primary_metric_name="v_measure",
            query_source="none", heuristic_top=heuristic_top_model,
        )
    sample_indices, labels = _llm_assign_labels(docs, llm, seed, max_n=n_queries)
    if len(set(labels)) < 2:
        return EvalReport(
            notes=[f"LLM produced {len(set(labels))} distinct topics — need ≥2 for "
                   "clustering eval. Try a stronger LLM or longer corpus."],
            task="clustering", primary_metric_name="v_measure",
            query_source="llm" if labels else "none",
            llm_model_used=llm_name,
            heuristic_top=heuristic_top_model,
        )
    sample_docs = [docs[i] for i in sample_indices]
    results = [
        _eval_clustering_one(
            name=_resolve_canonical_name(c["name"]),
            embed_prefix=c.get("embed_prefix", "") or "",
            sample_docs=sample_docs, labels=labels, seed=seed,
        )
        for c in candidates if c.get("name")
    ]
    return _finalize_report(
        results, task="clustering", n_supervision_samples=len(sample_docs),
        query_source="llm", llm_name=llm_name,
        heuristic_top_model=heuristic_top_model,
    )


def _run_deduplication(
    docs, candidates, n_queries, llm, seed, heuristic_top_model, llm_name,
) -> EvalReport:
    if llm is None:
        return EvalReport(
            notes=["Deduplication eval needs an LLM to produce paraphrase "
                   "positives. Provide one or switch task to retrieval."],
            task="deduplication", primary_metric_name="auc",
            query_source="none", heuristic_top=heuristic_top_model,
        )
    sample_indices, paraphrases = _llm_paraphrase_docs(
        docs, llm, seed, max_n=n_queries
    )
    if len(paraphrases) < 4:
        return EvalReport(
            notes=[f"LLM produced only {len(paraphrases)} usable paraphrases — "
                   "need ≥4 to evaluate dedup AUC. Try a stronger LLM."],
            task="deduplication", primary_metric_name="auc",
            query_source="llm" if paraphrases else "none",
            llm_model_used=llm_name,
            heuristic_top=heuristic_top_model,
        )
    sample_docs = [docs[i] for i in sample_indices]
    results = [
        _eval_deduplication_one(
            name=_resolve_canonical_name(c["name"]),
            embed_prefix=c.get("embed_prefix", "") or "",
            sample_docs=sample_docs, paraphrases=paraphrases, seed=seed,
        )
        for c in candidates if c.get("name")
    ]
    return _finalize_report(
        results, task="deduplication", n_supervision_samples=len(sample_docs),
        query_source="llm", llm_name=llm_name,
        heuristic_top_model=heuristic_top_model,
    )


def _run_similarity(
    docs, candidates, n_queries, llm, seed, heuristic_top_model, llm_name,
) -> EvalReport:
    if llm is None:
        return EvalReport(
            notes=["Similarity eval needs an LLM to rate (query, doc) pairs. "
                   "Provide one or switch task to retrieval."],
            task="similarity", primary_metric_name="spearman",
            query_source="none", heuristic_top=heuristic_top_model,
        )
    # 1. Generate queries from a sample of docs (correct pairs)
    query_pairs, _ = generate_eval_queries(
        docs, n=max(4, n_queries // 2), llm=llm, seed=seed
    )
    if len(query_pairs) < 4:
        return EvalReport(
            notes=[f"Only generated {len(query_pairs)} queries — need ≥4 for "
                   "similarity scoring."],
            task="similarity", primary_metric_name="spearman",
            query_source="llm" if query_pairs else "none",
            llm_model_used=llm_name,
            heuristic_top=heuristic_top_model,
        )
    rng = random.Random(seed)
    # 2. Build mixed pairs: half correct (rated higher), half random (rated lower).
    queries: List[str] = []
    paired_docs: List[str] = []
    # Correct pairs
    for q, idx in query_pairs:
        queries.append(q)
        paired_docs.append(docs[idx])
    # Random non-matching pairs (avoid pairing a query with its source doc).
    for q, idx in query_pairs:
        other = idx
        while other == idx and len(docs) > 1:
            other = rng.randrange(len(docs))
        queries.append(q)
        paired_docs.append(docs[other])
    # 3. LLM rates all pairs (the supervision)
    ratings = _llm_rate_similarity_pairs(list(zip(queries, paired_docs)), llm)
    keep_mask = [r is not None for r in ratings]
    if sum(keep_mask) < 4:
        return EvalReport(
            notes=[f"LLM produced only {sum(keep_mask)} valid 1-5 ratings — "
                   "need ≥4. Try a stronger LLM."],
            task="similarity", primary_metric_name="spearman",
            query_source="llm", llm_model_used=llm_name,
            heuristic_top=heuristic_top_model,
        )
    queries = [q for q, keep in zip(queries, keep_mask) if keep]
    paired_docs = [d for d, keep in zip(paired_docs, keep_mask) if keep]
    llm_scores = [float(r) for r, keep in zip(ratings, keep_mask) if keep]

    results = [
        _eval_similarity_one(
            name=_resolve_canonical_name(c["name"]),
            embed_prefix=c.get("embed_prefix", "") or "",
            query_prefix=c.get("query_prefix", "") or "",
            queries=queries, docs_for_pairs=paired_docs,
            llm_scores=llm_scores,
        )
        for c in candidates if c.get("name")
    ]
    return _finalize_report(
        results, task="similarity", n_supervision_samples=len(queries),
        query_source="llm", llm_name=llm_name,
        heuristic_top_model=heuristic_top_model,
    )
