"""
Evaluation Runner
=================

Runs the agent against a benchmark corpus and measures whether the
recommended model actually performs well on a downstream retrieval task.

The harness is intentionally simple: it loads a benchmark corpus that
includes labeled query/document pairs, runs the agent to get a model
recommendation, and computes Recall@k and MRR using whatever embedding
backend is available. When `sentence-transformers` is installed, the real
recommended model is used; otherwise the harness falls back to a hash-based
embedding so the framework still produces structural results in CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_orchestrator import run_pipeline_no_llm  # noqa: E402


@dataclass
class EvalResult:
    benchmark_name: str
    recommended_model: str
    recall_at_1: float
    recall_at_5: float
    mrr: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "recommended_model": self.recommended_model,
            "recall_at_1": round(self.recall_at_1, 4),
            "recall_at_5": round(self.recall_at_5, 4),
            "mrr": round(self.mrr, 4),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Embedding backend — real if sentence-transformers is available, else
# a deterministic hash-based fallback that lets the framework run anywhere.
# ---------------------------------------------------------------------------
def _build_embedder(model_name: str) -> Tuple[Callable[[str], List[float]], str]:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        return lambda text: model.encode(text).tolist(), f"sentence-transformers:{model_name}"
    except Exception as e:
        print(f"[eval] sentence-transformers unavailable ({e}); using hash-based fallback.")
        # 64-d deterministic hash embedding. Won't beat real models on real
        # benchmarks but keeps the eval harness exercisable.
        def hash_embed(text: str) -> List[float]:
            digest = hashlib.sha256(text.lower().encode("utf-8")).digest()
            return [b / 255.0 for b in digest[:64]]
        return hash_embed, "hash-fallback"


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (norm_a * norm_b)


def _retrieve(query_emb: List[float], doc_embs: List[List[float]], k: int) -> List[int]:
    scored = sorted(
        range(len(doc_embs)),
        key=lambda i: _cosine(query_emb, doc_embs[i]),
        reverse=True,
    )
    return scored[:k]


# ---------------------------------------------------------------------------
# Benchmark IO
# ---------------------------------------------------------------------------
def _load_benchmark(path: Path) -> Dict[str, Any]:
    """Benchmark format: a JSON file with `documents` (list of strings) and
    `queries` (list of {"query": str, "relevant_doc_indices": [int]})."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "documents" not in data or "queries" not in data:
        raise ValueError(f"Benchmark file missing required keys: {path}")
    return data


def _corpus_text(documents: List[str]) -> str:
    """Reassemble documents into the corpus shape the agent expects."""
    return "\n\n".join(documents)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_evaluation(benchmark_path: Path, k: int = 5) -> EvalResult:
    benchmark = _load_benchmark(benchmark_path)
    documents: List[str] = benchmark["documents"]
    queries: List[Dict[str, Any]] = benchmark["queries"]

    # Step 1: ask the agent which model to use.
    rec = run_pipeline_no_llm(_corpus_text(documents))
    model_name = rec["recommended_models"][0]["name"]

    # Step 2: embed and score.
    embedder, backend_name = _build_embedder(model_name)
    doc_embs = [embedder(d) for d in documents]

    hits_at_1 = 0
    hits_at_k = 0
    rr_sum = 0.0
    for q in queries:
        query_emb = embedder(q["query"])
        retrieved = _retrieve(query_emb, doc_embs, k)
        relevant = set(q["relevant_doc_indices"])
        if not relevant:
            continue
        if retrieved[0] in relevant:
            hits_at_1 += 1
        if any(r in relevant for r in retrieved):
            hits_at_k += 1
        for rank, doc_idx in enumerate(retrieved, start=1):
            if doc_idx in relevant:
                rr_sum += 1.0 / rank
                break

    n = len(queries) or 1
    return EvalResult(
        benchmark_name=benchmark.get("name", benchmark_path.stem),
        recommended_model=model_name,
        recall_at_1=hits_at_1 / n,
        recall_at_5=hits_at_k / n,
        mrr=rr_sum / n,
        notes=[f"backend: {backend_name}"],
    )


def _cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", help="Path to benchmark JSON.")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    result = run_evaluation(Path(args.benchmark), k=args.k)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
