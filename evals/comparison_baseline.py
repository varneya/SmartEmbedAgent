"""
Comparison Baseline
===================

Runs both the agent's recommended model and a fixed default model
(MiniLM-L6-v2) against the same benchmark, so the user can quantify
whether the agent's choice actually beats the naive default.

Usage:
    python -m evals.comparison_baseline <benchmark_path>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.evaluation_runner import (  # noqa: E402
    _build_embedder,
    _corpus_text,
    _cosine,
    _load_benchmark,
    _retrieve,
    run_evaluation,
)


DEFAULT_BASELINE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _evaluate_with_model(benchmark, model_name: str, k: int) -> Dict[str, Any]:
    embedder, backend = _build_embedder(model_name)
    docs = benchmark["documents"]
    queries = benchmark["queries"]
    doc_embs = [embedder(d) for d in docs]

    hits_at_1 = 0
    hits_at_k = 0
    rr_sum = 0.0
    for q in queries:
        qe = embedder(q["query"])
        retrieved = _retrieve(qe, doc_embs, k)
        relevant = set(q["relevant_doc_indices"])
        if not relevant:
            continue
        if retrieved[0] in relevant:
            hits_at_1 += 1
        if any(r in relevant for r in retrieved):
            hits_at_k += 1
        for rank, idx in enumerate(retrieved, 1):
            if idx in relevant:
                rr_sum += 1.0 / rank
                break

    n = len(queries) or 1
    return {
        "model": model_name,
        "backend": backend,
        "recall_at_1": round(hits_at_1 / n, 4),
        "recall_at_5": round(hits_at_k / n, 4),
        "mrr": round(rr_sum / n, 4),
    }


def compare(benchmark_path: Path, k: int = 5) -> Dict[str, Any]:
    benchmark = _load_benchmark(benchmark_path)

    # Agent's pick.
    agent_result = run_evaluation(benchmark_path, k=k).to_dict()

    # Fixed baseline.
    baseline_result = _evaluate_with_model(benchmark, DEFAULT_BASELINE_MODEL, k=k)

    # Delta.
    delta = {
        "recall_at_1": agent_result["recall_at_1"] - baseline_result["recall_at_1"],
        "recall_at_5": agent_result["recall_at_5"] - baseline_result["recall_at_5"],
        "mrr": agent_result["mrr"] - baseline_result["mrr"],
    }

    return {
        "benchmark": benchmark.get("name", benchmark_path.stem),
        "agent_recommendation": agent_result,
        "baseline": baseline_result,
        "delta_vs_baseline": {k: round(v, 4) for k, v in delta.items()},
    }


def _cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", help="Path to benchmark JSON.")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    result = compare(Path(args.benchmark), k=args.k)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
