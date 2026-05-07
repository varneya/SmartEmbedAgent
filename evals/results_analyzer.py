"""
Results Analyzer
================

Aggregates eval JSON files into a Markdown report. Run after running
`evaluation_runner.py` or `comparison_baseline.py` against multiple
benchmarks to produce a portfolio-style summary.

Usage:
    python -m evals.results_analyzer runs/eval_*.json --output report.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List


def load_results(paths: List[Path]) -> List[Dict[str, Any]]:
    results = []
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            data["_source_file"] = p.name
            results.append(data)
        except Exception as e:
            print(f"[results_analyzer] Skipping {p}: {e}", file=sys.stderr)
    return results


def render_report(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "# Evaluation Report\n\nNo results provided.\n"

    lines: List[str] = ["# SmartEmbedAgent Evaluation Report", ""]

    # If results are comparison-shaped (have agent_recommendation/baseline),
    # render the comparison table; else render the single-model table.
    is_comparison = any("agent_recommendation" in r for r in results)

    if is_comparison:
        lines += ["## Agent vs. Baseline", ""]
        lines += ["| Benchmark | Agent model | Recall@1 (agent) | Recall@1 (baseline) | Δ Recall@1 | MRR (agent) | MRR (baseline) | Δ MRR |"]
        lines += ["|---|---|---|---|---|---|---|---|"]
        recall_deltas = []
        mrr_deltas = []
        for r in results:
            ag = r.get("agent_recommendation", {})
            bl = r.get("baseline", {})
            d = r.get("delta_vs_baseline", {})
            lines.append(
                f"| {r.get('benchmark', '?')} | `{ag.get('recommended_model', '?')}` | "
                f"{ag.get('recall_at_1', 0):.3f} | {bl.get('recall_at_1', 0):.3f} | "
                f"{d.get('recall_at_1', 0):+.3f} | "
                f"{ag.get('mrr', 0):.3f} | {bl.get('mrr', 0):.3f} | "
                f"{d.get('mrr', 0):+.3f} |"
            )
            recall_deltas.append(d.get("recall_at_1", 0))
            mrr_deltas.append(d.get("mrr", 0))

        lines += [""]
        if recall_deltas:
            lines += [
                "### Aggregate",
                "",
                f"- Mean Δ Recall@1 vs. baseline: **{statistics.mean(recall_deltas):+.3f}**",
                f"- Mean Δ MRR vs. baseline: **{statistics.mean(mrr_deltas):+.3f}**",
                "",
            ]
    else:
        lines += ["## Per-benchmark results", ""]
        lines += ["| Benchmark | Model | Recall@1 | Recall@5 | MRR |"]
        lines += ["|---|---|---|---|---|"]
        for r in results:
            lines.append(
                f"| {r.get('benchmark_name', '?')} | `{r.get('recommended_model', '?')}` | "
                f"{r.get('recall_at_1', 0):.3f} | {r.get('recall_at_5', 0):.3f} | "
                f"{r.get('mrr', 0):.3f} |"
            )
        lines.append("")

    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="Eval result JSON files.")
    parser.add_argument("--output", default="evals_report.md", help="Output markdown path.")
    args = parser.parse_args()

    paths = [Path(p) for p in args.results]
    results = load_results(paths)
    report = render_report(results)

    Path(args.output).write_text(report, encoding="utf-8")
    print(f"Wrote {args.output} ({len(results)} result file(s) summarized).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
