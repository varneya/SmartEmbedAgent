"""
Tests for the evaluation harness. Use the hash-based fallback embedder so
these run in CI without sentence-transformers installed.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evals.comparison_baseline import compare  # noqa: E402
from evals.evaluation_runner import run_evaluation  # noqa: E402
from evals.results_analyzer import render_report  # noqa: E402


BENCHMARK = PROJECT_ROOT / "evals" / "benchmark_corpora" / "general_qa.json"


class TestEvaluationRunner(unittest.TestCase):
    def test_runs_to_completion(self):
        result = run_evaluation(BENCHMARK, k=5).to_dict()
        for key in ("benchmark_name", "recommended_model", "recall_at_1", "recall_at_5", "mrr"):
            self.assertIn(key, result)

    def test_metrics_in_range(self):
        result = run_evaluation(BENCHMARK, k=5).to_dict()
        for key in ("recall_at_1", "recall_at_5", "mrr"):
            self.assertGreaterEqual(result[key], 0.0)
            self.assertLessEqual(result[key], 1.0)


class TestComparison(unittest.TestCase):
    def test_compare_produces_delta(self):
        result = compare(BENCHMARK, k=5)
        self.assertIn("agent_recommendation", result)
        self.assertIn("baseline", result)
        self.assertIn("delta_vs_baseline", result)
        self.assertIn("recall_at_1", result["delta_vs_baseline"])


class TestResultsAnalyzer(unittest.TestCase):
    def test_render_report_with_comparison_data(self):
        data = [{
            "benchmark": "test",
            "agent_recommendation": {"recommended_model": "X", "recall_at_1": 0.8, "mrr": 0.85},
            "baseline": {"recall_at_1": 0.7, "mrr": 0.75},
            "delta_vs_baseline": {"recall_at_1": 0.1, "recall_at_5": 0.05, "mrr": 0.1},
        }]
        report = render_report(data)
        self.assertIn("Agent vs. Baseline", report)
        self.assertIn("test", report)

    def test_render_report_with_single_run_data(self):
        data = [{
            "benchmark_name": "test",
            "recommended_model": "X",
            "recall_at_1": 0.8,
            "recall_at_5": 0.9,
            "mrr": 0.85,
        }]
        report = render_report(data)
        self.assertIn("Per-benchmark results", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
