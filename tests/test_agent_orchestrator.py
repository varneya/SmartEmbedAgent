"""
Integration tests for src/agent_orchestrator.py.

These tests run the full pipeline (device profile -> PII removal -> corpus
analysis -> heuristic recommendation) without invoking the LLM, so they
work in CI without an API key. The deterministic fallback shares the same
output schema as the LLM-driven path, so contract tests on the schema
also cover the LLM path.

Separate tests cover the file-based search cache (TTL, eviction, hits).

Run with:
    python -m pytest tests/test_agent_orchestrator.py
or:
    python tests/test_agent_orchestrator.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_orchestrator import (
    FileCache,
    WebSearch,
    get_context,
    reset_context,
    run_pipeline_no_llm,
)


SAMPLE_CORPUS_SHORT = (
    "Customer feedback. Email alice@example.com.\n\n"
    "Server IP: 192.168.1.42.\n\n"
    "We use embeddings for semantic search."
)

SAMPLE_CORPUS_LONG = (
    "Customer feedback. Email alice@example.com.\n\n"
    + ("Lorem ipsum dolor sit amet consectetur adipiscing elit. " * 300)
    + "\n\nA short closing note."
)

REQUIRED_RECOMMENDATION_KEYS = {
    "recommended_models",
    "reasoning_explanation",
    "chunking_strategy",
    "fine_tuning_advice",
    "hardware_fit_analysis",
}


class TestPipelineSchema(unittest.TestCase):
    """The recommendation schema is the public contract — assert it
    rigorously against both short and long corpora."""

    def test_short_corpus_has_required_fields(self):
        out = run_pipeline_no_llm(SAMPLE_CORPUS_SHORT)
        for key in REQUIRED_RECOMMENDATION_KEYS:
            self.assertIn(key, out)

    def test_chunking_strategy_shape(self):
        out = run_pipeline_no_llm(SAMPLE_CORPUS_SHORT)
        cs = out["chunking_strategy"]
        for key in ("needed", "chunk_size_tokens", "overlap_tokens", "rationale"):
            self.assertIn(key, cs)
        self.assertIsInstance(cs["needed"], bool)

    def test_recommended_models_have_rank_and_rationale(self):
        out = run_pipeline_no_llm(SAMPLE_CORPUS_SHORT)
        for entry in out["recommended_models"]:
            self.assertIn("name", entry)
            self.assertIn("rank", entry)
            self.assertIn("rationale", entry)
            self.assertIsInstance(entry["rank"], int)

    def test_models_ranked_in_order(self):
        out = run_pipeline_no_llm(SAMPLE_CORPUS_SHORT)
        ranks = [m["rank"] for m in out["recommended_models"]]
        self.assertEqual(ranks, sorted(ranks))


class TestTaskAwareScoring(unittest.TestCase):
    """Adding `task=` should change the recommendation in known ways:
       - the response carries the task back as a field
       - prefixes get suppressed for symmetric tasks
       - reranker is omitted for non-retrieval tasks"""

    def test_default_task_is_retrieval(self):
        out = run_pipeline_no_llm(SAMPLE_CORPUS_SHORT)
        self.assertEqual(out.get("task"), "retrieval")

    def test_explicit_retrieval_keeps_reranker(self):
        out = run_pipeline_no_llm(SAMPLE_CORPUS_SHORT, task="retrieval")
        self.assertEqual(out["task"], "retrieval")
        self.assertIn("reranker_recommendation", out)
        self.assertIsNotNone(out["reranker_recommendation"].get("name"))

    def test_clustering_skips_reranker(self):
        out = run_pipeline_no_llm(SAMPLE_CORPUS_SHORT, task="clustering")
        self.assertEqual(out["task"], "clustering")
        self.assertIsNone(out["reranker_recommendation"]["name"])
        self.assertIn("clustering", out["reranker_recommendation"]["why"].lower())

    def test_deduplication_skips_reranker_and_suppresses_prefixes(self):
        out = run_pipeline_no_llm(SAMPLE_CORPUS_SHORT, task="deduplication")
        self.assertIsNone(out["reranker_recommendation"]["name"])
        # Symmetric tasks should NOT carry asymmetric prompt prefixes.
        for m in out["recommended_models"]:
            self.assertEqual(m["embed_prefix"], "", f"{m['name']} kept embed prefix on symmetric task")
            self.assertEqual(m["query_prefix"], "", f"{m['name']} kept query prefix on symmetric task")

    def test_similarity_task_propagates(self):
        out = run_pipeline_no_llm(SAMPLE_CORPUS_SHORT, task="similarity")
        self.assertEqual(out["task"], "similarity")
        self.assertIn("similarity", out["reasoning_explanation"].lower())

    def test_unknown_task_falls_back_to_retrieval(self):
        out = run_pipeline_no_llm(SAMPLE_CORPUS_SHORT, task="not_a_real_task")
        self.assertEqual(out["task"], "retrieval")


class TestDualRecommendation(unittest.TestCase):
    """Two parallel ranked lists: 'optimal for hardware' (uses total
    capacity, reproducible) and 'safe for current memory' (uses
    available_ram with 50% headroom, reflects current state)."""

    def test_response_carries_both_lists(self):
        out = run_pipeline_no_llm(SAMPLE_CORPUS_SHORT)
        self.assertIn("recommended_models", out)
        self.assertIn("recommended_models_available", out)
        self.assertIn("available_memory_budget_mb", out)
        self.assertIn("available_memory_basis_gb", out)

    def test_available_list_is_subset_or_same_size(self):
        # The available-memory list is always the same size as or smaller
        # than the optimal list (it's a constrained version).
        out = run_pipeline_no_llm(SAMPLE_CORPUS_SHORT)
        self.assertLessEqual(len(out["recommended_models_available"]),
                             len(out["recommended_models"]))

    def test_available_list_entries_have_full_shape(self):
        out = run_pipeline_no_llm(SAMPLE_CORPUS_SHORT)
        for m in out["recommended_models_available"]:
            for required in ("name", "rank", "rationale", "dimension",
                             "context_window", "size_mb", "multilingual",
                             "embed_prefix", "query_prefix"):
                self.assertIn(required, m)

    def test_budget_reflects_50pct_of_available(self):
        # The budget should be ~half the basis (we use a 50% headroom factor).
        out = run_pipeline_no_llm(SAMPLE_CORPUS_SHORT)
        basis_gb = out["available_memory_basis_gb"]
        budget_mb = out["available_memory_budget_mb"]
        # Budget ≈ basis × 1024 × 0.5; allow 1 MB rounding slack.
        expected = int(basis_gb * 1024 * 0.5)
        self.assertEqual(budget_mb, expected,
                         f"budget {budget_mb} MB doesn't match 50% of basis {basis_gb} GB")

    def test_all_available_entries_fit_under_budget(self):
        # Every model in the available list must satisfy size_mb <= budget.
        out = run_pipeline_no_llm(SAMPLE_CORPUS_SHORT)
        budget = out["available_memory_budget_mb"]
        for m in out["recommended_models_available"]:
            self.assertLessEqual(m["size_mb"], budget,
                                 f"{m['name']} ({m['size_mb']} MB) exceeds budget {budget} MB")


class TestPipelineBehavior(unittest.TestCase):
    """Behavior tests: the pipeline should respond to the input shape in
    sensible ways."""

    def test_short_corpus_does_not_recommend_chunking(self):
        out = run_pipeline_no_llm(SAMPLE_CORPUS_SHORT)
        self.assertFalse(out["chunking_strategy"]["needed"])

    def test_long_corpus_recommends_chunking(self):
        out = run_pipeline_no_llm(SAMPLE_CORPUS_LONG)
        self.assertTrue(out["chunking_strategy"]["needed"])
        self.assertIsNotNone(out["chunking_strategy"]["chunk_size_tokens"])
        self.assertIsNotNone(out["chunking_strategy"]["overlap_tokens"])

    def test_pii_redactions_are_counted(self):
        run_pipeline_no_llm(SAMPLE_CORPUS_SHORT)
        ctx = get_context()
        # alice@example.com + 192.168.1.42 = at least 2 redactions.
        self.assertGreaterEqual(ctx.pii_report["total"], 2)

    def test_user_config_whitelist_is_honored(self):
        # Email gets stripped by default; whitelist preserves it.
        config = {"whitelist": ["alice@example.com"]}
        run_pipeline_no_llm(SAMPLE_CORPUS_SHORT, user_config=config)
        ctx = get_context()
        self.assertIn("alice@example.com", ctx.cleaned_corpus)


class TestContextSharing(unittest.TestCase):
    """Tools share state through the module-level context. Verify each
    tool's output lands in the right context field."""

    def test_context_is_populated_after_run(self):
        run_pipeline_no_llm(SAMPLE_CORPUS_SHORT)
        ctx = get_context()
        self.assertIsNotNone(ctx.device_specs)
        self.assertIsNotNone(ctx.cleaned_corpus)
        self.assertIsNotNone(ctx.pii_report)
        self.assertIsNotNone(ctx.corpus_analysis)

    def test_reset_context_clears_state(self):
        run_pipeline_no_llm(SAMPLE_CORPUS_SHORT)
        reset_context("new corpus")
        ctx = get_context()
        self.assertEqual(ctx.raw_corpus, "new corpus")
        self.assertIsNone(ctx.device_specs)
        self.assertIsNone(ctx.cleaned_corpus)


class TestFileCache(unittest.TestCase):
    """The cache must:
       - return None on misses
       - return values on fresh hits
       - expire entries past their TTL
       - persist across instances pointed at the same file
    """

    def test_miss_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            cache = FileCache(Path(d) / "cache.json")
            self.assertIsNone(cache.get("nonexistent query"))

    def test_hit_returns_stored_value(self):
        with tempfile.TemporaryDirectory() as d:
            cache = FileCache(Path(d) / "cache.json")
            cache.set("q1", "stored value")
            self.assertEqual(cache.get("q1"), "stored value")

    def test_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.json"
            cache_a = FileCache(path)
            cache_a.set("shared", "persistent")
            cache_b = FileCache(path)
            self.assertEqual(cache_b.get("shared"), "persistent")

    def test_expiration(self):
        with tempfile.TemporaryDirectory() as d:
            cache = FileCache(Path(d) / "cache.json", ttl_seconds=1)
            cache.set("ephemeral", "v")
            self.assertEqual(cache.get("ephemeral"), "v")
            time.sleep(1.1)
            self.assertIsNone(cache.get("ephemeral"))

    def test_keys_are_normalized(self):
        # Whitespace and case shouldn't produce separate cache entries.
        with tempfile.TemporaryDirectory() as d:
            cache = FileCache(Path(d) / "cache.json")
            cache.set("Best Embedding Models", "answer")
            self.assertEqual(cache.get("best embedding models"), "answer")
            self.assertEqual(cache.get("  Best Embedding Models  "), "answer")


class TestWebSearch(unittest.TestCase):
    def test_uses_cache_on_repeat_query(self):
        calls = []

        def fake_backend(q: str) -> str:
            calls.append(q)
            return f"results for {q}"

        with tempfile.TemporaryDirectory() as d:
            cache = FileCache(Path(d) / "cache.json")
            ws = WebSearch(cache, backend=fake_backend)

            r1 = ws.query("embedding benchmarks 2026")
            r2 = ws.query("embedding benchmarks 2026")

            self.assertEqual(len(calls), 1)  # backend only called once
            self.assertFalse(r1["cached"])
            self.assertTrue(r2["cached"])
            self.assertEqual(r1["result"], r2["result"])

    def test_backend_failure_does_not_crash(self):
        def boom(_: str) -> str:
            raise RuntimeError("network down")

        with tempfile.TemporaryDirectory() as d:
            ws = WebSearch(FileCache(Path(d) / "cache.json"), backend=boom)
            result = ws.query("anything")
            self.assertIn("Search backend error", result["result"])


class TestRecommendationOutput(unittest.TestCase):
    """The recommendation output should be JSON-serializable end-to-end."""

    def test_output_is_json_serializable(self):
        out = run_pipeline_no_llm(SAMPLE_CORPUS_SHORT)
        json.dumps(out)  # must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
