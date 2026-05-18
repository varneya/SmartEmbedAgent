"""
Unit tests for src/evaluator.py.

Embedding is mocked (the real sentence-transformers models are gigabytes
and slow to load — not appropriate for CI). The mock returns deterministic
fake vectors that exercise the rank-finding + metric-computation logic.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Skip the whole module if numpy isn't installed (it's a transitive dep so
# usually is, but stay defensive).
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestQueryGeneration(unittest.TestCase):
    def test_keyword_fallback_picks_distinctive_tokens(self):
        from src.evaluator import _keyword_query_from_doc
        doc = (
            "The honeycrisp apple cultivar was developed by the University "
            "of Minnesota horticulture program for commercial orchards."
        )
        q = _keyword_query_from_doc(doc, n_keywords=4)
        # Should NOT include stopwords; should include distinctive multi-syllable terms.
        for sw in ("the", "for", "by"):
            self.assertNotIn(f" {sw} ", f" {q} ")
        self.assertTrue(any(w in q.lower() for w in ("honeycrisp", "horticulture", "minnesota", "commercial")),
                        f"keyword query missed distinctive terms: {q!r}")

    def test_generate_eval_queries_without_llm_uses_keyword_fallback(self):
        from src.evaluator import generate_eval_queries
        docs = [
            "Customer feedback about durability of plastic packaging materials.",
            "Engine maintenance instructions for two-wheeler commuter motorcycles.",
            "Apple cultivar evaluation criteria for cold-climate orchards.",
        ]
        pairs, source = generate_eval_queries(docs, n=3, llm=None, seed=42)
        self.assertEqual(source, "keyword-fallback")
        self.assertEqual(len(pairs), 3)
        for q, idx in pairs:
            self.assertIsInstance(q, str)
            self.assertGreater(len(q), 0)
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(docs))

    def test_generate_eval_queries_deterministic_under_seed(self):
        from src.evaluator import generate_eval_queries
        docs = [f"Document number {i}." for i in range(20)]
        pairs_a, _ = generate_eval_queries(docs, n=5, llm=None, seed=123)
        pairs_b, _ = generate_eval_queries(docs, n=5, llm=None, seed=123)
        self.assertEqual([i for _, i in pairs_a], [i for _, i in pairs_b])

    def test_generate_eval_queries_with_mocked_llm(self):
        from src.evaluator import generate_eval_queries
        # Mock LLM: returns a constant query for every passage.
        mock_resp = mock.Mock(content="how does the engine work")
        mock_llm = mock.Mock()
        mock_llm.invoke.return_value = mock_resp
        docs = [f"Document number {i}." for i in range(10)]
        pairs, source = generate_eval_queries(docs, n=5, llm=mock_llm, seed=0)
        self.assertEqual(source, "llm")
        self.assertEqual(len(pairs), 5)
        for q, _ in pairs:
            self.assertEqual(q, "how does the engine work")

    def test_llm_failure_falls_back_to_keyword_per_doc(self):
        from src.evaluator import generate_eval_queries
        # Mock LLM that raises on every invoke.
        mock_llm = mock.Mock()
        mock_llm.invoke.side_effect = RuntimeError("simulated LLM crash")
        docs = ["Some sufficiently long document about durability testing.",
                "Another doc concerning maintenance schedules and reliability."]
        pairs, source = generate_eval_queries(docs, n=2, llm=mock_llm, seed=0)
        # source is still 'llm' (we tried) but each query came from the fallback.
        self.assertEqual(source, "llm")
        self.assertEqual(len(pairs), 2)
        for q, _ in pairs:
            self.assertGreater(len(q), 0)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestMetrics(unittest.TestCase):
    """Synthetic geometry — perfect ranker scores 1.0 MRR, no-information
    ranker scores ~ ln(2)/n."""

    def test_perfect_retrieval_gives_perfect_metrics(self):
        from src.evaluator import _compute_metrics
        # 5 queries, 10 docs. Make query[i] == doc[i] exactly so the
        # target is always rank-1.
        N = 10
        doc_embs = np.eye(N, dtype=np.float32)
        # Pick the first 5 docs as targets.
        targets = list(range(5))
        query_embs = doc_embs[targets]
        m = _compute_metrics(query_embs, doc_embs, targets)
        self.assertAlmostEqual(m["mrr"], 1.0, places=4)
        self.assertAlmostEqual(m["recall_at_5"], 1.0, places=4)
        self.assertAlmostEqual(m["recall_at_10"], 1.0, places=4)
        # nDCG@10 with rank=1 every time = sum(1/log2(2))/5 = 1.0.
        self.assertAlmostEqual(m["ndcg_at_10"], 1.0, places=4)

    def test_random_retrieval_gives_low_mrr(self):
        from src.evaluator import _compute_metrics
        rng = np.random.default_rng(0)
        N = 100
        doc_embs = rng.standard_normal((N, 32)).astype(np.float32)
        # Normalize so cosine = dot.
        doc_embs /= np.linalg.norm(doc_embs, axis=1, keepdims=True)
        # Queries are independent random vectors → no signal.
        targets = list(range(20))
        q_embs = rng.standard_normal((20, 32)).astype(np.float32)
        q_embs /= np.linalg.norm(q_embs, axis=1, keepdims=True)
        m = _compute_metrics(q_embs, doc_embs, targets)
        # Expected MRR under random ordering of 100 docs is ~0.05; allow slack.
        self.assertLess(m["mrr"], 0.30)

    def test_recall_at_k_strictly_monotonic(self):
        from src.evaluator import _compute_metrics
        rng = np.random.default_rng(7)
        N = 50
        doc_embs = rng.standard_normal((N, 16)).astype(np.float32)
        doc_embs /= np.linalg.norm(doc_embs, axis=1, keepdims=True)
        targets = list(range(15))
        q_embs = rng.standard_normal((15, 16)).astype(np.float32)
        q_embs /= np.linalg.norm(q_embs, axis=1, keepdims=True)
        m = _compute_metrics(q_embs, doc_embs, targets)
        self.assertLessEqual(m["recall_at_5"], m["recall_at_10"])


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestEvaluateCandidatesEndToEnd(unittest.TestCase):
    """Full evaluate_candidates call with the embedder + LLM mocked."""

    @mock.patch("src.evaluator._embed_texts")
    def test_evaluate_two_candidates_with_mocked_embedder(self, mock_embed):
        """Mocked embedder returns identity-shaped vectors so the eval
        produces predictable metrics."""
        from src.evaluator import evaluate_candidates

        corpus = "\n\n".join([f"Document {i} about widget {i % 5}." for i in range(20)])

        # Mock embedder: for the doc set return identity rows; for the query
        # set return rows matching the target index. This guarantees the
        # 'correct' doc is rank-1 for every query → perfect MRR.
        def fake_embed(model_name, texts, prefix="", batch_size=64):
            n = len(texts)
            arr = np.zeros((n, 20), dtype=np.float32)
            for i in range(n):
                arr[i, i % 20] = 1.0
            return arr
        mock_embed.side_effect = fake_embed

        candidates = [
            {"name": "fake-model-a", "embed_prefix": "", "query_prefix": ""},
            {"name": "fake-model-b", "embed_prefix": "passage: ", "query_prefix": "query: "},
        ]
        report = evaluate_candidates(
            corpus=corpus,
            candidates=candidates,
            n_queries=5,
            llm=None,
            heuristic_top_model="fake-model-a",
        )
        self.assertEqual(len(report.results), 2)
        for r in report.results:
            self.assertIsNone(r.error)
            self.assertGreaterEqual(r.mrr, 0.0)
            self.assertLessEqual(r.mrr, 1.0)
        self.assertEqual(report.query_source, "keyword-fallback")
        self.assertIsNotNone(report.empirical_top)

    @mock.patch("src.evaluator._embed_texts")
    def test_evaluate_records_failures_per_candidate(self, mock_embed):
        from src.evaluator import evaluate_candidates
        # Make the embedder raise — every candidate should fail cleanly.
        mock_embed.side_effect = RuntimeError("model download failed")
        report = evaluate_candidates(
            corpus="doc one\n\ndoc two\n\ndoc three",
            candidates=[{"name": "fake-model"}],
            n_queries=2,
            llm=None,
        )
        self.assertEqual(len(report.results), 1)
        self.assertIsNotNone(report.results[0].error)
        self.assertIn("model download failed", report.results[0].error)
        self.assertEqual(report.empirical_top, None)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestTaskAwareDispatch(unittest.TestCase):
    """PR 3: evaluate_candidates dispatches on `task` and populates
    primary_metric_name / primary_metric_value uniformly across tasks."""

    def _make_well_separated_embedder(self, n_clusters=3):
        """Returns a fake _embed_texts that puts each text into one of
        n_clusters perfectly-separated 1-hot positions. Cycles through
        clusters by text-index so the mapping is deterministic."""
        def fake(name, texts, prefix="", batch_size=None):
            arr = np.zeros((len(texts), n_clusters), dtype=np.float32)
            for i in range(len(texts)):
                arr[i, i % n_clusters] = 1.0
            return arr
        return fake

    @mock.patch("src.evaluator._embed_texts")
    def test_retrieval_populates_primary_metric_as_mrr(self, mock_embed):
        from src.evaluator import evaluate_candidates
        mock_embed.side_effect = self._make_well_separated_embedder(20)
        corpus = "\n\n".join([f"Doc {i} about widget {i % 4}." for i in range(20)])
        report = evaluate_candidates(
            corpus=corpus,
            candidates=[{"name": "fake-a"}, {"name": "fake-b"}],
            n_queries=5, llm=None, heuristic_top_model="fake-a",
            task="retrieval",
        )
        self.assertEqual(report.task, "retrieval")
        self.assertEqual(report.primary_metric_name, "mrr")
        for r in report.results:
            self.assertEqual(r.task, "retrieval")
            self.assertEqual(r.primary_metric_name, "mrr")
            self.assertEqual(r.primary_metric_value, r.mrr)

    @mock.patch("src.evaluator._embed_texts")
    def test_classification_uses_macro_f1(self, mock_embed):
        """LLM labels 3 distinct classes; mocked embeddings cluster
        perfectly by label → expect F1 = 1.0."""
        from src.evaluator import evaluate_candidates
        mock_embed.side_effect = self._make_well_separated_embedder(3)
        corpus = "\n\n".join([f"Doc {i} category {i % 3}." for i in range(15)])

        # LLM returns JSON {index: label} matching the embedding cycle (i % 3).
        mock_llm = mock.Mock()
        mock_resp = mock.Mock()
        mock_resp.content = "{" + ", ".join(
            f'"{i}": "cat{i % 3}"' for i in range(15)
        ) + "}"
        mock_llm.invoke.return_value = mock_resp

        report = evaluate_candidates(
            corpus=corpus, candidates=[{"name": "fake-clf"}],
            n_queries=15, llm=mock_llm, heuristic_top_model="fake-clf",
            task="classification",
        )
        self.assertEqual(report.task, "classification")
        self.assertEqual(report.primary_metric_name, "f1_macro")
        self.assertEqual(len(report.results), 1)
        r = report.results[0]
        self.assertIsNone(r.error)
        self.assertGreater(r.primary_metric_value, 0.9)
        self.assertEqual(r.task, "classification")
        # Legacy retrieval fields are zero for non-retrieval tasks.
        self.assertEqual(r.mrr, 0.0)

    @mock.patch("src.evaluator._embed_texts")
    def test_clustering_uses_v_measure(self, mock_embed):
        from src.evaluator import evaluate_candidates
        mock_embed.side_effect = self._make_well_separated_embedder(3)
        corpus = "\n\n".join([f"Doc {i} topic {i % 3}." for i in range(15)])

        mock_llm = mock.Mock()
        mock_resp = mock.Mock()
        mock_resp.content = "{" + ", ".join(
            f'"{i}": "topic{i % 3}"' for i in range(15)
        ) + "}"
        mock_llm.invoke.return_value = mock_resp

        report = evaluate_candidates(
            corpus=corpus, candidates=[{"name": "fake-clu"}],
            n_queries=15, llm=mock_llm, heuristic_top_model="fake-clu",
            task="clustering",
        )
        self.assertEqual(report.primary_metric_name, "v_measure")
        r = report.results[0]
        self.assertIsNone(r.error)
        self.assertGreater(r.primary_metric_value, 0.9)

    @mock.patch("src.evaluator._embed_texts")
    def test_deduplication_uses_auc(self, mock_embed):
        """Anchors and their paraphrases get identical embeddings → AUC=1."""
        from src.evaluator import evaluate_candidates
        mock_embed.side_effect = self._make_well_separated_embedder(10)
        corpus = "\n\n".join([f"Doc {i} unique content." for i in range(10)])

        mock_llm = mock.Mock()
        # Every invoke (paraphrase requests) returns a non-empty rewrite.
        mock_resp = mock.Mock()
        mock_resp.content = "Reworded version of the input passage."
        mock_llm.invoke.return_value = mock_resp

        report = evaluate_candidates(
            corpus=corpus, candidates=[{"name": "fake-dedup"}],
            n_queries=8, llm=mock_llm, heuristic_top_model="fake-dedup",
            task="deduplication",
        )
        self.assertEqual(report.primary_metric_name, "auc")
        r = report.results[0]
        self.assertIsNone(r.error)
        # Anchor i and paraphrase i share embedding row → cos=1; random
        # non-pair → cos=0. Perfect separation → AUC=1.
        self.assertGreater(r.primary_metric_value, 0.95)

    def test_classification_without_llm_is_graceful(self):
        """Non-retrieval tasks need an LLM. With no LLM, return a
        clear note rather than crashing."""
        from src.evaluator import evaluate_candidates
        report = evaluate_candidates(
            corpus="Doc a\n\nDoc b\n\nDoc c",
            candidates=[{"name": "anything"}],
            n_queries=5, llm=None,
            task="classification",
        )
        self.assertEqual(report.task, "classification")
        self.assertEqual(report.results, [])
        self.assertTrue(any("LLM" in n for n in report.notes))

    def test_unknown_task_falls_back_to_retrieval(self):
        from src.evaluator import evaluate_candidates
        report = evaluate_candidates(
            corpus="Doc a\n\nDoc b",
            candidates=[],  # short-circuit before embedding
            n_queries=5, llm=None,
            task="something-unknown",
        )
        self.assertEqual(report.task, "retrieval")


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestPerModelBatchSize(unittest.TestCase):
    """Fix 3: BGE-M3's catalog entry sets eval_batch_size=8 to avoid
    the MPS 2**32-element tile crash on long-context inputs."""

    def test_bge_m3_uses_smaller_batch_size(self):
        from src.evaluator import _lookup_eval_batch_size
        self.assertEqual(_lookup_eval_batch_size("BAAI/bge-m3"), 8)
        # Other catalog models keep the default.
        self.assertEqual(_lookup_eval_batch_size("sentence-transformers/all-MiniLM-L6-v2"), 64)
        # Unknown models get the default.
        self.assertEqual(_lookup_eval_batch_size("some/unknown-model"), 64)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestDivergenceMargin(unittest.TestCase):
    """Fix 2: tiny MRR gaps (< 0.05) should NOT flag divergence — that's
    noise on a 30-query eval and trains users to ignore the signal."""

    @mock.patch("src.evaluator._embed_texts")
    def test_tiny_margin_does_not_flag_divergence(self, mock_embed):
        from src.evaluator import evaluate_candidates
        # Two models that produce slightly different but very-similar embeddings.
        def fake(name, texts, prefix="", batch_size=None):
            arr = np.zeros((len(texts), 20), dtype=np.float32)
            for i in range(len(texts)):
                arr[i, i % 20] = 1.0
                if name == "model-b":
                    # Small perturbation that flips rank for a tiny number of queries.
                    arr[i, (i + 1) % 20] = 0.01
            return arr
        mock_embed.side_effect = fake

        corpus = "\n\n".join([f"Doc {i} widget." for i in range(20)])
        report = evaluate_candidates(
            corpus=corpus,
            candidates=[{"name": "model-a"}, {"name": "model-b"}],
            n_queries=10, llm=None,
            heuristic_top_model="model-a",
            task="retrieval",
        )
        # If both models score equal/near-equal MRR, diverged must be False
        # (margin doesn't clear the 0.05 threshold).
        if report.empirical_top and report.empirical_top != "model-a":
            mrrs = {r.model: r.mrr for r in report.results}
            margin = mrrs[report.empirical_top] - mrrs["model-a"]
            if margin < 0.05:
                self.assertFalse(report.diverged,
                    f"diverged should be False when margin={margin:.3f} < 0.05")


if __name__ == "__main__":
    unittest.main(verbosity=2)
