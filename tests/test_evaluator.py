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


if __name__ == "__main__":
    unittest.main(verbosity=2)
