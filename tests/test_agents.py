"""
Unit tests for the multi-agent pipeline (src/agents/*).

We mock the expensive bits — sentence-transformers embedding and the
LLM — so the suite stays fast and runs in CI.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


SAMPLE_CORPUS = "\n\n".join([
    f"Customer feedback {i}: the product is excellent for our use case. "
    "Email us at contact@example.com for support questions."
    for i in range(10)
])


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestSuggesterAgent(unittest.TestCase):
    def test_produces_recommendation_with_full_schema(self):
        from src.agents import AgentContext
        from src.agents import suggester
        ctx = AgentContext(corpus=SAMPLE_CORPUS, task="retrieval")
        out = suggester.run(ctx)
        self.assertIsNotNone(out.recommendation)
        for key in ("recommended_models", "chunking_strategy", "task",
                    "index_estimate", "reranker_recommendation"):
            self.assertIn(key, out.recommendation)
        self.assertEqual(out.recommendation["task"], "retrieval")
        # Per-agent note should mention the heuristic top.
        self.assertTrue(any("Heuristic top" in n for n in out.per_agent_notes))


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestQueryGeneratorAgent(unittest.TestCase):
    def test_uses_keyword_fallback_when_no_llm(self):
        from src.agents import AgentContext
        from src.agents import query_generator
        ctx = AgentContext(corpus=SAMPLE_CORPUS, n_queries=4, llm=None, use_llm=False)
        out = query_generator.run(ctx)
        self.assertEqual(out.query_source, "keyword-fallback")
        self.assertEqual(len(out.queries), 4)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestEvaluatorAgent(unittest.TestCase):
    @mock.patch("src.evaluator._embed_texts")
    def test_evaluator_populates_eval_report(self, mock_embed):
        from src.agents import AgentContext
        from src.agents import suggester, evaluator_agent

        def fake_embed(name, texts, prefix="", batch_size=64):
            arr = np.zeros((len(texts), 20), dtype=np.float32)
            for i in range(len(texts)):
                arr[i, i % 20] = 1.0
            return arr
        mock_embed.side_effect = fake_embed

        ctx = AgentContext(corpus=SAMPLE_CORPUS, task="retrieval", use_llm=False, n_queries=5)
        ctx = suggester.run(ctx)
        ctx = evaluator_agent.run(ctx)
        self.assertIsNotNone(ctx.eval_report)
        self.assertGreater(len(ctx.eval_report["results"]), 0)
        for r in ctx.eval_report["results"]:
            self.assertIn("mrr", r)
            self.assertIn("model", r)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestDeciderAgent(unittest.TestCase):

    def _stub_recommendation(self):
        return {
            "recommended_models": [
                {"name": "model-A", "rank": 1, "dimension": 384, "size_mb": 80},
                {"name": "model-B", "rank": 2, "dimension": 768, "size_mb": 420},
            ],
            "recommended_models_available": [
                {"name": "model-A", "rank": 1, "dimension": 384, "size_mb": 80},
            ],
        }

    def _stub_eval(self, mrr_a, mrr_b, err_a=None, err_b=None):
        return {
            "results": [
                {"model": "model-A", "mrr": mrr_a, "ndcg_at_10": mrr_a, "recall_at_5": mrr_a,
                 "recall_at_10": mrr_a, "elapsed_seconds": 1.0, "n_docs_embedded": 10,
                 "n_queries": 5, "error": err_a},
                {"model": "model-B", "mrr": mrr_b, "ndcg_at_10": mrr_b, "recall_at_5": mrr_b,
                 "recall_at_10": mrr_b, "elapsed_seconds": 1.5, "n_docs_embedded": 10,
                 "n_queries": 5, "error": err_b},
            ],
            "n_queries_generated": 5, "query_source": "keyword-fallback",
            "heuristic_top": "model-A",
            "empirical_top": "model-A" if mrr_a >= mrr_b else "model-B",
            "diverged": (mrr_b > mrr_a) and not err_b,
            "notes": [],
        }

    def test_decider_confirms_heuristic_when_margin_too_small(self):
        from src.agents import AgentContext
        from src.agents import decider
        ctx = AgentContext(corpus="x", use_llm=False)
        ctx.recommendation = self._stub_recommendation()
        ctx.eval_report = self._stub_eval(mrr_a=0.70, mrr_b=0.72)  # margin 0.02, below 0.05 threshold
        ctx = decider.run(ctx)
        self.assertEqual(ctx.decision.final_pick, "model-A")
        self.assertEqual(ctx.decision.decision_basis, "heuristic")
        self.assertTrue(ctx.decision.agrees_with_heuristic)

    def test_decider_swaps_when_empirical_wins_by_large_margin(self):
        from src.agents import AgentContext
        from src.agents import decider
        ctx = AgentContext(corpus="x", use_llm=False)
        ctx.recommendation = self._stub_recommendation()
        ctx.eval_report = self._stub_eval(mrr_a=0.40, mrr_b=0.65)  # +0.25 margin
        ctx = decider.run(ctx)
        self.assertEqual(ctx.decision.final_pick, "model-B")
        self.assertEqual(ctx.decision.decision_basis, "empirical")
        self.assertFalse(ctx.decision.agrees_with_heuristic)

    def test_decider_recommends_fine_tuning_when_all_low(self):
        from src.agents import AgentContext
        from src.agents import decider
        ctx = AgentContext(corpus="x", use_llm=False)
        ctx.recommendation = self._stub_recommendation()
        ctx.eval_report = self._stub_eval(mrr_a=0.10, mrr_b=0.12)  # all below 0.25 ceiling
        ctx = decider.run(ctx)
        self.assertEqual(ctx.decision.decision_basis, "fine-tuning-needed")

    def test_decider_handles_all_errored(self):
        from src.agents import AgentContext
        from src.agents import decider
        ctx = AgentContext(corpus="x", use_llm=False)
        ctx.recommendation = self._stub_recommendation()
        ctx.eval_report = self._stub_eval(mrr_a=0, mrr_b=0,
                                          err_a="oom", err_b="oom")
        ctx = decider.run(ctx)
        self.assertEqual(ctx.decision.decision_basis, "no-good-option")
        self.assertEqual(ctx.decision.confidence, "low")

    def test_llm_decider_parses_valid_json(self):
        from src.agents import AgentContext
        from src.agents import decider

        # Mock LLM that returns a clean JSON decision.
        mock_resp = mock.Mock(content=json.dumps({
            "final_pick": "model-B",
            "decision_basis": "empirical",
            "reasoning": "Margin of 0.10 MRR is real signal.",
            "confidence": "high",
        }))
        mock_llm = mock.Mock()
        mock_llm.invoke.return_value = mock_resp

        ctx = AgentContext(corpus="x", use_llm=True, llm=mock_llm)
        ctx.recommendation = self._stub_recommendation()
        ctx.eval_report = self._stub_eval(mrr_a=0.55, mrr_b=0.65)
        ctx = decider.run(ctx)
        self.assertEqual(ctx.decision.final_pick, "model-B")
        self.assertEqual(ctx.decision.decision_basis, "empirical")
        self.assertEqual(ctx.decision.confidence, "high")
        self.assertFalse(ctx.decision.agrees_with_heuristic)

    def test_llm_decider_falls_back_on_bad_json(self):
        from src.agents import AgentContext
        from src.agents import decider
        mock_resp = mock.Mock(content="I'm not even going to try to give you JSON")
        mock_llm = mock.Mock()
        mock_llm.invoke.return_value = mock_resp
        ctx = AgentContext(corpus="x", use_llm=True, llm=mock_llm)
        ctx.recommendation = self._stub_recommendation()
        ctx.eval_report = self._stub_eval(mrr_a=0.70, mrr_b=0.72)
        ctx = decider.run(ctx)
        # Should still produce a decision via the deterministic path.
        self.assertIsNotNone(ctx.decision)
        self.assertTrue(any("deterministic rules" in n for n in ctx.per_agent_notes))


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestReporterAgent(unittest.TestCase):
    def test_template_fallback_produces_markdown(self):
        from src.agents import AgentContext, Decision
        from src.agents import reporter
        ctx = AgentContext(corpus="x", use_llm=False)
        ctx.recommendation = {
            "recommended_models": [
                {"name": "model-A", "rank": 1, "dimension": 384, "size_mb": 80,
                 "rationale": "good baseline", "embed_prefix": "", "query_prefix": ""}
            ],
            "reasoning_explanation": "Small corpus, English-only.",
            "chunking_strategy": {"needed": False, "rationale": "fits"},
            "fine_tuning_advice": "Not needed.",
            "hardware_fit_analysis": "M4 Max with 48 GB.",
            "memory_warnings": [],
            "reranker_recommendation": {"name": "bge-reranker-base", "size_mb": 280, "why": "lifts recall"},
        }
        ctx.decision = Decision(
            final_pick="model-A", decision_basis="heuristic",
            reasoning="Empirical agrees with heuristic.",
            confidence="high", agrees_with_heuristic=True,
        )
        ctx = reporter.run(ctx)
        md = ctx.markdown_report
        self.assertTrue(md.startswith("# Validated Embedding-Model Recommendation"))
        self.assertIn("model-A", md)
        self.assertIn("bge-reranker-base", md)
        self.assertIn("M4 Max", md)


@unittest.skipUnless(HAS_NUMPY, "numpy not installed")
class TestOrchestratorEndToEnd(unittest.TestCase):
    @mock.patch("src.evaluator._embed_texts")
    def test_orchestrator_runs_all_agents_and_produces_full_context(self, mock_embed):
        from src.agents import run_validated_recommendation

        def fake_embed(name, texts, prefix="", batch_size=64):
            arr = np.zeros((len(texts), 20), dtype=np.float32)
            for i in range(len(texts)):
                arr[i, i % 20] = 1.0
            return arr
        mock_embed.side_effect = fake_embed

        ctx = run_validated_recommendation(
            corpus=SAMPLE_CORPUS,
            task="retrieval",
            use_llm=False,
            llm=None,
            n_queries=5,
        )
        # Every agent should have populated its output.
        self.assertIsNotNone(ctx.recommendation)
        self.assertIsNotNone(ctx.queries)
        self.assertIsNotNone(ctx.eval_report)
        self.assertIsNotNone(ctx.decision)
        self.assertIsNotNone(ctx.markdown_report)
        # Process trace should have an entry from every agent.
        agent_tags = {n.split("]")[0].strip("[") for n in ctx.per_agent_notes}
        for expected in ("suggester", "query-generator", "evaluator", "decider", "reporter"):
            self.assertIn(expected, agent_tags, f"no note from {expected!r}")

    @mock.patch("src.evaluator._embed_texts")
    def test_orchestrator_survives_single_agent_crash(self, mock_embed):
        # Make the embedder raise so the Evaluator crashes — orchestrator
        # should keep going and produce a sensible final report.
        from src.agents import run_validated_recommendation
        mock_embed.side_effect = RuntimeError("simulated embed failure")

        ctx = run_validated_recommendation(
            corpus=SAMPLE_CORPUS, task="retrieval", use_llm=False, n_queries=3,
        )
        # Suggester still ran; eval_report is populated but with errored entries.
        self.assertIsNotNone(ctx.recommendation)
        self.assertIsNotNone(ctx.eval_report)
        self.assertTrue(any(r.get("error") for r in ctx.eval_report["results"]))
        # Decider should have fallen back to "no-good-option".
        self.assertEqual(ctx.decision.decision_basis, "no-good-option")
        # Reporter still rendered something.
        self.assertTrue(ctx.markdown_report.startswith("#"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
