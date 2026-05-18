"""
Evaluator agent — runs each candidate against the queries, computes
metrics. Wraps src.evaluator.evaluate_candidates.

Output: ctx.eval_report populated with the full EvalReport dict.

This is the slow step (1-5 min typical) because it actually loads each
sentence-transformers model and runs encode() on the full corpus.
"""

from __future__ import annotations

from src.evaluator import evaluate_candidates
from .context import AgentContext


def run(ctx: AgentContext) -> AgentContext:
    if not ctx.recommendation:
        ctx.add_note("evaluator", "Skipped — no recommendation from Suggester.")
        return ctx

    candidates = ctx.recommendation.get("recommended_models") or []
    if not candidates:
        ctx.add_note("evaluator", "Skipped — Suggester produced no candidates.")
        return ctx

    heuristic_top = candidates[0]["name"]
    # Pass the queries that QueryGenerator produced; if it failed, the
    # evaluate_candidates call will re-generate them internally (using
    # whatever fallback) — but we prefer to honor what the previous
    # agent did. Re-derive from ctx.corpus to keep evaluate_candidates'
    # signature unchanged (it accepts corpus + candidates and generates
    # its own queries). The shared `seed` keeps them aligned with
    # QueryGenerator's output.
    report = evaluate_candidates(
        corpus=ctx.corpus,
        candidates=candidates,
        n_queries=ctx.n_queries,
        llm=ctx.llm if ctx.use_llm else None,
        seed=ctx.seed,
        heuristic_top_model=heuristic_top,
        task=ctx.task,
    )
    ctx.eval_report = report.to_dict()

    successful = [r for r in report.results if not r.error]
    if not successful:
        ctx.add_note("evaluator", "All candidates failed during embedding/inference. See per-row errors.")
    else:
        # Rank on the task's primary metric — for retrieval this is MRR,
        # for classification it's macro-F1, etc. The EvalResult schema
        # exposes both legacy fields (mrr / ndcg_at_10) and the uniform
        # primary_metric_value, so the Decider downstream stays simple.
        top = max(successful, key=lambda r: r.primary_metric_value)
        ctx.add_note(
            "evaluator",
            f"Empirical winner: {top.model} "
            f"({top.primary_metric_name}={top.primary_metric_value:.3f}). "
            + ("Diverges from heuristic top." if report.diverged
               else "Agrees with heuristic top.")
        )
    return ctx
