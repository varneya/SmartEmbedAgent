"""
Suggester agent — produces the initial top-3 recommendation.

Delegates to `agent_orchestrator.recommend_with_optional_llm`, which is
the SAME helper the FastAPI `/recommend` endpoint calls. That guarantees
the Suggester's output and `/recommend`'s output match on the same
inputs — historically they didn't, because the Suggester always took
the heuristic-only path regardless of `ctx.use_llm`, while `/recommend`
applied the LLM overlay. That divergence made the validated-workflow
final pick look like it disagreed with the top recommendation card.

When `ctx.use_llm=True` and `ctx.llm` is provided, the LLM overlay is
applied on top of the deterministic heuristic. On any LLM failure
(server down, bad JSON, wrong shape), the helper falls back to pure
heuristic and surfaces the reason in `notes`, which we forward into
`ctx.per_agent_notes` so the UI sees what happened.

Output: ctx.recommendation populated with the full schema
(recommended_models, chunking_strategy, fine_tuning_advice,
hardware_fit_analysis, index_estimate, reranker_recommendation,
language_profile, memory_warnings, task, recommended_models_available,
available_memory_budget_mb, available_memory_basis_gb).
"""

from __future__ import annotations

from src.agent_orchestrator import recommend_with_optional_llm
from .context import AgentContext


def run(ctx: AgentContext) -> AgentContext:
    """Produce the recommendation, with optional LLM overlay when
    `ctx.use_llm=True` and `ctx.llm` is available."""
    use_llm = bool(ctx.use_llm and ctx.llm is not None)
    rec, notes = recommend_with_optional_llm(
        corpus=ctx.corpus,
        config=ctx.config or {},
        task=ctx.task,
        use_llm=use_llm,
        llm=ctx.llm,
        verbose=False,
    )
    ctx.recommendation = rec
    for n in notes:
        ctx.add_note("suggester", n)

    top = (rec.get("recommended_models") or [{}])[0]
    # Phrasing kept as "Heuristic top" / "LLM-overlaid top" so the existing
    # test (which asserts the literal "Heuristic top" string for the no-LLM
    # path) and downstream log scrapers stay backwards-compatible.
    path_label = "LLM-overlaid top" if use_llm else "Heuristic top"
    ctx.add_note(
        "suggester",
        f"{path_label}: {top.get('name', '?')} "
        f"(task={ctx.task}; {len(rec.get('recommended_models', []))} candidates).",
    )
    return ctx
