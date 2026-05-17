"""
Suggester agent — produces the initial top-3 recommendation.

Thin wrapper around the existing deterministic pipeline
(`run_pipeline_no_llm`). The LLM-overlay path (build_agent + invoke)
runs at the orchestrator level when use_llm=True so the merge logic
stays in one place.

Output: ctx.recommendation populated with the full schema
(recommended_models, chunking_strategy, fine_tuning_advice,
hardware_fit_analysis, index_estimate, reranker_recommendation,
language_profile, memory_warnings, task, recommended_models_available,
available_memory_budget_mb, available_memory_basis_gb).
"""

from __future__ import annotations

from typing import Any, Dict

from src.agent_orchestrator import run_pipeline_no_llm
from src.config_validator import extract_pii_config
from .context import AgentContext


def run(ctx: AgentContext) -> AgentContext:
    """Produce the deterministic recommendation. The LLM overlay is
    applied at the orchestrator level so the merge stays in one place."""
    pii_cfg = extract_pii_config(ctx.config or {})
    ctx.recommendation = run_pipeline_no_llm(
        corpus=ctx.corpus,
        user_config=pii_cfg,
        task=ctx.task,
    )
    top = (ctx.recommendation.get("recommended_models") or [{}])[0]
    ctx.add_note(
        "suggester",
        f"Heuristic top: {top.get('name', '?')} "
        f"(task={ctx.task}; {len(ctx.recommendation.get('recommended_models', []))} candidates)."
    )
    return ctx
