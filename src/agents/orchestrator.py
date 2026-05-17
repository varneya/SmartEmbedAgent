"""
Orchestrator — runs all 5 agents in sequence and returns the final
AgentContext.

The Evaluator step is what makes this slow (1-5 min on a typical
corpus); the other four agents combined are sub-second. Caller is
responsible for surfacing a "this is going to take a while" hint to
the user before invoking.

Failure handling: each agent is wrapped so a single agent's failure
doesn't kill the whole pipeline. The downstream agents adapt — e.g.
if the Evaluator fails, the Decider falls back to "no-good-option"
mode and the Reporter produces a report that says so.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from . import decider, evaluator_agent, query_generator, reporter, suggester
from .context import AgentContext

logger = logging.getLogger("agents.orchestrator")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[orchestrator] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


_AGENT_PIPELINE = [
    ("suggester", suggester.run),
    ("query-generator", query_generator.run),
    ("evaluator", evaluator_agent.run),
    ("decider", decider.run),
    ("reporter", reporter.run),
]


def _safe_run(name: str, fn, ctx: AgentContext) -> AgentContext:
    """Catch exceptions per-agent so one failure doesn't sink the pipeline.
    The agent's own note (if any) gets prefixed; the exception is logged
    and turned into a user-visible note."""
    try:
        logger.info("Running %s...", name)
        return fn(ctx)
    except Exception as e:
        logger.exception("Agent %s failed: %s", name, e)
        ctx.add_note(name, f"Agent crashed ({type(e).__name__}: {e}). Downstream agents will adapt.")
        return ctx


def run_validated_recommendation(
    corpus: str,
    config: Optional[Dict[str, Any]] = None,
    task: str = "retrieval",
    use_llm: bool = True,
    llm: Optional[Any] = None,
    n_queries: int = 30,
    seed: int = 0,
) -> AgentContext:
    """End-to-end validated recommendation pipeline.

    Parameters mirror the underlying run_pipeline_no_llm / evaluate_candidates
    signatures so callers can pass through user choices uniformly.

    Returns the final AgentContext with every intermediate result populated
    (recommendation, queries, eval_report, decision, markdown_report,
    per_agent_notes).
    """
    ctx = AgentContext(
        corpus=corpus or "",
        config=config or {},
        task=task,
        use_llm=use_llm,
        llm=llm,
        n_queries=n_queries,
        seed=seed,
    )
    for name, fn in _AGENT_PIPELINE:
        ctx = _safe_run(name, fn, ctx)
    return ctx
