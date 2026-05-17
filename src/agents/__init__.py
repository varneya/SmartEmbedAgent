"""
Multi-agent orchestration for the validated recommendation pipeline.

Five specialized agents, run in sequence, each with a single
responsibility and a clear input → output contract:

    Suggester        → top-3 candidates (heuristic + optional LLM overlay)
    QueryGenerator   → synthetic (query, doc_idx) pairs for evaluation
    Evaluator        → per-candidate MRR / nDCG@10 / recall@k
    Decider          → final pick + reasoning (LLM-driven, deterministic fallback)
    Reporter         → narrative Markdown report tying it all together

The orchestrator (`run_validated_recommendation`) sequences them and
returns a single `AgentContext` containing every intermediate result so
callers can render whichever pieces matter to them.

Each agent is functional: `agent.run(ctx) -> ctx`. Side-effect-free
except for LLM calls and embedding loads, which are expensive but
intentional. The orchestrator never silently swaps logic; if an agent
can't run (no LLM available, etc.) it surfaces a note and the downstream
agents adapt.
"""

from .context import AgentContext, Decision
from .orchestrator import run_validated_recommendation

__all__ = ["AgentContext", "Decision", "run_validated_recommendation"]
