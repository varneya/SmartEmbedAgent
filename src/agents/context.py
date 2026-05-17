"""
Shared context passed between agents in the validated-recommendation
pipeline. Each agent reads what previous agents wrote and writes its
own outputs. Functional style — agents return the updated context
rather than mutating it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Decision:
    """Output of the Decider agent — the final pick + why."""
    final_pick: str             # model name chosen as the final recommendation
    decision_basis: str         # "heuristic" | "empirical" | "fine-tuning-needed" | "no-good-option"
    reasoning: str              # plain-English explanation
    confidence: str             # "high" | "medium" | "low"
    agrees_with_heuristic: bool # convenience flag

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_pick": self.final_pick,
            "decision_basis": self.decision_basis,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "agrees_with_heuristic": self.agrees_with_heuristic,
        }


@dataclass
class AgentContext:
    """The single source of truth threaded through the pipeline.

    Inputs (set by the caller before run_validated_recommendation):
        corpus, config, task, use_llm, llm, n_queries, seed

    Outputs (filled in as agents run):
        recommendation       — from Suggester
        queries              — from QueryGenerator
        eval_report          — from Evaluator (Dict form of evaluator.EvalReport)
        decision             — from Decider
        markdown_report      — from Reporter
        per_agent_notes      — each agent appends short status lines
    """
    # ---- inputs ----
    corpus: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    task: str = "retrieval"
    use_llm: bool = True               # whether to use LLM-driven agents (Suggester overlay, Decider, Reporter)
    llm: Optional[Any] = None          # LangChain BaseChatModel-shaped object, or None
    n_queries: int = 30                # for QueryGenerator
    seed: int = 0                      # for QueryGenerator (reproducible sampling)

    # ---- outputs ----
    recommendation: Optional[Dict[str, Any]] = None
    queries: Optional[List[Any]] = None        # [(query, doc_idx), ...]
    query_source: Optional[str] = None         # "llm" | "keyword-fallback" | "none"
    eval_report: Optional[Dict[str, Any]] = None
    decision: Optional[Decision] = None
    markdown_report: Optional[str] = None
    per_agent_notes: List[str] = field(default_factory=list)

    def add_note(self, agent: str, message: str) -> None:
        """Append a short [agent] status line. The orchestrator returns
        these so the UI / CLI can show what each step did."""
        self.per_agent_notes.append(f"[{agent}] {message}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the runnable state — excludes the LLM handle and
        the raw queries (which are non-trivial to JSON-encode)."""
        return {
            "task": self.task,
            "used_llm": bool(self.llm) if self.use_llm else False,
            "n_queries": self.n_queries,
            "seed": self.seed,
            "recommendation": self.recommendation,
            "query_source": self.query_source,
            "n_queries_generated": len(self.queries) if self.queries else 0,
            "eval_report": self.eval_report,
            "decision": self.decision.to_dict() if self.decision else None,
            "markdown_report": self.markdown_report,
            "per_agent_notes": self.per_agent_notes,
        }
