"""
Decider agent — looks at the Suggester's heuristic recommendation AND
the Evaluator's empirical metrics, decides on a single final pick.

Three possible decisions:
  - "heuristic": confirm the Suggester's top-1
  - "empirical": swap to the empirical winner from the eval report
  - "fine-tuning-needed": all candidates score poorly; recommend
    domain adaptation rather than picking a base model
  - "no-good-option": no candidate succeeded at all (all errored)

The LLM-driven path asks a local Ollama model to reason about effect
size, model footprint, and operational trade-offs. When the LLM is
unavailable, a deterministic decision tree is used — same conclusions
for the common cases, less nuance for the edge cases.

Output: ctx.decision populated with a Decision dataclass.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .context import AgentContext, Decision


# ----------------------------------------------------------------------
# Deterministic decision tree (used as fallback OR when LLM is disabled)
# ----------------------------------------------------------------------
# Task-aware thresholds. Different metrics have different noise floors
# and different "this model is fundamentally inadequate" cutoffs.
#
#   SWAP_THRESHOLD: empirical winner must beat heuristic top by ≥
#                   this on the primary metric to justify swapping.
#                   Smaller swaps are noise.
#   FT_CEILING:     if no successful candidate clears this, recommend
#                   fine-tuning rather than picking a base model.
_TASK_SWAP_THRESHOLD: Dict[str, float] = {
    "retrieval": 0.05,        # MRR
    "classification": 0.05,   # F1 macro
    "clustering": 0.05,       # V-measure
    "deduplication": 0.02,    # AUC (small moves are real)
    "similarity": 0.05,       # Spearman ρ
}
_TASK_FT_CEILING: Dict[str, float] = {
    "retrieval": 0.25,        # MRR
    "classification": 0.50,   # macro-F1 — a coin flip on 2 classes is 0.5
    "clustering": 0.20,       # V-measure
    "deduplication": 0.70,    # AUC — below 0.7 means embeddings don't separate dupes
    "similarity": 0.20,       # Spearman ρ
}
# Default primary-metric NAME per task, for older eval reports that don't
# emit `primary_metric_name`. Keeps back-compat with existing test stubs
# that pre-date the task-aware schema.
_TASK_DEFAULT_PRIMARY: Dict[str, str] = {
    "retrieval": "mrr",
    "classification": "f1_macro",
    "clustering": "v_measure",
    "deduplication": "auc",
    "similarity": "spearman",
}


def _read_metric(r: Dict[str, Any], primary: str) -> float:
    """Read the primary metric for a single result, falling back to MRR
    for older eval reports that don't yet emit primary_metric_value."""
    val = r.get("primary_metric_value")
    if val is not None:
        return float(val)
    # Legacy / retrieval-only fallback
    if primary == "mrr":
        return float(r.get("mrr") or 0.0)
    return float(r.get(primary) or 0.0)


def _deterministic_decision(
    heuristic_top: str,
    eval_report: Dict[str, Any],
    available_top: Optional[str] = None,
) -> Decision:
    """No-LLM decision rules. Task-aware: reads the report's
    `primary_metric_name` and uses the matching swap / FT thresholds."""
    task = eval_report.get("task", "retrieval")
    primary = eval_report.get("primary_metric_name") or _TASK_DEFAULT_PRIMARY.get(task, "mrr")
    SWAP_THRESHOLD = _TASK_SWAP_THRESHOLD.get(task, 0.05)
    FT_CEILING = _TASK_FT_CEILING.get(task, 0.25)

    results: List[Dict[str, Any]] = eval_report.get("results", []) or []
    successful = [r for r in results if not r.get("error")]
    if not successful:
        return Decision(
            final_pick=heuristic_top,
            decision_basis="no-good-option",
            reasoning=(
                "All candidates errored during empirical evaluation (likely a "
                "model-download or memory failure). Falling back to the "
                "heuristic top-pick as a placeholder; rerun /evaluate after "
                "fixing the underlying issue to get an empirical answer."
            ),
            confidence="low",
            agrees_with_heuristic=True,
        )

    # Highest primary-metric value wins among successful candidates.
    empirical_winner = max(successful, key=lambda r: _read_metric(r, primary))
    winner_name = empirical_winner["model"]
    winner_score = _read_metric(empirical_winner, primary)
    heuristic_score = next(
        (_read_metric(r, primary) for r in successful if r["model"] == heuristic_top),
        None,
    )

    # All-low → recommend fine-tuning. The base-model choice barely matters.
    if all(_read_metric(r, primary) < FT_CEILING for r in successful):
        return Decision(
            final_pick=heuristic_top,
            decision_basis="fine-tuning-needed",
            reasoning=(
                f"Every candidate scored {primary} < {FT_CEILING:.2f} on the "
                f"synthetic {task} eval. The base-model choice is not the "
                "bottleneck; domain adaptation (contrastive fine-tuning on "
                f"task-specific pairs from your corpus) will likely yield a bigger "
                f"lift than swapping among {[r['model'] for r in successful]}."
            ),
            confidence="medium",
            agrees_with_heuristic=True,
        )

    # If heuristic top is one of the successful candidates and the
    # empirical winner doesn't beat it by enough, stay with heuristic.
    if (
        heuristic_score is not None
        and winner_score - heuristic_score < SWAP_THRESHOLD
    ):
        return Decision(
            final_pick=heuristic_top,
            decision_basis="heuristic",
            reasoning=(
                f"Empirical winner is {winner_name} ({primary}={winner_score:.3f}) "
                f"but only beats heuristic top {heuristic_top} "
                f"({primary}={heuristic_score:.3f}) by "
                f"{winner_score - heuristic_score:+.3f} — within noise on a "
                f"{len(successful)}-candidate {task} eval. Sticking with the "
                "heuristic pick; it's the smaller / faster / more conventional choice."
            ),
            confidence="high",
            agrees_with_heuristic=True,
        )

    # Empirical winner is materially better → swap.
    margin = (winner_score - heuristic_score) if heuristic_score is not None else winner_score
    return Decision(
        final_pick=winner_name,
        decision_basis="empirical",
        reasoning=(
            f"Empirical winner {winner_name} ({primary}={winner_score:.3f}) beats "
            f"heuristic top {heuristic_top} ({primary}={heuristic_score or 0:.3f}) "
            f"by {margin:+.3f} on the synthetic {task} eval — large enough to be "
            "real signal, not noise. Recommending the empirical winner."
        ),
        confidence="high",
        agrees_with_heuristic=False,
    )


# ----------------------------------------------------------------------
# LLM-driven decision (with deterministic fallback on parse failure)
# ----------------------------------------------------------------------
_DECIDER_PROMPT = """You are the Decider agent in a multi-agent embedding-model recommender.

A heuristic suggester already proposed candidates. An empirical evaluator
then ran them on synthetic supervision derived from the user's actual
corpus, and reported per-model {primary_metric} (the primary metric for
task={task}).

Your job: pick the single best model for the user, weighing both signals.

Heuristic suggestion (top-3, in order):
{heuristic_models}

Empirical results (sorted by {primary_metric} descending):
{empirical_table}

Heuristic top: {heuristic_top}
Empirical top: {empirical_top}

Consider:
  - Effect size: small {primary_metric} differences on a ~30-sample eval
    are noise. A real difference is typically >= {swap_threshold:.2f}.
  - Model footprint: prefer smaller / faster models when quality is
    comparable.
  - Operational pragmatism: a model that is slightly worse but 5x smaller
    is usually the better choice.
  - If ALL candidates score {primary_metric} < {ft_ceiling:.2f}, the
    base-model choice is not the bottleneck — recommend fine-tuning.
  - If all candidates errored, return decision_basis "no-good-option".

Reply with exactly this JSON object, nothing else:

{{
  "final_pick": "<exact model name from the candidates>",
  "decision_basis": "heuristic" | "empirical" | "fine-tuning-needed" | "no-good-option",
  "reasoning": "<2-3 sentence plain-English explanation a user can act on>",
  "confidence": "high" | "medium" | "low"
}}
"""


def _format_empirical_table(results: List[Dict[str, Any]], primary: str) -> str:
    if not results:
        return "(no successful results)"
    sorted_r = sorted(results, key=lambda r: -_read_metric(r, primary))
    lines = [f"  Model                                              {primary}"]
    for r in sorted_r:
        err = " (FAILED)" if r.get("error") else ""
        lines.append(
            f"  {r['model']:50s} {_read_metric(r, primary):.3f}{err}"
        )
    return "\n".join(lines)


def _format_heuristic_list(heuristic_models: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"  {m.get('rank', i+1)}. {m.get('name', '?')} "
        f"(dim {m.get('dimension', '?')}, {m.get('size_mb', '?')} MB)"
        for i, m in enumerate(heuristic_models)
    )


def _extract_decision_json(text: str) -> Optional[Dict[str, Any]]:
    """Salvage a JSON object from imperfect LLM output (markdown fences,
    preamble, etc.). Returns None if nothing valid is found."""
    if not text:
        return None
    candidates = [text.strip()]
    # Markdown fence
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1))
    # First { through last }
    first, last = text.find("{"), text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first:last + 1])
    for c in candidates:
        try:
            parsed = json.loads(c)
            if isinstance(parsed, dict) and "final_pick" in parsed:
                return parsed
        except (TypeError, json.JSONDecodeError):
            continue
    return None


def _llm_decision(
    llm: Any,
    heuristic_models: List[Dict[str, Any]],
    eval_report: Dict[str, Any],
) -> Optional[Decision]:
    """Returns None if the LLM call or parsing fails; caller falls back
    to the deterministic decision tree."""
    heuristic_top = (heuristic_models or [{}])[0].get("name", "?")
    empirical_top = eval_report.get("empirical_top") or "?"
    task = eval_report.get("task", "retrieval")
    primary = eval_report.get("primary_metric_name") or "mrr"
    prompt = _DECIDER_PROMPT.format(
        task=task,
        primary_metric=primary,
        swap_threshold=_TASK_SWAP_THRESHOLD.get(task, 0.05),
        ft_ceiling=_TASK_FT_CEILING.get(task, 0.25),
        heuristic_models=_format_heuristic_list(heuristic_models),
        empirical_table=_format_empirical_table(eval_report.get("results") or [], primary),
        heuristic_top=heuristic_top,
        empirical_top=empirical_top,
    )
    try:
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", None) or str(resp)
    except Exception:
        return None

    parsed = _extract_decision_json(text)
    if not parsed:
        return None

    final_pick = str(parsed.get("final_pick", "")).strip()
    if not final_pick:
        return None

    valid_bases = {"heuristic", "empirical", "fine-tuning-needed", "no-good-option"}
    basis = str(parsed.get("decision_basis", "heuristic"))
    if basis not in valid_bases:
        basis = "heuristic"
    confidence = str(parsed.get("confidence", "medium"))
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"

    return Decision(
        final_pick=final_pick,
        decision_basis=basis,
        reasoning=str(parsed.get("reasoning", "")).strip() or "(no reasoning provided)",
        confidence=confidence,
        agrees_with_heuristic=(final_pick == heuristic_top),
    )


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def run(ctx: AgentContext) -> AgentContext:
    if not ctx.recommendation:
        ctx.add_note("decider", "Skipped — no Suggester output to decide on.")
        return ctx

    heuristic_models = ctx.recommendation.get("recommended_models") or []
    if not heuristic_models:
        ctx.add_note("decider", "Skipped — Suggester returned no candidates.")
        return ctx
    heuristic_top = heuristic_models[0]["name"]

    if not ctx.eval_report:
        # No eval ran — there's nothing to override the heuristic with.
        ctx.decision = Decision(
            final_pick=heuristic_top,
            decision_basis="heuristic",
            reasoning="No empirical evaluation was run; defaulting to heuristic top-pick.",
            confidence="medium",
            agrees_with_heuristic=True,
        )
        ctx.add_note("decider", f"No eval — confirming heuristic top {heuristic_top}.")
        return ctx

    available_top = None
    avail = ctx.recommendation.get("recommended_models_available") or []
    if avail:
        available_top = avail[0]["name"]

    # Try LLM first if available; fall back to the deterministic tree on
    # any failure (LLM unavailable, non-JSON output, etc.)
    decision: Optional[Decision] = None
    if ctx.use_llm and ctx.llm is not None:
        decision = _llm_decision(ctx.llm, heuristic_models, ctx.eval_report)
        if decision is None:
            ctx.add_note("decider", "LLM decider failed or returned bad JSON; using deterministic rules.")

    if decision is None:
        decision = _deterministic_decision(heuristic_top, ctx.eval_report, available_top)

    ctx.decision = decision
    ctx.add_note(
        "decider",
        f"Final pick: {decision.final_pick} ({decision.decision_basis}, "
        f"confidence={decision.confidence})."
    )
    return ctx
