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
def _deterministic_decision(
    heuristic_top: str,
    eval_report: Dict[str, Any],
    available_top: Optional[str] = None,
) -> Decision:
    """No-LLM decision rules. Documented thresholds:

      MRR_SWAP_THRESHOLD: empirical winner must beat heuristic by ≥
                          this margin to justify swapping. Small swaps
                          are noise on 30 queries.

      MRR_FT_CEILING:    if NO candidate clears this, recommend
                         fine-tuning rather than picking a base model.
    """
    MRR_SWAP_THRESHOLD = 0.05
    MRR_FT_CEILING = 0.25

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

    # Highest MRR wins among the successful candidates.
    empirical_winner = max(successful, key=lambda r: r["mrr"])
    winner_name = empirical_winner["model"]
    winner_mrr = float(empirical_winner["mrr"])
    heuristic_mrr = next(
        (float(r["mrr"]) for r in successful if r["model"] == heuristic_top),
        None,
    )

    # All-low → recommend fine-tuning. The base-model choice barely matters.
    if all(float(r["mrr"]) < MRR_FT_CEILING for r in successful):
        return Decision(
            final_pick=heuristic_top,
            decision_basis="fine-tuning-needed",
            reasoning=(
                f"Every candidate scored MRR < {MRR_FT_CEILING:.2f} on the "
                "synthetic eval set. The base-model choice is not the bottleneck; "
                "domain adaptation (contrastive fine-tuning on (query, positive) "
                f"pairs from your corpus) will likely yield a bigger lift than "
                f"swapping among {[r['model'] for r in successful]}."
            ),
            confidence="medium",
            agrees_with_heuristic=True,
        )

    # If heuristic top is one of the successful candidates and the
    # empirical winner doesn't beat it by enough, stay with heuristic.
    if (
        heuristic_mrr is not None
        and winner_mrr - heuristic_mrr < MRR_SWAP_THRESHOLD
    ):
        return Decision(
            final_pick=heuristic_top,
            decision_basis="heuristic",
            reasoning=(
                f"Empirical winner is {winner_name} (MRR={winner_mrr:.3f}) but "
                f"only beats heuristic top {heuristic_top} (MRR={heuristic_mrr:.3f}) "
                f"by {winner_mrr - heuristic_mrr:+.3f} — within noise on a "
                f"30-query eval. Sticking with the heuristic pick; it's the "
                "smaller / faster / more conventional choice."
            ),
            confidence="high",
            agrees_with_heuristic=True,
        )

    # Empirical winner is materially better → swap.
    margin = (winner_mrr - heuristic_mrr) if heuristic_mrr is not None else winner_mrr
    return Decision(
        final_pick=winner_name,
        decision_basis="empirical",
        reasoning=(
            f"Empirical winner {winner_name} (MRR={winner_mrr:.3f}) beats "
            f"heuristic top {heuristic_top} (MRR={heuristic_mrr or 0:.3f}) by "
            f"{margin:+.3f} on the synthetic eval — large enough to be real "
            "signal, not noise. Recommending the empirical winner."
        ),
        confidence="high",
        agrees_with_heuristic=False,
    )


# ----------------------------------------------------------------------
# LLM-driven decision (with deterministic fallback on parse failure)
# ----------------------------------------------------------------------
_DECIDER_PROMPT = """You are the Decider agent in a multi-agent embedding-model recommender.

A heuristic suggester already proposed candidates. An empirical evaluator
then ran them on synthetic (query, source-doc) pairs sampled from the
user's actual corpus, and reported per-model MRR / nDCG@10 / recall@k.

Your job: pick the single best model for the user, weighing both signals.

Heuristic suggestion (top-3, in order):
{heuristic_models}

Empirical results (sorted by MRR descending):
{empirical_table}

Heuristic top: {heuristic_top}
Empirical top: {empirical_top}

Consider:
  - Effect size: small MRR differences on a 30-query eval are noise.
    A real difference is typically >= 0.05.
  - Model footprint: prefer smaller / faster models when quality is
    comparable.
  - Operational pragmatism: a model that is 0.02 MRR worse but 5x
    smaller is usually the better choice.
  - If ALL candidates score MRR < 0.25, the base-model choice is not
    the bottleneck — recommend fine-tuning instead.
  - If all candidates errored, return decision_basis "no-good-option".

Reply with exactly this JSON object, nothing else:

{{
  "final_pick": "<exact model name from the candidates>",
  "decision_basis": "heuristic" | "empirical" | "fine-tuning-needed" | "no-good-option",
  "reasoning": "<2-3 sentence plain-English explanation a user can act on>",
  "confidence": "high" | "medium" | "low"
}}
"""


def _format_empirical_table(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "(no successful results)"
    sorted_r = sorted(results, key=lambda r: -float(r.get("mrr") or 0))
    lines = ["  Model                                              MRR    nDCG@10  recall@10"]
    for r in sorted_r:
        err = " (FAILED)" if r.get("error") else ""
        lines.append(
            f"  {r['model']:50s} {float(r.get('mrr') or 0):.3f}  "
            f"{float(r.get('ndcg_at_10') or 0):.3f}     {float(r.get('recall_at_10') or 0):.3f}{err}"
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
    prompt = _DECIDER_PROMPT.format(
        heuristic_models=_format_heuristic_list(heuristic_models),
        empirical_table=_format_empirical_table(eval_report.get("results") or []),
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
