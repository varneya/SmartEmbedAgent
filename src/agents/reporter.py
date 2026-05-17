"""
Reporter agent — synthesizes the whole pipeline into a single readable
Markdown report. Sections: corpus profile, candidates considered,
empirical results table, final decision + reasoning, suggested next
steps (chunking, fine-tuning, reranker pairing, prompt prefixes).

LLM-driven variant produces a flowing narrative. Deterministic
fallback produces a structured template — same information, less
polished prose. The template fallback never depends on the LLM, so the
agent always produces something useful.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .context import AgentContext


# ----------------------------------------------------------------------
# Deterministic template (fallback / no-LLM mode)
# ----------------------------------------------------------------------
def _format_eval_row(r: Dict[str, Any]) -> str:
    err = " (FAILED — see error field)" if r.get("error") else ""
    return (
        f"| `{r['model']}` | {float(r['mrr']):.3f} | {float(r['ndcg_at_10']):.3f} | "
        f"{float(r['recall_at_5']):.3f} | {float(r['recall_at_10']):.3f} | "
        f"{float(r['elapsed_seconds']):.1f}s{err} |"
    )


def _template_report(ctx: AgentContext) -> str:
    rec = ctx.recommendation or {}
    decision = ctx.decision
    eval_report = ctx.eval_report or {}

    lines: List[str] = [
        "# Validated Embedding-Model Recommendation",
        "",
        "*Produced by the multi-agent pipeline: Suggester → QueryGenerator → Evaluator → Decider → Reporter.*",
        "",
    ]

    # --- Final pick ---
    if decision:
        lines += [
            "## Final pick",
            "",
            f"**`{decision.final_pick}`** ({decision.decision_basis}, confidence: {decision.confidence})",
            "",
            decision.reasoning,
            "",
        ]
        if not decision.agrees_with_heuristic:
            lines.append("> ⚠ This differs from the heuristic top-pick. The empirical evidence supports the swap.")
            lines.append("")

    # --- Corpus profile ---
    reasoning = rec.get("reasoning_explanation") or ""
    if reasoning:
        lines += ["## Corpus profile", "", reasoning, ""]

    # --- Candidates considered ---
    candidates = rec.get("recommended_models") or []
    if candidates:
        lines += ["## Candidates considered (heuristic ranking)", ""]
        for m in candidates:
            lines.append(
                f"- **`{m.get('name', '?')}`** — "
                f"dim {m.get('dimension', '?')}, "
                f"ctx {m.get('context_window', '?')}, "
                f"{m.get('size_mb', '?')} MB. {m.get('rationale', '')}"
            )
            ep, qp = m.get("embed_prefix", ""), m.get("query_prefix", "")
            if ep or qp:
                lines.append(f"  - **Prefixes:** index docs with `{ep!r}`, queries with `{qp!r}`.")
        lines.append("")

    # --- Empirical results ---
    results = eval_report.get("results") or []
    if results:
        lines += [
            "## Empirical evaluation",
            "",
            f"Generated **{eval_report.get('n_queries_generated', 0)}** synthetic queries "
            f"via *{eval_report.get('query_source', 'unknown')}*"
            + (f" ({eval_report.get('llm_model_used')})" if eval_report.get('llm_model_used') else "")
            + ".",
            "",
            "| Model | MRR | nDCG@10 | recall@5 | recall@10 | Time |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        # Sort by MRR descending so the winner is at the top.
        for r in sorted(results, key=lambda r: -float(r.get("mrr") or 0)):
            lines.append(_format_eval_row(r))
        lines.append("")
        if eval_report.get("diverged"):
            lines.append(
                f"> The empirical winner (`{eval_report.get('empirical_top', '?')}`) "
                f"differs from the heuristic top (`{eval_report.get('heuristic_top', '?')}`). "
                "The Decider weighed both signals and produced the final pick above."
            )
            lines.append("")

    # --- Operational notes ---
    chunk = rec.get("chunking_strategy") or {}
    if chunk:
        lines += ["## Chunking", ""]
        if chunk.get("needed"):
            lines.append(
                f"**Recommended.** Chunk size: {chunk.get('chunk_size_tokens')} tokens, "
                f"overlap: {chunk.get('overlap_tokens')} tokens."
            )
        else:
            lines.append("**Not needed.** " + (chunk.get("rationale") or ""))
        lines.append("")

    ft = rec.get("fine_tuning_advice")
    if ft:
        lines += ["## Fine-tuning", "", ft, ""]

    rr = rec.get("reranker_recommendation") or {}
    if rr.get("name"):
        lines += [
            "## Reranker",
            "",
            f"**Suggested:** `{rr['name']}` (~{rr.get('size_mb', '?')} MB). {rr.get('why', '')}",
            "",
        ]

    hw = rec.get("hardware_fit_analysis")
    if hw:
        lines += ["## Hardware fit", "", hw, ""]

    mem = rec.get("memory_warnings") or []
    if mem:
        lines += ["## Operational warnings", ""]
        for w in mem:
            lines.append(f"- ⚠ {w}")
        lines.append("")

    # --- Process trace ---
    if ctx.per_agent_notes:
        lines += ["## Process trace", ""]
        for n in ctx.per_agent_notes:
            lines.append(f"- {n}")
        lines.append("")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# LLM-driven narrative (with template fallback on failure)
# ----------------------------------------------------------------------
_REPORTER_PROMPT = """You are the Reporter agent in a multi-agent embedding-model recommender. Synthesize the result of the pipeline into a clean Markdown report a user can act on.

Required sections, in this order:

  1. # Validated Embedding-Model Recommendation
  2. ## Final pick — the chosen model + the Decider's reasoning. If the
     pick disagrees with the heuristic top, call that out explicitly.
  3. ## Corpus profile — short summary of what the corpus looked like.
  4. ## Candidates considered — heuristic top-3 with one-line each.
  5. ## Empirical evaluation — Markdown table of MRR / nDCG@10 / recall.
  6. ## Chunking / Fine-tuning / Reranker / Hardware — only sections
     that have content. Don't fabricate.
  7. ## Operational warnings — if memory_warnings is non-empty.

Tone: confident, concise, no marketing fluff. Use code spans for model
names. Don't add sections not listed above. Don't include the JSON itself.
Don't hallucinate metrics — only use what's in the data below.

Pipeline output (JSON):

{context_json}

Output ONLY the Markdown report, starting with the H1. No preamble."""


def _llm_report(ctx: AgentContext) -> Optional[str]:
    if not ctx.llm:
        return None
    import json
    payload = ctx.to_dict()
    # Strip the per_agent_notes from the context we hand to the LLM —
    # they're for transparency, not for the LLM to paraphrase. Same for
    # markdown_report (the field we're about to populate).
    payload.pop("per_agent_notes", None)
    payload.pop("markdown_report", None)
    try:
        resp = ctx.llm.invoke(_REPORTER_PROMPT.format(
            context_json=json.dumps(payload, indent=2)[:12000]   # cap to be polite
        ))
        text = getattr(resp, "content", None) or str(resp)
        text = text.strip()
        if text.startswith("# "):
            return text
        # Reporter sometimes adds preamble despite being told not to — try
        # to find the first H1.
        idx = text.find("\n# ")
        if idx >= 0:
            return text[idx + 1:].strip()
        return None
    except Exception:
        return None


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def run(ctx: AgentContext) -> AgentContext:
    if not ctx.recommendation:
        ctx.markdown_report = "# Validated Recommendation\n\n_No recommendation was produced._\n"
        ctx.add_note("reporter", "Skipped — no Suggester output.")
        return ctx

    # Try LLM first if available; fall back to the template.
    md = _llm_report(ctx) if ctx.use_llm and ctx.llm is not None else None
    if md is None:
        if ctx.use_llm and ctx.llm is not None:
            ctx.add_note("reporter", "LLM narrative failed; rendered structured template instead.")
        else:
            ctx.add_note("reporter", "Rendered structured template (no LLM available).")
        md = _template_report(ctx)
    else:
        ctx.add_note("reporter", "Rendered LLM-narrative report.")

    ctx.markdown_report = md
    return ctx
