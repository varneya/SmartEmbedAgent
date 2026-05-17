"""
QueryGenerator agent — produces (query, source_doc_idx) pairs for the
empirical evaluation step. Wraps src.evaluator.generate_eval_queries.

Output:
    ctx.queries        — list of (query, doc_idx) tuples
    ctx.query_source   — "llm" if the LLM was used, "keyword-fallback" otherwise
"""

from __future__ import annotations

from src.evaluator import _split_corpus, generate_eval_queries
from .context import AgentContext


def run(ctx: AgentContext) -> AgentContext:
    docs = _split_corpus(ctx.corpus)
    if not docs:
        ctx.queries = []
        ctx.query_source = "none"
        ctx.add_note("query-generator", "Corpus is empty after splitting; no queries generated.")
        return ctx

    pairs, source = generate_eval_queries(
        docs=docs,
        n=ctx.n_queries,
        llm=ctx.llm if ctx.use_llm else None,
        seed=ctx.seed,
    )
    ctx.queries = pairs
    ctx.query_source = source

    if source == "keyword-fallback":
        ctx.add_note(
            "query-generator",
            f"Generated {len(pairs)} queries via keyword extraction (no LLM available). "
            "Metrics will still rank candidates correctly relative to each other."
        )
    else:
        ctx.add_note(
            "query-generator",
            f"Generated {len(pairs)} natural-language queries via the local LLM."
        )
    return ctx
