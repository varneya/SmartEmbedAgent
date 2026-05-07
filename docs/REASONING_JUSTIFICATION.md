# Why an LLM Agent (and Not a Deterministic Script)

A reasonable question for any project framed as "agentic" is: would a deterministic script have done the job? For embedding model recommendation specifically, the answer depends on the step. Some parts of the pipeline are well served by deterministic code; others require judgment that doesn't compose cleanly into a scoring function. This document goes through each step and explains the choice.

## Step 1 — Hardware profiling

**Verdict: deterministic, no LLM needed.**

`psutil.virtual_memory().total`, `torch.cuda.is_available()`, and `torch.cuda.get_device_properties()` return ground-truth values. There's no judgment involved; the user has 16 GB of RAM or they don't. Running an LLM on this would be wasteful and would introduce non-determinism into a step that should be exact.

This is implemented in `src/device_profiler.py` as a pure function that returns a fixed-shape dict.

## Step 2 — PII redaction

**Verdict: mostly deterministic, with a pretrained model for the NER stage.**

Regex patterns for emails, phone numbers, SSNs, credit cards, and IPs are deterministic. Running them through an LLM would be slower and less reliable than `re.finditer`.

The NER stage uses a pretrained Hugging Face model — `dslim/bert-base-NER` by default — but this is a token-classification model, not an LLM with reasoning capabilities. It's a fixed-output classifier; given the same input, it produces the same entities every time.

The user's whitelist and force-redaction list are applied through string equality and regex. Again, deterministic.

The reason this step doesn't need an LLM: PII detection is a recognition task with a well-defined output space. An LLM could do it, but at a much higher cost and with worse accuracy on the patterns the regex catches reliably.

## Step 3 — Corpus analysis

**Verdict: deterministic.**

Token counts come from a tokenizer. Statistics (mean, median, std) are arithmetic. The decision about whether chunking is needed is a threshold check: if fewer than 90% of documents fit in 512 tokens, recommend chunking. The suggested chunk size is the median of fitting documents, capped at 85% of the threshold.

These are formulas, not judgments. An LLM here would be slower and would introduce variability into a step where reproducibility is valuable for compliance and audit purposes.

## Step 4 — Model recommendation

**Verdict: this is where the LLM earns its place.**

Picking an embedding model from a candidate pool involves trade-offs that don't compose into a single scoring function. Specifically:

### Trade-off 1: chunking vs. long-context

If your corpus has p95 token length of 1500, you can either:
- Use a 512-token model with chunking (more retrieval calls per query, sometimes better recall on focused queries)
- Use an 8192-token model without chunking (fewer calls, potentially worse precision on highly specific queries)

The right choice depends on the user's downstream workload — interactive search vs. batch retrieval, latency tolerance, cost ceiling — none of which are inputs to the deterministic pipeline. A scoring function would have to encode an assumed user; an LLM can ask "what matters more for you here?" or weigh both options conversationally.

### Trade-off 2: candidate ranking under multiple constraints

Suppose the candidates are MiniLM, BGE-small, BGE-large, and text-embedding-3-large. The user is on CPU, has a privacy-sensitive corpus, and cares about latency. A simple scoring function might:

```
score = α * accuracy + β * (1/latency) + γ * privacy_compatible + δ * fits_in_context
```

But the weights α, β, γ, δ are exactly what the LLM is good at deciding. The user's actual objective is "good enough recall at acceptable latency for a privacy-sensitive workload." Translating that into weights is the judgment the LLM provides. Different users with different framings produce different weights, and the LLM does that translation per-run.

### Trade-off 3: freshness

Embedding models release frequently. A model that was state-of-the-art six months ago may have been superseded. A scoring function with hard-coded model names goes stale. The agent's `web_search` tool can pull current benchmarks and incorporate them into the recommendation. The LLM decides whether the question being asked is one where freshness matters.

### Trade-off 4: domain considerations

Generic-purpose embedding models underperform on domain-specific corpora. PubMedBERT outperforms BGE on medical text; CodeBERT outperforms it on source code. A scoring function would have to maintain a mapping of domain to specialized model. The agent reasons about whether the corpus's domain indicators justify a specialized model and which one fits.

### Trade-off 5: licensing and policy constraints

A user might be locked into a particular cloud provider, prohibited from using certain models, or required to use specific vendors. These constraints are best expressed in natural language and reasoned about. A scoring function would have to be re-coded for each new constraint.

## Step 5 — Fine-tuning recommendation

**Verdict: LLM, with a deterministic heuristic backstop.**

The deterministic heuristic flags fine-tuning as worth considering when type-token ratio is below 0.2 and top-term frequency is high — both signals of a domain-specific corpus. But whether fine-tuning is actually a good investment depends on:

- How much labeled data the user has or can produce
- What their downstream evaluation looks like
- Whether they have the engineering capacity to maintain a fine-tuned model
- Whether the gains from fine-tuning would justify the operational complexity

These are conversation topics, not inputs to a formula. The agent makes a recommendation and explains the reasoning; the user can push back if their actual situation differs from the agent's assumptions.

## Step 6 — Free-form explanation

**Verdict: LLM only.**

The `reasoning_explanation` field is plain English. A deterministic script can emit a templated string ("Recommended X because Y"), but a templated explanation is brittle: it can't adapt phrasing to the audience, can't combine multiple factors elegantly, and can't admit uncertainty when the case is borderline. The LLM produces a paragraph that meets the user where they are.

## Summary

The pipeline is roughly 80% deterministic by line count and 20% LLM-driven by responsibility. The LLM is concentrated where it adds the most value: synthesizing trade-offs that don't reduce to formulas. Everything else — measurement, statistics, threshold checks — runs in pure Python where it's fast, reproducible, and testable.

If you're considering a similar project structure for a different domain, the heuristic for the split is: **measurement is deterministic, judgment is LLM**. Tools should produce ground-truth facts; the agent should reason over them.
