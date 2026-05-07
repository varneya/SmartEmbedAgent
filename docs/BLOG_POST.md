# Building an Agentic Embedding Model Recommender: Why Reasoning Beats Rules

## The problem

I wanted to build a tool that picks the right embedding model for a given corpus. This sounds easy. It is not.

The naive approach is a scoring function: gather a few facts about the corpus and the host, plug them into a formula, output the model with the best score. I started there. By the third edge case I had a wall of `if/else` branches that none of my colleagues would want to maintain, and the recommendations were still wrong half the time. The problem isn't that the inputs are missing; it's that the *judgment* required to weigh those inputs doesn't compose into a formula.

Concretely: if a user has a 2,000-token document and is on CPU, do you tell them to upgrade to a long-context model or to chunk? The right answer depends on whether they care more about latency or precision, whether they're willing to pay for a hosted model, whether their downstream task tolerates lost cross-chunk context. None of those preferences are inputs to the deterministic pipeline. Encoding them as additional config flags pushes the user-facing complexity onto the user, which is exactly the wrong direction.

So I rebuilt it as an agent.

## What "agentic" means here

The pipeline still does the boring work in deterministic Python: profile the hardware with `psutil` and `torch.cuda`, run a layered regex + NER pass to strip PII, compute token statistics with a real tokenizer, decide whether chunking is structurally necessary based on a percentage-fit threshold. Those steps don't need an LLM and would be slower and less reproducible if they used one.

The LLM only enters the loop where judgment is required. Specifically:

- Picking from the candidate model pool given multiple soft constraints (privacy, latency, cost, domain freshness).
- Deciding whether chunking is the right answer or whether a long-context model is.
- Deciding whether to recommend fine-tuning given corpus shape and the user's context.
- Synthesizing a free-form explanation that the user can engage with.
- Knowing when to query the web for current benchmarks rather than trusting cached priors.

The agent has four tools: `device_profiler`, `pii_remover`, `corpus_analyzer`, and `web_search`. The first three return structured JSON; the agent calls them in order, then uses the gathered context plus the user's framing to make a recommendation. The web search tool is invoked only when the agent decides the answer to a freshness or domain question would change the recommendation.

This split — measurement is deterministic, judgment is LLM — turned out to be the design principle that made the project tractable.

## What deterministic scripts get wrong

The scoring-function approach fails in three ways.

**It encodes an assumed user.** Any score `α * accuracy + β * latency + γ * privacy` has weights baked in by the script's author. Different users have different objectives. The script can be made configurable by exposing those weights, but now the user has to know what their weights are, which is exactly the question they came to the tool to have answered.

**It goes stale.** Embedding models release frequently. A leaderboard from six months ago doesn't reflect the current state of the field. A scoring function with hard-coded model names rots; an agent that can search the web stays current.

**It can't explain itself.** A scoring function emits a ranked list. It doesn't say "I picked this over that because your corpus is medical and the hosted API would expose patient information that you'd then need a BAA for." Free-form explanation isn't a luxury — it's how the user verifies that the tool understood their actual problem.

## What I learned

A few things, in rough order of how surprised I was by them.

**The agent reads the structured tool outputs better than I expected it to.** I worried that passing JSON blobs back from tools would confuse the LLM. It didn't. Claude in particular is good at picking the relevant fields out of a 200-line tool response and reasoning over them. This let me skip a lot of post-processing I'd planned to write.

**Most of the work is still Python.** The LLM is concentrated at one synthesis step. By line count, the project is maybe 80% deterministic, 20% LLM-driven. The agent framing is more about responsibility than runtime — the LLM owns the parts that need judgment and nothing else.

**The deterministic fallback is more useful than I expected.** I built `run_pipeline_no_llm` initially as a CI convenience: I didn't want the test suite to require an API key. It turned out to be useful for debugging too — when the LLM produces a weird recommendation, I can run the deterministic path on the same input, compare, and figure out whether the LLM's deviation was reasoned or arbitrary.

**Tool order matters less than I thought.** I started with a strict prescribed order in the system prompt: profile, then redact, then analyze, then search. The agent mostly follows it but occasionally calls `corpus_analyzer` before `pii_remover` when the user's framing suggests it. As long as the data dependencies are honored (you can't analyze the cleaned corpus before cleaning it), the order is the agent's call.

**Caching the web search tool was non-negotiable.** Without a cache, every run that hit `web_search` would consume API credits and add latency. With a 24-hour file cache and SHA1-keyed lookups, repeat runs on similar corpora are free after the first one.

## Future directions

The current version is a CLI. A web UI would let non-technical users drop a corpus and see the recommendation without setting up a Python environment. A WhatsApp or Slack bot integration would make ad-hoc analysis more conversational. End-to-end retrieval evaluation against a real benchmark like MTEB would let the agent verify its recommendations rather than reasoning about them in the abstract.

The deeper interesting direction is automatic fine-tuning recipe generation. When the agent recommends fine-tuning, it could produce a complete recipe — training data format, contrastive loss choice, validation split, expected gain — that the user could run. That's the version I want to build next.

## Takeaway

Not every problem needs an agent, but problems that involve weighing user-specific trade-offs against ground-truth measurements tend to benefit from one. The shape that worked here was strict separation: pure-Python tools for facts, an LLM for judgment, a structured JSON output schema both paths share. If you're building something similar in a different domain, that's the structure I'd start from.

The full source is at [`https://github.com/<your-username>/SmartEmbedAgent`](https://github.com/). MIT-licensed.
