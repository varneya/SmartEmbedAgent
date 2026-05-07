# Decision Points — LLM vs. Deterministic

SmartEmbedAgent splits work between the LLM (judgment) and Python (measurement). Knowing which decisions live where matters for two reasons: the deterministic decisions are testable and reproducible; the LLM-driven decisions are explainable but variable across runs.

## Deterministic (no LLM)

These run in pure Python and are fully reproducible across invocations:

| Decision | Where | Notes |
|---|---|---|
| RAM total / available / % used | `device_profiler.get_hardware_specs` | `psutil.virtual_memory()` — no model involvement. |
| GPU presence and memory | `device_profiler.get_hardware_specs` | NVIDIA via CUDA, AMD via ROCm, CPU fallback. |
| Regex PII detection | `pii_remover._regex_spans` | Email, phone, SSN, credit-card, IP. Always exact. |
| Whitelist exemption | `pii_remover._is_whitelisted` | Case-insensitive equality check. |
| Custom-list force-redaction | `pii_remover._custom_spans` | Literal escape, then regex. |
| Token counts | `corpus_analyzer._Tokenizer.count` | HF tokenizer / tiktoken / whitespace. Same input → same count. |
| Token statistics | `corpus_analyzer._token_stats` | Mean, median, min, max, std. |
| Context-window fit % | `corpus_analyzer._context_window_fit` | At 512/1024/2048/4096/8192. |
| `chunking_needed` flag | `corpus_analyzer._decide_chunking` | True if <90% of docs fit in 512 tokens. |
| Suggested chunk size | `corpus_analyzer._suggest_chunk_size` | Median-of-fitting-docs capped at 85% of threshold. |
| Suggested overlap | `corpus_analyzer._suggest_overlap` | 15% of chunk size with a 16-token floor. |
| TTR + domain indicators | `corpus_analyzer._vocabulary_metrics` and `_domain_indicators` | Frequency counts with stopword filter. |

The `run_pipeline_no_llm` heuristic also makes a few rule-based decisions that *could* be LLM-driven but are kept deterministic for testability:

| Decision | Rule |
|---|---|
| Privacy-sensitive flag | True if `pii_report.total >= 5`. Drops hosted models from candidate pool. |
| GPU-required filter | If no GPU or GPU memory < 4 GB, drop large transformer models. |
| Candidate ranking | Score = (under-fit penalty, distance-to-target). |
| Fine-tuning recommendation (heuristic path only) | Recommend when TTR < 0.2 and top-term frequency >= 20. |

## LLM-driven (in `build_agent`)

The LLM is responsible for all judgment that requires weighing trade-offs no formula captures cleanly:

| Decision | Why LLM |
|---|---|
| **Whether to upgrade to a long-context model vs. chunking** | Trade-off depends on the user's downstream workload, latency tolerance, and cost ceiling — none of which are inputs to the deterministic pipeline. |
| **Final model pick from the candidate pool** | The deterministic ranking is a starting point; the LLM can override based on freshness signals (recent benchmark releases via `web_search`), domain-specific reputation, or licensing concerns the heuristic doesn't model. |
| **Whether to recommend fine-tuning** | Beyond the TTR rule of thumb, the decision involves the user's data volume, label availability, and resource budget — all softer signals. |
| **`reasoning_explanation`** | Free-form synthesis of why the recommendation makes sense, in a tone the user can read. |
| **`hardware_fit_analysis`** | Conversational framing of what the deterministic specs mean for the chosen model. |
| **Whether to query the web** | The agent decides when guideline freshness or benchmark currency is worth the cache miss. |

## Why this split

Putting measurement in code means the same input always produces the same numbers — important for compliance review of PII redaction, and for users who want to compare runs over time. Putting synthesis in the LLM means the recommendation can incorporate context the user only explains in natural language ("we already pay for OpenAI", "we can't use anything not on the EU model list") without re-coding the heuristic.

The deterministic fallback (`run_pipeline_no_llm`) emits the same JSON schema, so callers can swap between paths without changing downstream code. In practice: use the LLM path in production, use the heuristic path in CI.
