# `agent_orchestrator` — Example Output

The orchestrator has two entry points that produce the same JSON schema:

- `build_agent(...).invoke({"input": "..."})` — full LLM-driven pipeline using ChatAnthropic
- `run_pipeline_no_llm(...)` — deterministic heuristic pipeline (no API key needed)

Both end with the same structured recommendation, so callers can swap between them without changing downstream code.

## Sample input

```text
Customer feedback report. Contact alice@example.com or 555-123-4567.

We use embeddings for semantic search across our knowledge base.

(Lorem ipsum filler repeated 200 times — pushes the corpus over 512 tokens
to trigger the chunking branch.)
```

User config:

```json
{ "whitelist": ["Acme"] }
```

## Output (`run_pipeline_no_llm`)

```json
{
  "recommended_models": [
    {
      "name": "BAAI/bge-small-en-v1.5",
      "rank": 1,
      "rationale": "Context window 512, small (~130 MB). CPU-friendly. Open-source / on-device — privacy-preserving."
    },
    {
      "name": "openai/text-embedding-3-small",
      "rank": 2,
      "rationale": "Context window 8191, hosted API. CPU-friendly. Hosted API."
    },
    {
      "name": "openai/text-embedding-3-large",
      "rank": 3,
      "rationale": "Context window 8191, hosted API. CPU-friendly. Hosted API."
    }
  ],
  "reasoning_explanation": "Detected 2 PII redactions (standard privacy). Corpus has 3 documents averaging 539 tokens. Chunking required. Selected model balances corpus size, hardware, and privacy.",
  "chunking_strategy": {
    "needed": true,
    "chunk_size_tokens": 384,
    "overlap_tokens": 58,
    "rationale": "Significant fraction of documents exceed 512 tokens; chunking preserves compact-model viability."
  },
  "fine_tuning_advice": "Recommended. The corpus has low lexical diversity (TTR < 0.2) and concentrated domain terminology, both signals that domain adaptation via fine-tuning or contrastive training would improve retrieval quality.",
  "hardware_fit_analysis": "Total RAM: 3.81 GB. CPU-only. Top recommendation 'BAAI/bge-small-en-v1.5' is small (~130 MB) — fits comfortably."
}
```

(Chunk size and overlap will vary based on tokenizer used and corpus token distribution. The values shown here are produced when `transformers` is installed; the whitespace fallback produces smaller numbers because each whitespace token is shorter than a subword token.)

## Output (LLM-driven `build_agent`)

When `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is set and the agent runs end-to-end, the LLM emits the same schema with a richer `reasoning_explanation` that weighs trade-offs the heuristic can't (recent benchmarks fetched via `web_search`, the user's domain, freshness of model releases, etc.).

## Test results

18 integration + cache tests pass:

```
test_short_corpus_has_required_fields                ... ok
test_chunking_strategy_shape                         ... ok
test_recommended_models_have_rank_and_rationale      ... ok
test_models_ranked_in_order                          ... ok
test_short_corpus_does_not_recommend_chunking        ... ok
test_long_corpus_recommends_chunking                 ... ok
test_pii_redactions_are_counted                      ... ok
test_user_config_whitelist_is_honored                ... ok
test_context_is_populated_after_run                  ... ok
test_reset_context_clears_state                      ... ok
test_miss_returns_none                               ... ok
test_hit_returns_stored_value                        ... ok
test_persists_across_instances                       ... ok
test_expiration                                      ... ok
test_keys_are_normalized                             ... ok
test_uses_cache_on_repeat_query                      ... ok
test_backend_failure_does_not_crash                  ... ok
test_output_is_json_serializable                     ... ok
```

Combined suite (parts 2–5): **95 tests, all passing.**
