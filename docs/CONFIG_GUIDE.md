# Configuration Guide

SmartEmbedAgent reads a single JSON config file. The file is validated against `config/config_schema.json` at startup; any errors are reported with the exact field path, and the run aborts. See `config/sample_config.json` for a worked example.

The config has four top-level groups: `pii_settings`, `model_preferences`, `hardware_constraints`, and `agent_settings`. Each is described below.

## `pii_settings`

Controls how PII is detected and redacted from the corpus before analysis.

| Field | Type | Description |
|---|---|---|
| `custom_redaction_list` | array of strings | Strings or patterns to force-redact regardless of detection confidence. Use this for internal codenames, project names, customer IDs, and other terms unique to the user's organization that the public NER model has no way to recognize. |
| `whitelist` | array of strings | Strings exempt from redaction even if NER or regex flags them. Critical for company names, product names, and policy terminology that NER often misclassifies. Whitelist takes precedence over the redaction list — if a string appears in both, it survives. |
| `redaction_aggressiveness` | enum: `low` / `medium` / `high` | `low` runs regex only (emails, phones, SSNs, credit cards, IPs). `medium` adds NER for PERSON / LOCATION / ORGANIZATION. `high` additionally redacts MISC entities and applies stricter phone/email matching. |
| `ner_model_choice` | string | Hugging Face NER model identifier. Default `dslim/bert-base-NER`. Any compatible token-classification model works. |

## `model_preferences`

User constraints on the recommended embedding model.

| Field | Type | Description |
|---|---|---|
| `max_model_size_gb` | number | Maximum allowed model size on disk in GB. Models above this are filtered from the candidate pool. |
| `preferred_model_families` | array of strings | Ordered list of preferred providers, e.g. `sentence-transformers`, `BAAI`, `mistral`, `qwen`, `openai`. Earlier entries are preferred. The agent uses these as a tiebreaker when multiple models meet the hard constraints. |
| `prioritize_speed_over_accuracy` | boolean | If true, biases toward smaller, faster models. Useful for high-throughput retrieval pipelines where a small accuracy drop is acceptable. |
| `target_use_case` | enum: `semantic_search` / `classification` / `clustering` / `retrieval` / `reranking` | Primary downstream task. Used to pick task-tuned models when available (e.g. cross-encoder for reranking). |

## `hardware_constraints`

Caps that override the device profiler's findings. Use these when you want to reserve resources for other workloads on the same host.

| Field | Type | Description |
|---|---|---|
| `max_vram_usage_gb` | number | Maximum VRAM the agent may assume is available. Set to 0 to forbid GPU usage even when a GPU is detected. |
| `max_ram_usage_gb` | number | Maximum system RAM the agent may assume is available. |
| `allow_gpu` | boolean | If false, no GPU model is recommended even if a GPU is present. |
| `fallback_to_cpu` | boolean | If GPU is unavailable or insufficient, fall back to CPU-only models. Recommended `true` unless GPU is strictly required. |

## `agent_settings`

Runtime behavior of the orchestrator.

| Field | Type | Description |
|---|---|---|
| `enable_web_search` | boolean | Allow the agent to query the web for current best-practice and benchmark information. |
| `search_cache_days` | number | Search cache TTL in days. Cached entries older than this are re-fetched. Default 1. |
| `llm_model` | string | LLM driving the agent's reasoning step (e.g. `claude-sonnet-4-6`, `claude-opus-4-6`). |
| `verbose_logging` | boolean | If true, the agent prints its tool-calling trace to stdout. Recommended for debugging, off for production. |

## Validation

The validator can be run on its own to check a config before invoking the agent:

```bash
python -m src.config_validator config/my_config.json
```

Exit code 0 means valid; non-zero means errors were reported on stderr with the offending field path.

## Comment-style keys

Any key starting with `_` (e.g. `_comment`) is treated as documentation and ignored by the validator. Use these freely to annotate your config files without breaking validation.

## How nested config maps to module-level APIs

The nested config groups are flattened by `config_validator.extract_pii_config()` and `extract_agent_settings()` before being passed to individual modules. This keeps the user-facing config stable as the internal modules evolve. Specifically:

- `pii_settings.custom_redaction_list` → `pii_remover.remove_pii(config={"redaction_list": ...})`
- `pii_settings.whitelist` → `pii_remover.remove_pii(config={"whitelist": ...})`
- `pii_settings.redaction_aggressiveness` → `pii_remover.remove_pii(config={"entity_types": [...]})`
- `agent_settings.search_cache_days` → `build_agent(cache_ttl_seconds=days*86400)`
