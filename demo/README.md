# Demo & Sample Outputs

This folder shows what SmartEmbedAgent produces on representative inputs. Three artifacts are included:

- [`agent_reasoning_trace.md`](agent_reasoning_trace.md) — a verbatim trace of an LLM-driven agent run on a privacy-sensitive medical corpus, showing each tool call, the tool's JSON response, and the agent's reasoning between calls.
- [`sample_recommendation.json`](sample_recommendation.json) — the structured JSON output that the agent emits as its final answer, suitable for programmatic consumption.
- [`sample_recommendation.md`](sample_recommendation.md) — the human-readable Markdown report rendered alongside the JSON.

## How to regenerate

```bash
# Deterministic (no API key required) — produces JSON + MD in runs/
python main.py \
  --corpus_path data/sample_medical.txt \
  --config_path config/sample_config.json \
  --output_path runs/medical.json \
  --no_llm

# Full LLM-driven run (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
python main.py \
  --corpus_path data/sample_medical.txt \
  --config_path config/sample_config.json \
  --output_path runs/medical.json \
  --verbose
```

The reasoning trace in this folder was captured from a `--verbose` LLM run on `data/sample_medical.txt`. It is reproduced here as documentation; you can produce a fresh one any time by re-running the above command.

## What to look for

- **Tool call ordering**: profile → PII → analyze, with web search fired only when the agent judges that benchmark currency matters.
- **Reasoning between calls**: short paragraphs in the agent's voice that explain why it took the next action it took.
- **Structured final answer**: the agent's closing message is a single JSON object with the documented schema, not free-form text.
