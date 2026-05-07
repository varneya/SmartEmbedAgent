---
name: recommend-embedding-model
description: Recommend the optimal embedding model for a corpus given hardware, privacy, and corpus shape. Returns ranked models, chunking strategy, and fine-tuning advice. Runs SmartEmbedAgent locally via Ollama on Apple Silicon.
homepage: https://github.com/varneya/SmartEmbedAgent
user-invocable: true
metadata:
  {
    "openclaw":
      {
        "emoji": "🧭",
        "requires":
          {
            "bins": ["python3", "ollama"],
            "env": ["SMARTEMBED_HOME"],
          },
      },
  }
---

# Recommend an embedding model

## When to use this skill

INVOKE this skill when the user says any of:
- "Recommend an embedding model …"
- "What embedding model should I use for …"
- "Pick an embedding model for …"
- "Should I chunk my documents?" (a corpus is involved)
- "Should I fine-tune for my domain?" (a corpus is involved)

## How to invoke (do this; do NOT search the filesystem first)

The user provides a path to a corpus file (`.txt`, `.csv`, `.json`) or directory of `.txt` files. Call the **exec** tool with this exact command (the shell expands the variables; the python search picks the first interpreter with the project's deps):

```bash
cd "$SMARTEMBED_HOME" && \
PY="$SMARTEMBED_HOME/venv/bin/python3"; \
[ -x "$PY" ] || PY="$HOME/miniforge3/bin/python3"; \
[ -x "$PY" ] || PY="$(command -v python3)"; \
"$PY" main.py \
  --corpus_path "<USER_PATH>" \
  --config_path config/sample_config.json \
  --output_path /tmp/se_recommendation.json
```

Substitute `<USER_PATH>` with the absolute corpus path the user gave. The script writes both `/tmp/se_recommendation.json` and `/tmp/se_recommendation.md`. Run it once — do not retry on success.

If you see `ModuleNotFoundError: psutil` or similar, the python interpreter doesn't have the SmartEmbedAgent deps. Tell the user to install them: `cd "$SMARTEMBED_HOME" && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`. Do NOT attempt to install packages yourself.

If the user pasted raw text instead of a path, first write it to `/tmp/se_corpus.txt` using the `write` tool, then pass that path.

If the user says "fast" or "no LLM", or if Ollama isn't reachable, append `--no_llm` to skip the agentic step and use the deterministic heuristic instead.

## How to respond — STRICT TWO-STEP

### Step A (MANDATORY): read the actual output

After `exec` completes, you MUST call the **read** tool on `/tmp/se_recommendation.json` to load the actual recommendation. Do NOT skip this step. Do NOT summarize from the exec stdout — it only contains log lines, not the recommendation. Do NOT invent model names from memory.

### Step B: summarize using ONLY the JSON you just read

The JSON has these exact keys you must surface verbatim:
- `recommended_models[0].name` → use this for the top recommendation. Do not substitute a different model.
- `recommended_models[0].rationale` → use as the rationale.
- `chunking_strategy.needed` (boolean) → say "yes" or "no" exactly per this value.
- `chunking_strategy.chunk_size_tokens` and `chunking_strategy.overlap_tokens` → use these exact integers (in tokens, not characters).
- `hardware_fit_analysis` → quote or paraphrase this string verbatim. Do NOT replace with a generic "standard hardware" sentence.
- `fine_tuning_advice` → quote or paraphrase this string verbatim.

Format:
```
🧭 Top: <recommended_models[0].name> — <one-line rationale from JSON>
⚙️ Chunking: <yes/no>. <if yes: chunk_size_tokens tokens, overlap_tokens tokens overlap>
💾 Hardware: <hardware_fit_analysis>
🎯 Fine-tuning: <fine_tuning_advice>

Full Markdown report at /tmp/se_recommendation.md — want me to read it back?
```

Do not paste raw JSON. Keep under ~1500 chars.

## Failure handling

| Symptom | Action |
|---|---|
| `SMARTEMBED_HOME: unbound` | Tell user: `export SMARTEMBED_HOME=~/path/to/SmartEmbedAgent` and reopen OpenClaw. |
| Ollama / connection / model errors in stderr | Re-run with `--no_llm`. Mention the result is from the deterministic heuristic. |
| Corpus file not found | Confirm the path with the user; offer to list the parent directory. |
| Non-zero exit | Surface the relevant stderr line. Do not retry blindly. |

## Important

- The script is fast (~1–5 seconds heuristic, ~10–30 seconds with LLM). Do not spawn it as a background process.
- The skill IS the SmartEmbedAgent repo — do not go looking for it elsewhere on the filesystem; it lives at `$SMARTEMBED_HOME` (already verified at skill-load time).
