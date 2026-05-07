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

The user provides a path to a corpus file (`.txt`, `.csv`, `.json`) or directory of `.txt` files. Call the **exec** tool with this exact command. It runs the recommender AND prints a chat-ready summary to stdout — so you do not need to read any output files afterwards.

```bash
cd "$SMARTEMBED_HOME" && \
PY="$SMARTEMBED_HOME/venv/bin/python3"; \
[ -x "$PY" ] || PY="$HOME/miniforge3/bin/python3"; \
[ -x "$PY" ] || PY="$(command -v python3)"; \
"$PY" main.py \
  --corpus_path "<USER_PATH>" \
  --config_path config/sample_config.json \
  --output_path /tmp/se_recommendation.json > /dev/null 2>&1 && \
"$PY" -c "
import json
d = json.load(open('/tmp/se_recommendation.json'))
top = d['recommended_models'][0]
cs = d['chunking_strategy']
chunk_line = f\"{cs['chunk_size_tokens']} tokens chunk size, {cs['overlap_tokens']} tokens overlap\" if cs['needed'] else 'not needed'
print(f\"🧭 Top: {top['name']} — {top['rationale']}\")
print(f\"⚙️ Chunking: {'yes' if cs['needed'] else 'no'}. {chunk_line}\")
print(f\"💾 Hardware: {d['hardware_fit_analysis']}\")
print(f\"🎯 Fine-tuning: {d['fine_tuning_advice']}\")
print()
print('Full Markdown report at /tmp/se_recommendation.md')
"
```

Substitute `<USER_PATH>` with the absolute corpus path the user gave. Run it once.

If the exec returns a non-zero exit and stderr mentions `ModuleNotFoundError: psutil` or similar, the python interpreter doesn't have the SmartEmbedAgent deps. Tell the user to install them: `cd "$SMARTEMBED_HOME" && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`. Do NOT attempt to install packages yourself.

If the user pasted raw text instead of a path, first write it to `/tmp/se_corpus.txt` using the `write` tool, then pass that path.

If the user says "fast" or "no LLM", or if Ollama isn't reachable, append `--no_llm` to skip the agentic step and use the deterministic heuristic instead.

## How to respond — RELAY EXEC STDOUT VERBATIM

The exec command above already prints the formatted summary (🧭 ⚙️ 💾 🎯 lines) on stdout. **Send that stdout to the user as-is.** Do NOT paraphrase. Do NOT substitute model names. Do NOT add a "standard hardware" sentence. Do NOT call read on /tmp/se_recommendation.json — the values you need are already in the exec stdout.

You may add ONE optional closing line: *"Want me to read back the full Markdown report?"*

Keep the entire reply under ~1500 chars (chat-friendly).

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
