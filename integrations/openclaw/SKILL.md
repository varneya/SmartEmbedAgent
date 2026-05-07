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

The user provides a path to a corpus file (`.txt`, `.csv`, `.json`) or directory of `.txt` files. Call the **exec** tool with this exact command (the shell expands `$SMARTEMBED_HOME` and `$CORPUS_PATH`):

```bash
cd "$SMARTEMBED_HOME" && CORPUS_PATH="<USER_PATH>" python3 main.py \
  --corpus_path "$CORPUS_PATH" \
  --config_path config/sample_config.json \
  --output_path /tmp/se_recommendation.json
```

Substitute `<USER_PATH>` with the absolute corpus path the user gave. The script writes both `/tmp/se_recommendation.json` and `/tmp/se_recommendation.md`. Run it once — do not retry on success.

If the user pasted raw text instead of a path, first write it to `/tmp/se_corpus.txt` using the `write` tool, then pass that path.

If the user says "fast" or "no LLM", or if Ollama isn't reachable, append `--no_llm` to skip the agentic step and use the deterministic heuristic instead.

## How to respond

Read `/tmp/se_recommendation.json` with the **read** tool. Reply with a chat-friendly summary under ~1500 characters:

- 🧭 **Top recommendation**: model name + 1-line rationale
- ⚙️ **Chunking**: yes/no; if yes, chunk size + overlap
- 💾 **Hardware fit**: 1 sentence
- 🎯 **Fine-tuning**: yes/no/why

Do not paste raw JSON. End with: *"Full Markdown report at `/tmp/se_recommendation.md` — want me to read it back?"*

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
