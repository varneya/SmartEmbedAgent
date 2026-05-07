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

Use this skill when the user asks for any of:
- "What embedding model should I use for X?"
- "Pick an embedding model for this corpus"
- "Recommend an embedding model for [a file path / a directory / pasted text]"
- "Should I chunk my documents? What chunk size?"
- "Should I fine-tune for my domain?"

## Inputs to gather from the user

1. **Corpus**: a path to a `.txt`, `.csv`, or `.json` file, or a directory of `.txt` files. If the user pastes raw text instead of a path, save it to `/tmp/se_corpus.txt` first using the file-write tool, then use that path.
2. **Config** (optional): path to a JSON config file. Default to `config/sample_config.json` if not specified.
3. **Use LLM reasoning?** (optional): default yes (uses Ollama). If the user says "fast" or "no LLM" or Ollama isn't running, append `--no_llm`.

## How to invoke

Use the `exec` tool with `workdir` set to `$SMARTEMBED_HOME`:

```
python3 main.py \
  --corpus_path <CORPUS_PATH> \
  --config_path <CONFIG_PATH> \
  --output_path /tmp/se_recommendation.json
```

Add `--no_llm` if the user requested heuristic-only mode, or if a previous LLM-mode run failed because Ollama wasn't reachable.

## How to respond

After the command completes, read `/tmp/se_recommendation.json` and summarize for the user. Keep the reply under ~1500 characters (chat-friendly):

- **Top recommendation**: model name + 1-line rationale
- **Chunking**: needed yes/no; if yes, chunk size and overlap
- **Hardware fit**: 1 sentence (e.g. "Fits comfortably on your M2 Pro's 32GB unified memory")
- **Fine-tuning**: 1 sentence (yes / not necessary / why)

Format as a clean message with light emoji headers (📊 ⚙️ 💾 🎯) — they render well in WhatsApp / Telegram / Signal. Do not paste raw JSON.

End with: "Full Markdown report at `/tmp/se_recommendation.md` — want me to read it back?"

## Failure modes

- **`SMARTEMBED_HOME` not set** → tell the user to export it and point to their cloned repo, e.g. `export SMARTEMBED_HOME=~/code/SmartEmbedAgent`.
- **Ollama not reachable / model not pulled** → re-invoke with `--no_llm` and note that the recommendation came from the deterministic heuristic; suggest `ollama serve` + `ollama pull hermes3:8b` for the agentic version.
- **Corpus file not found** → ask the user to confirm the path; offer to list the directory.
- **Non-zero exit code** → read stderr from the exec output and surface the relevant error line; do not retry blindly.
