# SmartEmbedAgent skill for OpenClaw

Use SmartEmbedAgent from any chat app — WhatsApp, Telegram, Signal, iMessage — by installing it as a skill in [OpenClaw](https://openclaw.ai).

## What this gives you

You message your OpenClaw bot:

> "Recommend an embedding model for `~/Downloads/customer_feedback.txt`"

OpenClaw runs SmartEmbedAgent locally on your Mac (Apple Silicon, Ollama-backed), reads the JSON recommendation, and replies with a tidy summary — top model, chunking advice, hardware fit, fine-tuning note.

No data leaves your machine. No API keys. No Twilio.

## Install

### 1. Install OpenClaw and SmartEmbedAgent
Follow each project's setup:
- OpenClaw: https://openclaw.ai
- SmartEmbedAgent: see this repo's [README.md](../../README.md) (Quickstart on Apple Silicon)

### 2. Point the skill at your SmartEmbedAgent install
```bash
export SMARTEMBED_HOME=/path/to/SmartEmbedAgent     # wherever you cloned this repo
```
Add it to your shell rc file (`~/.zshrc`) so OpenClaw inherits it.

### 3. Install the skill
Pick one:

**Option A: symlink (recommended — picks up future updates)**
```bash
mkdir -p ~/.openclaw/skills
ln -s "$SMARTEMBED_HOME/integrations/openclaw" \
      ~/.openclaw/skills/recommend-embedding-model
```

**Option B: copy**
```bash
mkdir -p ~/.openclaw/skills/recommend-embedding-model
cp "$SMARTEMBED_HOME/integrations/openclaw/SKILL.md" \
   ~/.openclaw/skills/recommend-embedding-model/SKILL.md
```

### 4. Try it
Start a chat with your OpenClaw bot and say:
> Recommend an embedding model for `data/sample_long.txt`

The skill should fire, run `python main.py`, and reply with a recommendation summary. If Ollama is running with `hermes3:8b` pulled, you'll get the agentic explanation; otherwise it falls back to the deterministic heuristic.

## What gets invoked

The skill (see [`SKILL.md`](SKILL.md)) tells OpenClaw to run:

```bash
python main.py \
  --corpus_path <user-supplied path> \
  --config_path config/sample_config.json \
  --output_path /tmp/se_recommendation.json
```

with `workdir=$SMARTEMBED_HOME`. The full Markdown report is also written to `/tmp/se_recommendation.md` in case the user wants the long version.

## Customizing

Edit `SKILL.md` to:
- Change the default config path
- Always pass `--no_llm` (skip Ollama, faster)
- Change the response format (e.g., remove emoji, tighten length)
- Add gating ("only respond if the message starts with `embed:`")

Skills are plain Markdown — no compilation, no restart needed.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `SMARTEMBED_HOME: unbound variable` | `echo 'export SMARTEMBED_HOME=...' >> ~/.zshrc && source ~/.zshrc` |
| `python: command not found` in OpenClaw exec | Use the venv's interpreter: edit SKILL.md to call `$SMARTEMBED_HOME/venv/bin/python` |
| Skill doesn't appear | Verify the symlink/copy is at `~/.openclaw/skills/recommend-embedding-model/SKILL.md` and restart OpenClaw |
| Ollama errors in the report | `ollama serve &` then `ollama pull hermes3:8b`, or pass `--no_llm` |
