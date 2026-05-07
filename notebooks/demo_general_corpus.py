"""
Notebook-style demo — general corpus.

Run as a script (`python notebooks/demo_general_corpus.py`) or convert to
.ipynb with jupytext if you want to step through cells. Each `# %%` block is
a separate cell.
"""

# %% [markdown]
# # SmartEmbedAgent demo: general corpus
#
# This notebook walks through the full pipeline on the included
# `data/sample_general.txt` sample. It uses the deterministic path so it
# runs without an API key.

# %%
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# %% [markdown]
# ## Step 1 — load corpus and config
# %%
corpus = (PROJECT_ROOT / "data" / "sample_general.txt").read_text()
config = json.loads((PROJECT_ROOT / "config" / "sample_config.json").read_text())
config = {k: v for k, v in config.items() if not k.startswith("_")}
print(f"Corpus: {len(corpus)} chars")
print(f"Config groups: {list(config.keys())}")

# %% [markdown]
# ## Step 2 — run the deterministic pipeline
# %%
from src.agent_orchestrator import run_pipeline_no_llm
from src.config_validator import extract_pii_config

pii_cfg = extract_pii_config(config)
result = run_pipeline_no_llm(corpus, user_config=pii_cfg)
print(json.dumps(result, indent=2))

# %% [markdown]
# ## Step 3 — inspect the device profile
# %%
from src.agent_orchestrator import get_context

ctx = get_context()
print(json.dumps(ctx.device_specs, indent=2))

# %% [markdown]
# ## Step 4 — inspect the PII redaction report
# %%
print(json.dumps(ctx.pii_report, indent=2))

# %% [markdown]
# ## Step 5 — inspect the corpus analysis
# %%
print(json.dumps(ctx.corpus_analysis, indent=2))

# %% [markdown]
# ## Step 6 — switch to the agentic path
#
# Set `ANTHROPIC_API_KEY` and run:
#
# ```python
# from src.agent_orchestrator import build_agent
# agent = build_agent(corpus=corpus, user_config=pii_cfg)
# response = agent.invoke({"input": "Analyze and recommend a model."})
# print(response["output"])
# ```
