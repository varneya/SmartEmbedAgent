"""
Notebook-style demo — medical corpus (privacy-sensitive).

Highlights how the agent's recommendation changes when the corpus contains
high-volume PII: hosted models are filtered out and the recommendation
defaults to on-device alternatives.
"""

# %%
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_orchestrator import run_pipeline_no_llm
from src.config_validator import extract_pii_config

# %% [markdown]
# ## Run on the medical corpus
# %%
corpus = (PROJECT_ROOT / "data" / "sample_medical.txt").read_text()
config = json.loads((PROJECT_ROOT / "config" / "sample_config.json").read_text())
config = {k: v for k, v in config.items() if not k.startswith("_")}

result = run_pipeline_no_llm(corpus, user_config=extract_pii_config(config))
print(json.dumps(result, indent=2))

# %% [markdown]
# ## Why this differs from the general corpus
#
# Note that hosted-API models (`text-embedding-3-small`, `text-embedding-3-large`)
# are absent from the recommendations. The pipeline filters them out when the
# PII redaction count exceeds a threshold, on the principle that
# privacy-sensitive corpora should not be sent to a hosted API by default.
#
# In the LLM-driven path, the agent could argue for an exception (e.g., if
# the user has a Business Associate Agreement in place), but the heuristic
# is conservative.
