# Contributing to SmartEmbedAgent

Thanks for your interest in improving SmartEmbedAgent. This document covers how to set up a development environment, the standards we hold contributions to, and how to add new tools to the agent.

## Development setup

```bash
git clone https://github.com/<your-username>/SmartEmbedAgent.git
cd SmartEmbedAgent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
make test
```

The full test suite runs in under two seconds with no external API key. If you have an `ANTHROPIC_API_KEY` set, you can also run the LLM-driven path end to end with `make run`.

## Adding a new tool to the agent

The agent's tools live in `src/agent_orchestrator.py`. To add a new tool:

1. **Write a deterministic implementation** as a new module under `src/`, with its own unit tests under `tests/`. Tools should be testable in isolation, without the LLM in the loop.
2. **Wrap the implementation as a `StructuredTool`** in `agent_orchestrator.py`. Provide a clear name, a one-line description that tells the LLM when to call it, and a Pydantic input schema if the tool accepts user-supplied arguments.
3. **Update the system prompt** to mention the new tool in the workflow section. The LLM uses the prompt to learn when each tool is appropriate.
4. **Add the tool to the deterministic fallback** (`run_pipeline_no_llm`) if it produces output that the recommendation depends on.
5. **Add an integration test** that verifies the tool fires in the expected position of the pipeline.

The bar for new tools: each tool should have a single, clearly-named purpose. Tools that overlap with existing tools usually indicate that the existing tool should be extended instead.

## Testing requirements

- All new code must include unit tests. Aim for behavior-level assertions (the function returns the right shape, errors are raised on bad inputs) rather than implementation-level mocking.
- Integration tests for changes that touch `main.py` or the orchestrator should run with `--no_llm` so they pass in CI.
- The full suite must stay green: `make test` with zero failures.

## Code style

- Type hints on all public functions.
- Docstrings on all public functions and classes. NumPy or Google style; we are not strict about which.
- Format with `black` (line length 100). Lint with `ruff`.
- Imports sorted by `isort` conventions: stdlib, third-party, local, separated by blank lines.

```bash
black src tests main.py
ruff check src tests main.py
```

## Pull request process

1. Fork the repo and create a feature branch named after the change (e.g. `feature/add-cohere-models` or `fix/regex-phone-overmatch`).
2. Commit early and often. Squash before opening the PR if your history is messy.
3. Open a PR using the template in `.github/PULL_REQUEST_TEMPLATE.md`. Describe what changes, why, and how it was tested.
4. CI must be green before merge.
5. At least one approving review from a maintainer is required.

## Reporting bugs

Use the GitHub issue tracker. Include the corpus characteristics (size, domain), the user config, the command you ran, and the actual vs. expected behavior. Anonymized snippets of the corpus are helpful when relevant — never share real PII.

## Code of Conduct

This project follows the standards in `CODE_OF_CONDUCT.md`. Be kind, be curious, and assume good intent.
