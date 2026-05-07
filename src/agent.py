"""
Compatibility shim — re-exports the public API from `agent_orchestrator`.

The orchestrator was renamed in Part Five to better describe its role.
Anything importing `build_agent` or `run_pipeline_no_llm` from `src.agent`
will continue to work; new code should import from
`src.agent_orchestrator` directly.
"""

from .agent_orchestrator import (  # noqa: F401
    AgentContext,
    FileCache,
    WebSearch,
    build_agent,
    get_context,
    reset_context,
    run_pipeline_no_llm,
    synthesize_heuristic_recommendation,
)
