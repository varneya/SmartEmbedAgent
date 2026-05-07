"""
SmartEmbedAgent — LangChain Agent Orchestrator
==============================================

Wraps the device profiler, PII remover, corpus analyzer, and a cached web
search tool as LangChain `Tool`s. Drives a local-LLM-backed agent (Ollama by
default) through a deterministic prescribed workflow:

    1. device_profiler        — hardware constraints
    2. pii_remover            — sanitize the corpus
    3. corpus_analyzer        — token stats, chunking decision
    4. (optional) web_search  — current best-practice guidelines, cached
    5. synthesize a structured recommendation

The agent's *measurements* are deterministic — the tools compute facts. The
agent's *judgment* (which model to pick, whether to fine-tune) is LLM-driven.
This split is deliberate: it keeps the recommendation explainable while
letting the LLM contribute reasoning over trade-offs.

Decision points
---------------
Deterministic (no LLM):
    - Hardware enumeration (RAM, GPU, compute_device)
    - Regex PII redaction
    - Token statistics, percentage-fit per context window
    - The boolean `chunking_needed` flag and the suggested chunk size formula

LLM-driven:
    - Whether to upgrade to a long-context model vs. chunking
    - Which embedding model to pick from the candidate pool, given the hardware
      / privacy / domain trade-offs
    - Whether to recommend fine-tuning
    - Free-form `reasoning_explanation`

Caching
-------
Web search results are cached to a JSON file (default
`.cache/agent_search.json`) keyed by query string, with a configurable TTL
(default 24 hours). Cache hits never hit the network.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Local module imports — these are deterministic, no LLM involved.
from .corpus_analyzer import analyze_corpus
from .device_profiler import get_hardware_specs
from .pii_remover import remove_pii


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("agent_orchestrator")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[agent] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Shared agent context
# ---------------------------------------------------------------------------
@dataclass
class AgentContext:
    """Holds inputs and intermediate results that tools share. Passing big
    payloads (the corpus) through the LLM prompt would be wasteful and
    error-prone, so tools read from this shared object instead."""

    raw_corpus: str = ""
    user_config: Dict[str, Any] = field(default_factory=dict)
    tokenizer_name: str = "bert-base-uncased"

    # Filled in as tools run:
    device_specs: Optional[Dict[str, Any]] = None
    cleaned_corpus: Optional[str] = None
    pii_report: Optional[Dict[str, Any]] = None
    corpus_analysis: Optional[Dict[str, Any]] = None
    search_results: Dict[str, Any] = field(default_factory=dict)


# Module-level singleton; reset_context() initializes it for each run.
_CONTEXT = AgentContext()


def reset_context(
    corpus: str,
    user_config: Optional[Dict[str, Any]] = None,
    tokenizer_name: str = "bert-base-uncased",
) -> None:
    global _CONTEXT
    _CONTEXT = AgentContext(
        raw_corpus=corpus,
        user_config=user_config or {},
        tokenizer_name=tokenizer_name,
    )


def get_context() -> AgentContext:
    return _CONTEXT


# ---------------------------------------------------------------------------
# File-based cache for web search
# ---------------------------------------------------------------------------
class FileCache:
    """A trivial JSON-on-disk cache with TTL. Threadsafe enough for single-
    process use; not intended for concurrent writers."""

    def __init__(self, path: Path, ttl_seconds: int = 24 * 3600) -> None:
        self.path = path
        self.ttl_seconds = ttl_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}")

    def _load(self) -> Dict[str, Any]:
        try:
            return json.loads(self.path.read_text() or "{}")
        except json.JSONDecodeError:
            return {}

    def _save(self, data: Dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data, indent=2))

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha1(query.strip().lower().encode("utf-8")).hexdigest()[:16]

    def get(self, query: str) -> Optional[Any]:
        key = self._key(query)
        data = self._load()
        entry = data.get(key)
        if not entry:
            return None
        if time.time() - entry.get("ts", 0) > self.ttl_seconds:
            return None  # expired
        return entry.get("value")

    def set(self, query: str, value: Any) -> None:
        key = self._key(query)
        data = self._load()
        data[key] = {"ts": time.time(), "query": query, "value": value}
        self._save(data)


# ---------------------------------------------------------------------------
# Web search tool — pluggable backend with file cache
# ---------------------------------------------------------------------------
class WebSearch:
    """Wraps a search backend with the file cache. The backend is pluggable:

    - If `backend` is given, use it.
    - Else try `langchain_community.tools.DuckDuckGoSearchRun` if installed.
    - Else fall back to a stub that returns a clear "search unavailable"
      payload, so the agent never crashes if no backend is configured.
    """

    def __init__(
        self,
        cache: FileCache,
        backend: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.cache = cache
        self.backend = backend or self._default_backend()

    @staticmethod
    def _default_backend() -> Callable[[str], str]:
        try:
            from langchain_community.tools import DuckDuckGoSearchRun

            search = DuckDuckGoSearchRun()
            return lambda q: search.run(q)
        except Exception:
            def stub(q: str) -> str:
                return (
                    "[web_search] No search backend configured. "
                    "Install langchain-community or pass a backend "
                    "callable to WebSearch."
                )
            return stub

    def query(self, q: str) -> Dict[str, Any]:
        cached = self.cache.get(q)
        if cached is not None:
            logger.info("Cache hit for query: %r", q)
            return {"cached": True, "query": q, "result": cached}

        logger.info("Cache miss; running web search: %r", q)
        try:
            result = self.backend(q)
        except Exception as e:
            logger.warning("Web search failed: %s", e)
            result = f"[web_search] Search backend error: {e}"

        self.cache.set(q, result)
        return {"cached": False, "query": q, "result": result}


# ---------------------------------------------------------------------------
# Tool input schemas (Pydantic) — gives the LLM a clear contract per tool
# ---------------------------------------------------------------------------
def _build_input_schemas():
    """Build pydantic schemas lazily, so the module imports cleanly even if
    pydantic v1/v2 differs across environments."""
    try:
        from pydantic import BaseModel, Field
    except ImportError:
        from pydantic.v1 import BaseModel, Field  # type: ignore

    class EmptyInput(BaseModel):
        """Tools that read from shared context don't need user-supplied input."""
        pass

    class WebSearchInput(BaseModel):
        query: str = Field(
            ...,
            description=(
                "Search query, e.g. 'best embedding models 2026 enterprise data' "
                "or 'PII redaction GDPR guidance 2026'."
            ),
        )

    return EmptyInput, WebSearchInput


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def _tool_device_profiler(_: str = "") -> str:
    """Inspect host hardware and write specs into shared context."""
    specs = get_hardware_specs()
    _CONTEXT.device_specs = specs
    return json.dumps(specs, indent=2)


def _tool_pii_remover(_: str = "") -> str:
    """Run PII removal on the corpus held in shared context. Stores the
    cleaned corpus and report for downstream tools."""
    if not _CONTEXT.raw_corpus:
        return json.dumps({"error": "No corpus in context."})

    cleaned, report = remove_pii(_CONTEXT.raw_corpus, _CONTEXT.user_config)
    _CONTEXT.cleaned_corpus = cleaned
    _CONTEXT.pii_report = report
    preview = cleaned[:300] + ("..." if len(cleaned) > 300 else "")
    return json.dumps(
        {
            "summary": report["summary"],
            "total_redactions": report["total"],
            "cleaned_preview": preview,
        },
        indent=2,
    )


def _tool_corpus_analyzer(_: str = "") -> str:
    """Run corpus analysis on the cleaned corpus (preferred) or raw corpus.
    Stores the analysis in context."""
    text = _CONTEXT.cleaned_corpus or _CONTEXT.raw_corpus
    if not text:
        return json.dumps({"error": "No corpus available."})

    analysis = analyze_corpus(text, tokenizer_name=_CONTEXT.tokenizer_name)
    _CONTEXT.corpus_analysis = analysis.to_dict()
    return json.dumps(analysis.to_dict(), indent=2)


def _make_search_tool(web_search: WebSearch) -> Callable[[str], str]:
    def _tool_search(query: str) -> str:
        result = web_search.query(query)
        # Stash so the final synthesis step can reference what was searched.
        _CONTEXT.search_results[result["query"]] = result["result"]
        return json.dumps(result, indent=2)

    return _tool_search


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are SmartEmbedAgent, an expert assistant for selecting embedding models.

Your role is to recommend the optimal embedding model and chunking strategy
for a user's corpus, weighing hardware constraints, privacy considerations,
and corpus characteristics.

You have these tools:

  1. device_profiler   — return host hardware (RAM, GPU, compute_device).
  2. pii_remover       — sanitize the corpus using regex + NER, honoring the
                         user's whitelist and custom redaction list. ALWAYS
                         run before analyzing the corpus, so token counts
                         reflect the data you'll embed.
  3. corpus_analyzer   — token stats, context-window fit, chunking decision,
                         vocabulary metrics, domain indicators.
  4. web_search        — query the web for current best practices on PII
                         handling or embedding benchmarks. Results are
                         cached; reuse cached answers when possible.

Workflow (in this order):
  Step 1: Call device_profiler.
  Step 2: Call pii_remover.
  Step 3: Call corpus_analyzer.
  Step 4: Optionally call web_search for guideline/benchmark questions.
  Step 5: Reason about candidate models. Then output a final answer as a
          single JSON object with these exact fields:

  {
    "recommended_models": [
      {"name": "...", "rank": 1, "rationale": "..."},
      ...
    ],
    "reasoning_explanation": "Free-form explanation of how hardware, PII volume, and corpus shape drove the choice.",
    "chunking_strategy": {
      "needed": true|false,
      "chunk_size_tokens": int|null,
      "overlap_tokens": int|null,
      "rationale": "..."
    },
    "fine_tuning_advice": "Whether to fine-tune, and why or why not.",
    "hardware_fit_analysis": "How the recommended top model fits the user's hardware."
  }

Reasoning principles:
  - All candidate models are open-source and run locally. Pick the one that
    best fits the corpus and hardware — don't worry about hosted-API trade-offs.
  - Match the model's context window to the corpus's p95 token length, OR
    chunk. Don't default to the largest model.
  - On CPU-only hardware (compute_device = "cpu"), prefer small models
    (MiniLM, BGE-small) over large transformers. On accelerated hardware
    (compute_device in {"cuda", "rocm", "mps"}) larger models like
    BGE-large, BGE-M3, or mxbai-embed-large are viable; on Apple Silicon
    (mps) the relevant memory budget is unified RAM, not a separate VRAM
    figure.
  - Type-token ratio + domain indicators are signals for whether fine-tuning
    on a domain corpus would help.

Output format:
  - Output ONLY the JSON object. No preamble. No markdown code fences.
  - Do not omit any of the 5 top-level fields.
  - Do not include explanatory text before or after the JSON."""


# ---------------------------------------------------------------------------
# LLM factory — local Ollama by default; swap point for llama.cpp / MLX
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_MODEL = "hermes3:8b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def _ollama_pull_hint(model: str, base_url: str) -> str:
    return (
        f"Ollama model '{model}' is not available at {base_url}. "
        f"Start Ollama (`brew install ollama && ollama serve`) and run "
        f"`ollama pull {model}` to fetch it. "
        "See README 'Quickstart on Apple Silicon' for full setup."
    )


def build_llm(model: Optional[str] = None, base_url: Optional[str] = None) -> Any:
    """Construct the default local LLM (ChatOllama).

    Resolution order for parameters: explicit args → env vars
    (`SMARTEMBED_LLM_MODEL`, `OLLAMA_BASE_URL`) → built-in defaults.

    Verifies that the Ollama server is reachable and that the requested model
    has been pulled. Raises RuntimeError with the exact `ollama pull` command
    if not — callers (e.g. main.py) catch this and fall back to the
    deterministic heuristic so the user still gets a recommendation.
    """
    model = model or os.getenv("SMARTEMBED_LLM_MODEL", DEFAULT_OLLAMA_MODEL)
    base_url = base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)

    try:
        from langchain_ollama import ChatOllama
    except ImportError as e:
        raise RuntimeError(
            "langchain-ollama not installed. Run `pip install langchain-ollama` "
            "or pass `llm=` explicitly to build_agent()."
        ) from e

    # Verify model availability via Ollama's HTTP API. Avoids a confusing
    # langchain runtime error later if the server is down or the model is
    # not pulled. urllib is in the stdlib so no extra dep needed.
    import urllib.request
    import urllib.error

    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RuntimeError(
            f"Ollama server not reachable at {base_url}: {e}. "
            f"Start it with `ollama serve` and run `ollama pull {model}`."
        ) from e

    available = {m.get("name", "") for m in tags.get("models", [])}
    # Ollama tags include the implicit ":latest" suffix when missing.
    if model not in available and f"{model}:latest" not in available:
        raise RuntimeError(_ollama_pull_hint(model, base_url))

    return ChatOllama(model=model, temperature=0, base_url=base_url)


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------
def build_agent(
    corpus: str,
    user_config: Optional[Dict[str, Any]] = None,
    llm: Any = None,
    cache_path: Optional[Path] = None,
    cache_ttl_seconds: int = 24 * 3600,
    tokenizer_name: str = "bert-base-uncased",
    verbose: bool = True,
):
    """Build a LangChain AgentExecutor. Returns the executor; callers invoke
    it with `executor.invoke({"input": "..."})`.

    Parameters
    ----------
    corpus : str
        The corpus to analyze. Stored in shared context.
    user_config : dict, optional
        PII config (whitelist, redaction_list, ner_model).
    llm : BaseLanguageModel, optional
        Override the default. Default is `build_llm()`, which returns a
        ChatOllama instance pointed at a local Ollama server (Apple Silicon
        Metal-accelerated).
    cache_path : Path, optional
        Where to store the web-search cache. Default `.cache/agent_search.json`.
    cache_ttl_seconds : int
        Cache entry TTL in seconds. Default 24h.
    tokenizer_name : str
        Tokenizer to use during corpus analysis.
    """
    # Initialize shared context for this run.
    reset_context(corpus, user_config, tokenizer_name)

    # Local imports so the module can still be imported in environments
    # without LangChain installed (the deterministic fallback below works).
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain.tools import StructuredTool
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    if llm is None:
        llm = build_llm()

    # Cache + web search.
    cache_path = cache_path or Path(".cache") / "agent_search.json"
    web_search = WebSearch(FileCache(cache_path, ttl_seconds=cache_ttl_seconds))

    EmptyInput, WebSearchInput = _build_input_schemas()
    search_func = _make_search_tool(web_search)

    tools = [
        StructuredTool.from_function(
            name="device_profiler",
            description=(
                "Inspect the host machine. Returns JSON with total_ram_gb, "
                "available_ram_gb, ram_percentage_used, gpu_available, "
                "gpu_name, gpu_memory_gb, compute_device. Call this first."
            ),
            func=_tool_device_profiler,
            args_schema=EmptyInput,
        ),
        StructuredTool.from_function(
            name="pii_remover",
            description=(
                "Sanitize the user's corpus using regex + NER and honoring "
                "their whitelist + custom redaction list. Returns a redaction "
                "summary. Call AFTER device_profiler and BEFORE corpus_analyzer."
            ),
            func=_tool_pii_remover,
            args_schema=EmptyInput,
        ),
        StructuredTool.from_function(
            name="corpus_analyzer",
            description=(
                "Analyze the cleaned corpus. Returns doc_count, token_stats, "
                "context_window_recommendations at 512/1024/2048/4096/8192 "
                "tokens, chunking_needed flag, suggested_chunk_size, "
                "suggested_overlap, vocabulary_metrics (TTR), and "
                "domain_indicators. Call AFTER pii_remover."
            ),
            func=_tool_corpus_analyzer,
            args_schema=EmptyInput,
        ),
        StructuredTool.from_function(
            name="web_search",
            description=(
                "Search the web for current PII guidelines or embedding model "
                "benchmarks. Results are cached for 24 hours. Use sparingly — "
                "only for questions whose answer changes over time."
            ),
            func=search_func,
            args_schema=WebSearchInput,
        ),
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=verbose)


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------
def run_pipeline_no_llm(
    corpus: str,
    user_config: Optional[Dict[str, Any]] = None,
    tokenizer_name: str = "bert-base-uncased",
) -> Dict[str, Any]:
    """Run the deterministic portion of the pipeline (no LLM, no agent
    reasoning) and produce a heuristic recommendation. Useful for tests,
    CI, and environments without an Anthropic API key.

    The output schema matches what the LLM agent returns, so callers can
    treat the two interchangeably.
    """
    reset_context(corpus, user_config, tokenizer_name)
    _tool_device_profiler()
    _tool_pii_remover()
    _tool_corpus_analyzer()

    return synthesize_heuristic_recommendation(_CONTEXT)


def synthesize_heuristic_recommendation(ctx: AgentContext) -> Dict[str, Any]:
    """Produce a structured recommendation from the gathered context using
    deterministic rules. The same schema the LLM agent emits, so downstream
    consumers don't need to branch on which path produced it."""
    specs = ctx.device_specs or {}
    analysis = ctx.corpus_analysis or {}
    pii = ctx.pii_report or {"summary": {}, "total": 0}

    has_gpu = bool(specs.get("gpu_available"))
    gpu_mem = float(specs.get("gpu_memory_gb") or 0.0)
    ram = float(specs.get("total_ram_gb") or 0.0)
    pii_count = int(pii.get("total", 0))
    p95_tokens = int(analysis.get("token_stats", {}).get("max", 0))
    chunking_needed = bool(analysis.get("chunking_needed"))

    # Candidate model pool — open-source only, all run locally on Apple Silicon
    # via sentence-transformers / transformers with MPS acceleration.
    candidates: List[Tuple[str, int, str, bool]] = [
        # (name, ctx_window, footprint, requires_gpu_for_speed)
        ("sentence-transformers/all-MiniLM-L6-v2", 256, "tiny (~80 MB)", False),
        ("sentence-transformers/all-mpnet-base-v2", 384, "medium (~420 MB)", False),
        ("BAAI/bge-small-en-v1.5", 512, "small (~130 MB)", False),
        ("BAAI/bge-large-en-v1.5", 512, "large (~1.3 GB)", True),
        ("intfloat/e5-large-v2", 512, "large (~1.3 GB)", True),
        ("nomic-ai/nomic-embed-text-v1.5", 8192, "medium (~550 MB)", False),
        ("BAAI/bge-m3", 8192, "large (~2.3 GB)", True),
        ("mixedbread-ai/mxbai-embed-large-v1", 512, "large (~1.3 GB)", True),
    ]

    # Privacy filter: every candidate is on-device / open-source, so this is a
    # no-op today — kept so the heuristic still records the privacy posture
    # and stays compatible with future hosted-model additions.
    privacy_sensitive = pii_count >= 5
    if privacy_sensitive:
        candidates = [c for c in candidates if "hosted" not in c[2]]

    # Hardware filter: drop GPU-hungry models only on truly compute-light hosts.
    # MPS (Apple Silicon Metal) counts as GPU acceleration, even though it
    # doesn't expose a discrete VRAM number — it shares unified memory.
    compute_device = str(specs.get("compute_device", "cpu"))
    has_accelerator = compute_device in {"cuda", "rocm", "mps"} and (
        compute_device == "mps" or has_gpu
    )
    sufficient_memory = (compute_device == "mps" and ram >= 16.0) or gpu_mem >= 4.0
    if not (has_accelerator and sufficient_memory):
        candidates = [c for c in candidates if not c[3]]

    # Score: prefer models whose context window is close to (but at least)
    # what the corpus needs without chunking; if chunking is OK, prefer
    # smaller models.
    target_window = 512 if chunking_needed else max(p95_tokens, 256)

    def score(c: Tuple[str, int, str, bool]) -> Tuple[int, int]:
        name, window, _, _ = c
        # Penalty for under-fit, milder penalty for over-fit.
        if window < target_window:
            return (1, target_window - window)
        return (0, window - target_window)

    ranked = sorted(candidates, key=score)[:3]

    recommended_models = [
        {
            "name": name,
            "rank": i + 1,
            "rationale": (
                f"Context window {window}, {footprint}. "
                + ("CPU-friendly. " if not gpu_required else "Best on GPU. ")
                + ("Open-source / on-device — privacy-preserving." if "hosted" not in footprint else "Hosted API.")
            ),
        }
        for i, (name, window, footprint, gpu_required) in enumerate(ranked)
    ]

    chunking_strategy = {
        "needed": chunking_needed,
        "chunk_size_tokens": analysis.get("suggested_chunk_size") if chunking_needed else None,
        "overlap_tokens": analysis.get("suggested_overlap") if chunking_needed else None,
        "rationale": (
            "Significant fraction of documents exceed 512 tokens; chunking "
            "preserves compact-model viability."
            if chunking_needed
            else "Documents fit comfortably within compact-model context windows."
        ),
    }

    # Fine-tuning heuristic: low TTR + high-frequency domain terms suggests
    # the corpus is domain-specific enough to benefit from adaptation.
    vocab = analysis.get("vocabulary_metrics", {})
    ttr = float(vocab.get("type_token_ratio") or 1.0)
    top_term_freq = (
        analysis.get("domain_indicators", [{}])[0].get("frequency", 0)
        if analysis.get("domain_indicators")
        else 0
    )
    if ttr < 0.2 and top_term_freq >= 20:
        fine_tuning_advice = (
            "Recommended. The corpus has low lexical diversity (TTR < 0.2) and "
            "concentrated domain terminology, both signals that domain "
            "adaptation via fine-tuning or contrastive training would improve "
            "retrieval quality."
        )
    else:
        fine_tuning_advice = (
            "Not necessary as a first pass. The corpus is lexically diverse "
            "enough that a strong general-purpose embedding model should "
            "perform well; revisit only if retrieval quality is poor."
        )

    if compute_device == "mps":
        accel_desc = f"Apple Silicon ({specs.get('gpu_name', 'Apple GPU')}) with {ram} GB unified memory. "
    elif has_gpu:
        accel_desc = f"GPU: {specs.get('gpu_name')} with {gpu_mem} GB VRAM. "
    else:
        accel_desc = "CPU-only. "

    hardware_fit_analysis = (
        f"Total RAM: {ram} GB. "
        + accel_desc
        + f"Top recommendation '{ranked[0][0]}' is {ranked[0][2]} — fits comfortably."
        if ranked
        else "No candidate model passed the hardware/privacy filters."
    )

    return {
        "recommended_models": recommended_models,
        "reasoning_explanation": (
            f"Detected {pii_count} PII redactions ({'privacy-sensitive' if privacy_sensitive else 'standard privacy'}). "
            f"Corpus has {analysis.get('doc_count', 0)} documents averaging "
            f"{analysis.get('token_stats', {}).get('mean', 0)} tokens. "
            f"{'Chunking required' if chunking_needed else 'No chunking required'}. "
            "Selected model balances corpus size, hardware, and privacy."
        ),
        "chunking_strategy": chunking_strategy,
        "fine_tuning_advice": fine_tuning_advice,
        "hardware_fit_analysis": hardware_fit_analysis,
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "Customer feedback report. Contact alice@example.com or 555-123-4567.\n\n"
        "We use embeddings for semantic search across our knowledge base.\n\n"
        + ("Lorem ipsum dolor sit amet consectetur adipiscing elit. " * 200)
    )
    out = run_pipeline_no_llm(sample, user_config={"whitelist": ["Acme"]})
    print(json.dumps(out, indent=2))
