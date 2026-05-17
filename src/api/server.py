"""
FastAPI HTTP front-end for SmartEmbedAgent.

Runs the deterministic recommender (or the agentic LLM path if Ollama is
reachable and the caller opts in) over corpora delivered as either:

  • A list of absolute paths already on disk (`POST /recommend`)
  • Multipart-uploaded files (`POST /recommend/upload`)
  • A pasted text blob (`POST /recommend` with `corpus_text`)

A minimal HTML form at `/` calls the upload endpoint and renders the
result inline using Tailwind via CDN — no build step.

The Python library, CLI, and OpenClaw skill remain the canonical entry
points; this is an additional transport for browser / programmatic use.
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Make the project root importable when running `python -m src.api.server`
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
    from fastapi.templating import Jinja2Templates
    from fastapi import Request
    from pydantic import BaseModel, Field
except ImportError as e:
    raise SystemExit(
        "FastAPI not installed. Install the optional API extras with:\n"
        "    pip install -r requirements-api.txt"
    ) from e

# Import the existing project guts — zero changes required to those modules.
import re

from main import load_corpus, render_markdown_report  # noqa: E402
from src.agent_orchestrator import build_agent, run_pipeline_no_llm  # noqa: E402


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Salvage JSON from imperfect LLM output. Small Ollama models often
    wrap structured output in markdown code fences ('```json ... ```'),
    add a preamble ('Here is the recommendation:'), or trail with prose
    after the closing brace. We try in order:

      1. Strict json.loads on the raw text.
      2. Strip ```json ... ``` or ``` ... ``` fences and retry.
      3. Find the first '{' and the last '}' and try the substring.

    Returns the parsed dict or None. None means we should fall back.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    candidates: List[str] = [text.strip()]

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1))

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(text[first:last + 1])

    for c in candidates:
        try:
            parsed = json.loads(c)
            if isinstance(parsed, dict):
                return parsed
        except (TypeError, json.JSONDecodeError):
            continue
    return None
from src.config_validator import (  # noqa: E402
    DEFAULT_SCHEMA_PATH,
    extract_model_preferences,
    extract_pii_config,
    validate_config,
)
from src.agent_orchestrator import SUPPORTED_TASKS, build_llm  # noqa: E402
from src.evaluator import evaluate_candidates  # noqa: E402
from src.agents import run_validated_recommendation  # noqa: E402


SAMPLE_CONFIG_PATH = PROJECT_ROOT / "config" / "sample_config.json"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


# ---------------------------------------------------------------------------
# Security: corpus path allowlist
# ---------------------------------------------------------------------------
# By default the server binds to 127.0.0.1, but if a user ever switches to
# HOST=0.0.0.0 (or tunnels via cloudflared / ngrok / Tailscale), the
# `corpus_paths` field on POST /recommend would otherwise let any caller
# read any file the server-process user can read — including ~/.ssh, OpenClaw
# session credentials, or arbitrary documents.
#
# The allowlist gates every corpus path through resolve()-then-prefix-check,
# which defeats `..` traversal AND symlink-escape (resolve() follows symlinks
# before we compare). Default roots cover the obvious "places people put
# corpora": $HOME, /tmp, /Volumes (mounted external drives on macOS), and
# the project's own data/ dir (so the bundled samples work out of the box).
#
# Override via env var: SMARTEMBED_ALLOWED_CORPUS_ROOTS=/path1:/path2:...
import os as _os


def _default_allowed_roots() -> List[Path]:
    return [
        Path.home().resolve(),
        Path("/tmp").resolve(),
        Path("/Volumes").resolve(),
        (PROJECT_ROOT / "data").resolve(),
    ]


def _allowed_roots() -> List[Path]:
    raw = _os.getenv("SMARTEMBED_ALLOWED_CORPUS_ROOTS")
    if raw:
        return [Path(p).expanduser().resolve() for p in raw.split(":") if p.strip()]
    return _default_allowed_roots()


async def _save_uploads_and_load_corpus(
    files: List[UploadFile],
    max_upload_bytes: Optional[int] = None,
) -> Tuple[str, int, Path]:
    """Shared upload handler used by every multipart endpoint
    (/recommend/upload, /evaluate/upload, /validated_recommend/upload).

    Streams each upload to a tempdir, enforces an aggregate size cap,
    and runs load_corpus over the saved files. Returns (corpus_text,
    n_files_saved, tmpdir_path) — the caller is responsible for
    cleaning up tmpdir in its own finally block (the corpus string is
    already extracted so the files on disk aren't needed after this).
    """
    if max_upload_bytes is None:
        max_upload_bytes = int(_os.getenv("SMARTEMBED_MAX_UPLOAD_MB", "100")) * 1024 * 1024

    tmpdir = Path(tempfile.mkdtemp(prefix="smartembed_api_"))
    saved_paths: List[Path] = []
    bytes_written = 0
    for f in files:
        if not f.filename:
            continue
        # Strip path components to avoid `..`-style escapes; keep extension.
        safe_name = Path(f.filename).name
        target = tmpdir / safe_name
        with target.open("wb") as out:
            while chunk := await f.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(f"Upload exceeded {max_upload_bytes // (1024*1024)} MB "
                                f"(SMARTEMBED_MAX_UPLOAD_MB). Aborted at "
                                f"{bytes_written // (1024*1024)} MB."),
                    )
                out.write(chunk)
        saved_paths.append(target)

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid files uploaded.")
    corpus = load_corpus(saved_paths)
    return corpus, len(saved_paths), tmpdir


def _cleanup_tempdir(tmpdir: Path) -> None:
    """Best-effort cleanup. Failures are non-fatal."""
    try:
        for child in tmpdir.iterdir():
            try:
                child.unlink(missing_ok=True)
            except OSError:
                pass
        tmpdir.rmdir()
    except OSError:
        pass


def _enforce_corpus_path_allowlist(p: Path) -> None:
    """Raise 403 if `p` is not under any allowed root. Resolves symlinks
    before comparison to defeat symlink-escape attacks."""
    try:
        resolved = p.expanduser().resolve()
    except (RuntimeError, OSError) as e:
        raise HTTPException(status_code=400, detail=f"invalid corpus path: {p} ({e})")
    for root in _allowed_roots():
        try:
            resolved.relative_to(root)
            return  # path is under an allowed root
        except ValueError:
            continue
    raise HTTPException(
        status_code=403,
        detail=(
            f"corpus path {p} (resolves to {resolved}) is not under any allowed root. "
            f"Allowed roots: {[str(r) for r in _allowed_roots()]}. "
            "Override with the SMARTEMBED_ALLOWED_CORPUS_ROOTS env var "
            "(colon-separated, like PATH)."
        ),
    )

app = FastAPI(
    title="SmartEmbedAgent",
    description=(
        "Local embedding-model recommender. POST a corpus, get back a "
        "ranked recommendation with concrete index/throughput numbers, "
        "chunking strategy, hardware fit, fine-tuning advice, language "
        "profile, and a paired reranker suggestion."
    ),
    version="0.2.0",
)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class RecommendRequest(BaseModel):
    corpus_paths: Optional[List[str]] = Field(
        default=None,
        description="Absolute paths to corpus files or directories. "
                    "Each may be .txt / .md / .csv / .json or a directory of those.",
    )
    corpus_text: Optional[str] = Field(
        default=None,
        description="Raw text to analyze (alternative to corpus_paths).",
    )
    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Inline user config matching config/config_schema.json. "
                    "If omitted, falls back to config_path or the bundled sample.",
    )
    config_path: Optional[str] = Field(
        default=None,
        description="Absolute path to a config JSON file. Ignored when `config` is provided.",
    )
    use_llm: bool = Field(
        default=False,
        description="If True, invoke the Ollama-backed LangChain agent. "
                    "Requires Ollama running locally with a tool-calling model. "
                    "Default False = ~1-2s deterministic heuristic.",
    )
    task: Optional[str] = Field(
        default=None,
        description=("Workload type. One of: retrieval (default), classification, "
                     "clustering, deduplication, similarity. Drives task-aware "
                     "scoring, prefix suppression for symmetric tasks, and reranker "
                     "recommendation. Overrides the value in `config.model_preferences.task`."),
    )


class RecommendResponse(BaseModel):
    recommendation: Dict[str, Any]
    config_used: Dict[str, Any]
    used_llm: bool
    markdown_report: str
    notes: List[str] = Field(default_factory=list)


class EvaluateRequest(BaseModel):
    corpus_paths: Optional[List[str]] = Field(default=None)
    corpus_text: Optional[str] = Field(default=None)
    candidates: List[Dict[str, Any]] = Field(
        ...,
        description="List of candidate model entries. Each needs 'name'; "
                    "may include 'embed_prefix' and 'query_prefix' for asymmetric models. "
                    "Typically the top-3 from a prior /recommend response.",
    )
    n_queries: int = Field(default=30, ge=5, le=200,
        description="How many synthetic queries to generate. 30 is a good speed/accuracy balance.")
    use_llm_for_queries: bool = Field(default=True,
        description="If True, asks the local Ollama LLM to write natural-language queries. "
                    "If False, falls back to keyword extraction (no LLM dependency, lossier).")
    heuristic_top_model: Optional[str] = Field(default=None,
        description="Optional: name of the model the heuristic recommended first. "
                    "Used to compute the `diverged` flag in the response.")


class EvaluateResponse(BaseModel):
    report: Dict[str, Any]
    notes: List[str] = Field(default_factory=list)


class ValidatedRecommendRequest(BaseModel):
    corpus_paths: Optional[List[str]] = Field(default=None)
    corpus_text: Optional[str] = Field(default=None)
    config: Optional[Dict[str, Any]] = Field(default=None)
    config_path: Optional[str] = Field(default=None)
    task: Optional[str] = Field(default=None)
    use_llm: bool = Field(default=True,
        description="If True, the Suggester / Decider / Reporter agents use the local Ollama LLM. "
                    "If False, all agents fall back to deterministic logic (faster, no LLM dependency).")
    n_queries: int = Field(default=30, ge=5, le=200)


class ValidatedRecommendResponse(BaseModel):
    context: Dict[str, Any]  # serialized AgentContext (recommendation, eval_report, decision, markdown_report, per_agent_notes)
    notes: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _resolve_config(req_config: Optional[Dict[str, Any]],
                    req_config_path: Optional[str]) -> Dict[str, Any]:
    """Pick the user config in precedence order: inline > path > bundled sample."""
    if req_config is not None:
        return req_config
    if req_config_path:
        path = Path(req_config_path).expanduser()
        if not path.exists():
            raise HTTPException(status_code=400, detail=f"config_path not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(SAMPLE_CONFIG_PATH.read_text(encoding="utf-8"))


def _validate_or_400(config: Dict[str, Any]) -> None:
    schema = json.loads(Path(DEFAULT_SCHEMA_PATH).read_text(encoding="utf-8"))
    errors = validate_config(config, schema)
    if errors:
        raise HTTPException(status_code=422, detail={"config_validation_errors": errors})


def _load_text_for_request(corpus_paths: Optional[List[str]],
                           corpus_text: Optional[str]) -> str:
    if corpus_text and corpus_paths:
        raise HTTPException(status_code=400,
                            detail="Provide either corpus_text or corpus_paths, not both.")
    if corpus_text:
        return corpus_text
    if not corpus_paths:
        raise HTTPException(status_code=400,
                            detail="Provide corpus_text, corpus_paths, or upload files.")
    # Enforce allowlist BEFORE load_corpus touches the filesystem. Without
    # this, a public-facing instance would let any caller read any file the
    # server-process user can read.
    paths = [Path(p) for p in corpus_paths]
    for p in paths:
        _enforce_corpus_path_allowlist(p)
    try:
        return load_corpus(paths)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Fields we let the LLM override on top of the deterministic heuristic
# when use_llm=True. Everything else (index_estimate,
# reranker_recommendation, language_profile) stays deterministic — those
# are just math/lookup over corpus stats and gain nothing from LLM
# rephrasing, but they DO get lost if the LLM doesn't know to emit them.
_LLM_OVERRIDABLE_FIELDS = {
    "recommended_models",
    "reasoning_explanation",
    "chunking_strategy",
    "fine_tuning_advice",
    "hardware_fit_analysis",
}


def _merge_llm_into_heuristic(heuristic: Dict[str, Any],
                              llm: Dict[str, Any]) -> Dict[str, Any]:
    """Start from the deterministic recommendation (which has every field,
    including the new data-scientist additions). Overlay only the LLM's
    judgment fields. Anything the LLM didn't emit keeps the heuristic
    value, so the response is always shape-complete."""
    merged = dict(heuristic)
    for key in _LLM_OVERRIDABLE_FIELDS:
        if key in llm and llm[key]:
            merged[key] = llm[key]
    return merged


def _resolve_task(req_task: Optional[str], config: Dict[str, Any],
                  notes: List[str]) -> str:
    """Pick the task string, with precedence:  request override > config field
    > legacy `target_use_case` > 'retrieval'. Unknown values are mapped back
    to 'retrieval' with a note so the user sees what happened."""
    chosen = req_task if req_task else extract_model_preferences(config)["task"]
    if chosen not in SUPPORTED_TASKS:
        notes.append(
            f"Unknown task '{chosen}'. Falling back to 'retrieval'. "
            f"Supported: {list(SUPPORTED_TASKS)}."
        )
        chosen = "retrieval"
    return chosen


def _run_recommendation(corpus: str, config: Dict[str, Any], use_llm: bool,
                        task: Optional[str] = None) -> Tuple[Dict[str, Any], List[str]]:
    """Returns (recommendation, notes). Notes carry user-visible signals
    such as 'LLM requested but fell back to heuristic because …' so the
    UI never silently substitutes a different code path."""
    pii_cfg = extract_pii_config(config)
    notes: List[str] = []
    resolved_task = _resolve_task(task, config, notes)
    # Always compute the deterministic recommendation. It populates every
    # field the API contract promises (incl. index_estimate, reranker,
    # language_profile, task). The LLM either overrides the judgment fields
    # or we fall back to it entirely.
    heuristic = run_pipeline_no_llm(corpus, user_config=pii_cfg, task=resolved_task)
    if not use_llm:
        return heuristic, notes
    try:
        executor = build_agent(corpus=corpus, user_config=pii_cfg, verbose=False)
        response = executor.invoke({
            "input": "Analyze the loaded corpus. Run device_profiler, then pii_remover, "
                     "then corpus_analyzer, then output the final structured JSON recommendation.",
        })
        output = response.get("output", "") if isinstance(response, dict) else str(response)
        parsed = _extract_json(output)
        # The LLM might return valid JSON that doesn't match our schema —
        # e.g. small models often dump the LAST tool's output (corpus_analyzer's
        # raw stats) and call that the answer. Require at least
        # `recommended_models` to consider the response usable.
        if isinstance(parsed, dict) and "recommended_models" in parsed:
            # Merge the LLM's judgment fields onto the deterministic
            # baseline so index_estimate / reranker / language_profile are
            # always present.
            return _merge_llm_into_heuristic(heuristic, parsed), notes
        if parsed is not None:
            wrong_shape_keys = sorted(parsed.keys())[:6]
            notes.append(
                "LLM agent returned valid JSON but with the wrong shape "
                f"(top-level keys: {wrong_shape_keys}). This typically means "
                "the model dumped the last tool's output instead of synthesizing "
                "a recommendation. Used deterministic heuristic instead. "
                "Try a stronger model (qwen2.5:32b, llama3.3:70b) via "
                "SMARTEMBED_LLM_MODEL."
            )
        else:
            clipped = output.strip().replace("\n", " ")[:140]
            notes.append(
                "LLM agent returned non-JSON output (no JSON found). "
                f"Sample: '{clipped}…'. Used deterministic heuristic instead. "
                "Try a stronger model (qwen2.5:32b, llama3.3:70b) via "
                "SMARTEMBED_LLM_MODEL."
            )
        return heuristic, notes
    except ModuleNotFoundError as e:
        notes.append(
            f"LLM requested but '{e.name}' not installed. Run "
            "`pip install -r requirements.txt` to enable the agentic path. "
            "Used deterministic heuristic instead."
        )
        return heuristic, notes
    except Exception as e:
        traceback.print_exc()
        notes.append(f"LLM agent failed ({type(e).__name__}: {e}). Used deterministic heuristic instead.")
        return heuristic, notes


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/favicon.ico")
def favicon() -> Any:
    """Avoid noisy 404s in the access log when browsers auto-fetch /favicon.ico.
    Returning 204 (no content) is the conventional 'I have no favicon, please
    stop asking' response — cheaper than serving an actual icon."""
    from fastapi.responses import Response
    return Response(status_code=204)


@app.get("/healthz")
@app.get("/api/health")
def healthz() -> Dict[str, Any]:
    """Liveness check + reachability info for downstream services."""
    import urllib.request
    import urllib.error
    import os
    ollama_reachable = False
    try:
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        with urllib.request.urlopen(f"{base}/api/tags", timeout=1) as r:
            ollama_reachable = r.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        ollama_reachable = False
    return {
        "ok": True,
        "version": app.version,
        "ollama_reachable": ollama_reachable,
        "default_model": os.getenv("SMARTEMBED_LLM_MODEL", "hermes3:8b"),
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "version": app.version},
    )


@app.get("/methodology", response_class=HTMLResponse)
def methodology(request: Request) -> Any:
    """Static page explaining how the recommendation is computed end-to-end —
    corpus profile, hardware profile, candidate pool, scoring, task-aware
    bumps, the LLM agent overlay, privacy posture, and known limits."""
    return templates.TemplateResponse(
        "methodology.html",
        {"request": request, "version": app.version},
    )


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    config = _resolve_config(req.config, req.config_path)
    _validate_or_400(config)
    corpus = _load_text_for_request(req.corpus_paths, req.corpus_text)
    rec, notes = _run_recommendation(corpus, config, req.use_llm, task=req.task)
    return RecommendResponse(
        recommendation=rec,
        config_used=config,
        used_llm=bool(req.use_llm),
        markdown_report=render_markdown_report(rec),
        notes=notes,
    )


@app.post("/recommend/upload", response_model=RecommendResponse)
async def recommend_upload(
    files: List[UploadFile] = File(..., description="Corpus files to analyze."),
    config_file: Optional[UploadFile] = File(None, description="Optional user config JSON."),
    use_llm: bool = Form(False),
    task: Optional[str] = Form(None,
        description="Workload type. retrieval | classification | clustering | deduplication | similarity. "
                    "Defaults to value in the (uploaded or bundled) config; falls back to 'retrieval'."),
) -> RecommendResponse:
    """Multipart variant — accept the corpus as one or more uploaded files
    and (optionally) a config JSON. Files are written to a tempdir and
    processed via the same load_corpus path the CLI uses."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one corpus file is required.")

    if config_file is not None:
        try:
            config = json.loads((await config_file.read()).decode("utf-8"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid config JSON: {e}")
    else:
        config = json.loads(SAMPLE_CONFIG_PATH.read_text(encoding="utf-8"))
    _validate_or_400(config)

    tmpdir = None
    try:
        corpus, n_files, tmpdir = await _save_uploads_and_load_corpus(files)
        rec, run_notes = _run_recommendation(corpus, config, use_llm, task=task)
        return RecommendResponse(
            recommendation=rec,
            config_used=config,
            used_llm=bool(use_llm),
            markdown_report=render_markdown_report(rec),
            notes=[f"Analyzed {n_files} uploaded file(s) totalling {len(corpus):,} chars."]
                  + run_notes,
        )
    finally:
        if tmpdir is not None:
            _cleanup_tempdir(tmpdir)


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    """Empirically rank candidate embedders on the supplied corpus.

    This is slow (1-5 min typical on M4 Max with cached models, longer
    on first-run downloads). It produces per-model MRR / nDCG@10 /
    recall@k metrics measured on synthetic (query, source-doc) pairs
    generated from the corpus itself."""
    # Reuse the existing corpus-loading + path-allowlist machinery.
    corpus = _load_text_for_request(req.corpus_paths, req.corpus_text)

    notes: List[str] = []
    llm = None
    if req.use_llm_for_queries:
        try:
            llm = build_llm()
        except Exception as e:
            notes.append(
                f"Couldn't initialise local LLM for query generation ({type(e).__name__}: {e}). "
                "Falling back to keyword extraction."
            )

    report = evaluate_candidates(
        corpus=corpus,
        candidates=req.candidates,
        n_queries=req.n_queries,
        llm=llm,
        heuristic_top_model=req.heuristic_top_model,
    )
    return EvaluateResponse(report=report.to_dict(), notes=notes)


@app.post("/evaluate/upload", response_model=EvaluateResponse)
async def evaluate_upload(
    files: List[UploadFile] = File(..., description="Corpus files to evaluate on."),
    candidates_json: str = Form(..., description="JSON-encoded list of candidate model dicts (same shape as /evaluate's `candidates` field)."),
    n_queries: int = Form(30),
    use_llm_for_queries: bool = Form(True),
    heuristic_top_model: Optional[str] = Form(None),
) -> EvaluateResponse:
    """Multipart variant of /evaluate. Takes the same uploaded files
    /recommend/upload took; the UI uses this so users in 'Upload files'
    mode can validate without having to switch to 'Paste text'."""
    try:
        candidates = json.loads(candidates_json)
        if not isinstance(candidates, list):
            raise ValueError("candidates_json must be a JSON array")
    except (ValueError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=400, detail=f"invalid candidates_json: {e}")

    notes: List[str] = []
    llm = None
    if use_llm_for_queries:
        try:
            llm = build_llm()
        except Exception as e:
            notes.append(
                f"Couldn't initialise local LLM for query generation ({type(e).__name__}: {e}). "
                "Falling back to keyword extraction."
            )

    tmpdir = None
    try:
        corpus, n_files, tmpdir = await _save_uploads_and_load_corpus(files)
        notes.append(f"Evaluating on {n_files} uploaded file(s) totalling {len(corpus):,} chars.")
        report = evaluate_candidates(
            corpus=corpus,
            candidates=candidates,
            n_queries=n_queries,
            llm=llm,
            heuristic_top_model=heuristic_top_model,
        )
        return EvaluateResponse(report=report.to_dict(), notes=notes)
    finally:
        if tmpdir is not None:
            _cleanup_tempdir(tmpdir)


@app.post("/validated_recommend", response_model=ValidatedRecommendResponse)
def validated_recommend(req: ValidatedRecommendRequest) -> ValidatedRecommendResponse:
    """End-to-end validated pipeline: Suggester → QueryGenerator → Evaluator
    → Decider → Reporter. Returns the full AgentContext (recommendation +
    eval report + decision + markdown narrative + per-agent process trace).

    Slow (~1-5 min on M4 Max for typical corpora with cached models;
    longer on first-run model downloads). The Evaluator step dominates."""
    notes: List[str] = []
    config = _resolve_config(req.config, req.config_path)
    _validate_or_400(config)
    corpus = _load_text_for_request(req.corpus_paths, req.corpus_text)

    resolved_task = _resolve_task(req.task, config, notes)

    llm = None
    if req.use_llm:
        try:
            llm = build_llm()
        except Exception as e:
            notes.append(
                f"Couldn't initialise local LLM ({type(e).__name__}: {e}). "
                "All agents will use deterministic fallback paths."
            )

    ctx = run_validated_recommendation(
        corpus=corpus,
        config=config,
        task=resolved_task,
        use_llm=req.use_llm,
        llm=llm,
        n_queries=req.n_queries,
    )
    return ValidatedRecommendResponse(context=ctx.to_dict(), notes=notes)


@app.post("/validated_recommend/upload", response_model=ValidatedRecommendResponse)
async def validated_recommend_upload(
    files: List[UploadFile] = File(..., description="Corpus files."),
    config_file: Optional[UploadFile] = File(None, description="Optional user config JSON."),
    task: Optional[str] = Form(None),
    use_llm: bool = Form(True),
    n_queries: int = Form(30),
) -> ValidatedRecommendResponse:
    """Multipart variant of /validated_recommend. Runs the full 5-agent
    pipeline on uploaded files so users in 'Upload files' mode aren't
    cut off from the validated workflow."""
    notes: List[str] = []
    if config_file is not None:
        try:
            config = json.loads((await config_file.read()).decode("utf-8"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid config JSON: {e}")
    else:
        config = json.loads(SAMPLE_CONFIG_PATH.read_text(encoding="utf-8"))
    _validate_or_400(config)

    resolved_task = _resolve_task(task, config, notes)

    llm = None
    if use_llm:
        try:
            llm = build_llm()
        except Exception as e:
            notes.append(
                f"Couldn't initialise local LLM ({type(e).__name__}: {e}). "
                "All agents will use deterministic fallback paths."
            )

    tmpdir = None
    try:
        corpus, n_files, tmpdir = await _save_uploads_and_load_corpus(files)
        notes.append(f"Validated workflow over {n_files} uploaded file(s) totalling {len(corpus):,} chars.")
        ctx = run_validated_recommendation(
            corpus=corpus,
            config=config,
            task=resolved_task,
            use_llm=use_llm,
            llm=llm,
            n_queries=n_queries,
        )
        return ValidatedRecommendResponse(context=ctx.to_dict(), notes=notes)
    finally:
        if tmpdir is not None:
            _cleanup_tempdir(tmpdir)


@app.get("/recommend/markdown", response_class=PlainTextResponse)
def recommend_markdown_demo() -> str:
    """Convenience endpoint — runs the recommender on the bundled sample
    corpus and returns the Markdown report as plain text. Useful for
    smoke-testing the install."""
    sample_corpus = (PROJECT_ROOT / "data" / "sample_short.txt").read_text(encoding="utf-8")
    config = json.loads(SAMPLE_CONFIG_PATH.read_text(encoding="utf-8"))
    rec, _notes = _run_recommendation(sample_corpus, config, use_llm=False)
    return render_markdown_report(rec)


# ---------------------------------------------------------------------------
# Console entry point
# ---------------------------------------------------------------------------
def run() -> None:
    """Console-script entry point. Reads PORT / HOST / RELOAD from env."""
    import os
    import uvicorn
    uvicorn.run(
        "src.api.server:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() in {"true", "1", "yes"},
    )


if __name__ == "__main__":
    run()
