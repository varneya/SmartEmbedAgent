"""
SmartEmbedAgent — main entry point.

Run from the command line:

    python main.py --corpus_path data/sample_general.txt \
                   --config_path config/sample_config.json \
                   --output_path runs/recommendation.json \
                   --verbose

The script:
  1. Loads and validates the user config.
  2. Loads the corpus (txt / csv / json supported, plus directory-of-txt).
  3. Initializes the LangChain agent (Ollama-backed, runs locally).
  4. Runs the full pipeline (or the deterministic fallback if Ollama is
     unreachable or the model isn't pulled).
  5. Writes the recommendation as JSON and as a human-readable Markdown
     report alongside it.

Errors that are usually recoverable (missing files, invalid corpus formats,
GPU detection failures, NER model failures, LLM failures) are caught with
informative messages rather than raw tracebacks. LLM calls have a small
retry-with-backoff loop so transient errors don't fail the whole run.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is importable when run as `python main.py`.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_orchestrator import build_agent, run_pipeline_no_llm  # noqa: E402
from src.config_validator import (  # noqa: E402
    DEFAULT_SCHEMA_PATH,
    extract_agent_settings,
    extract_pii_config,
    load_and_validate,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("smart_embed_agent")


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    # Quiet very chatty third-party libs unless --verbose is set.
    if not verbose:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Progress indicator
# ---------------------------------------------------------------------------
@dataclass
class Step:
    name: str
    label: str


PIPELINE_STEPS = [
    Step("validate_config", "Validating user config"),
    Step("load_corpus",     "Loading corpus"),
    Step("init_agent",      "Initializing agent"),
    Step("run_pipeline",    "Running agent reasoning pipeline"),
    Step("write_report",    "Generating recommendation reports"),
]


def _step_announce(step: Step, total: int, idx: int) -> None:
    """Single-line progress announcement, easy to grep in logs."""
    log.info("[%d/%d] %s ...", idx + 1, total, step.label)


# ---------------------------------------------------------------------------
# Corpus loading — txt, csv, json, or directory of .txt
# ---------------------------------------------------------------------------
TEXT_SUFFIXES = {".txt", ".md"}
KNOWN_SUFFIXES = TEXT_SUFFIXES | {".csv", ".json"}


def _load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_csv(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    header = rows[0]
    data_rows = rows[1:] if any(c.lower() in {"text", "content", "body"} for c in header) else rows
    return "\n\n".join(" ".join(c for c in row if c) for row in data_rows)


def _load_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        docs: List[str] = []
        for item in data:
            if isinstance(item, str):
                docs.append(item)
            elif isinstance(item, dict):
                for key in ("text", "content", "body"):
                    if key in item:
                        docs.append(str(item[key]))
                        break
        if not docs:
            raise ValueError(f"JSON list contains no usable documents: {path}")
        return "\n\n".join(docs)
    if isinstance(data, dict):
        for key in ("text", "content", "body", "documents"):
            if key in data:
                val = data[key]
                if isinstance(val, str):
                    return val
                if isinstance(val, list):
                    return "\n\n".join(str(v) for v in val)
        raise ValueError(f"JSON object has no recognized text field: {path}")
    raise ValueError(f"Unsupported JSON shape in {path}")


def _load_one_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return _load_text_file(path)
    if suffix == ".csv":
        return _load_csv(path)
    if suffix == ".json":
        return _load_json(path)
    raise ValueError(f"Unsupported corpus format: {suffix} ({path})")


def _load_directory(path: Path) -> str:
    files = sorted(c for c in path.iterdir() if c.is_file() and c.suffix.lower() in KNOWN_SUFFIXES)
    if not files:
        raise ValueError(f"Directory contains no supported files (.txt/.md/.csv/.json): {path}")
    docs: List[str] = []
    for f in files:
        try:
            docs.append(_load_one_file(f))
        except (ValueError, json.JSONDecodeError) as e:
            log.warning("Skipping unparseable file %s in directory: %s", f.name, e)
    if not docs:
        raise ValueError(f"Directory {path} had supported file types but none were parseable.")
    return "\n\n".join(docs)


def _load_path(path: Path) -> str:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Corpus path not found: {path}")
    if path.is_dir():
        return _load_directory(path)
    if path.is_file():
        return _load_one_file(path)
    raise ValueError(f"Path is neither file nor directory: {path}")


def load_corpus(paths) -> str:
    """Load a corpus from disk, returning a single joined string. Documents
    are joined with double newlines so the corpus analyzer's blank-line
    splitting recovers them as separate documents.

    Accepts a single Path/str or a list of Paths/strs. Each path may point
    to a file (.txt/.md/.csv/.json) or to a directory containing any mix of
    those file types.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    return "\n\n".join(_load_path(Path(p)) for p in paths)


# ---------------------------------------------------------------------------
# LLM run with retry
# ---------------------------------------------------------------------------
def _run_with_retry(invoke_fn, max_attempts: int = 3, base_delay: float = 1.0) -> Any:
    """Retry transient LLM/API failures with exponential backoff. Idempotent
    operations only — the agent's tool calls are read-only, so retrying the
    whole `invoke` is safe."""
    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return invoke_fn()
        except Exception as e:
            last_err = e
            if attempt == max_attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            log.warning("LLM call failed (attempt %d/%d): %s. Retrying in %.1fs.",
                        attempt, max_attempts, e, delay)
            time.sleep(delay)
    raise last_err  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------
def render_markdown_report(rec: Dict[str, Any], reasoning_trace: Optional[str] = None) -> str:
    """Render the structured recommendation as a human-readable Markdown
    report. Sections match the spec: executive summary, recommended models,
    chunking strategy, fine-tuning advice, hardware fit analysis, full
    reasoning trace."""
    lines: List[str] = ["# SmartEmbedAgent Recommendation Report", ""]

    # Executive summary — top model + one-line takeaway.
    top = (rec.get("recommended_models") or [{}])[0]
    lines += [
        "## Executive Summary",
        "",
        f"**Top recommendation:** `{top.get('name', 'n/a')}`",
        "",
        rec.get("reasoning_explanation", "_No summary provided._"),
        "",
    ]

    # Recommended models.
    lines += ["## Recommended Embedding Models", ""]
    for entry in rec.get("recommended_models", []):
        lines += [
            f"### {entry.get('rank', '?')}. `{entry.get('name', 'n/a')}`",
            "",
            entry.get("rationale", "_No rationale._"),
            "",
        ]

    # Chunking strategy.
    cs = rec.get("chunking_strategy", {})
    lines += ["## Chunking Strategy", ""]
    if cs.get("needed"):
        lines += [
            f"- **Required:** yes",
            f"- **Chunk size (tokens):** {cs.get('chunk_size_tokens')}",
            f"- **Overlap (tokens):** {cs.get('overlap_tokens')}",
            "",
            cs.get("rationale", ""),
            "",
        ]
    else:
        lines += [
            "- **Required:** no",
            "",
            cs.get("rationale", "Documents fit comfortably in compact-model context windows."),
            "",
        ]

    # Fine-tuning advice.
    lines += ["## Fine-Tuning Advice", "", rec.get("fine_tuning_advice", "_No advice provided._"), ""]

    # Hardware fit.
    lines += ["## Hardware Fit Analysis", "", rec.get("hardware_fit_analysis", "_No analysis provided._"), ""]

    # Full reasoning trace.
    if reasoning_trace:
        lines += [
            "## Full Reasoning Trace",
            "",
            "```",
            reasoning_trace.strip(),
            "```",
            "",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="smart-embed-agent",
        description=(
            "Recommend an embedding model and chunking strategy for a corpus. "
            "Uses a local Ollama LLM (default: hermes3:8b) for agentic reasoning; "
            "falls back to a deterministic heuristic if Ollama is unreachable."
        ),
    )
    p.add_argument("--corpus_path", required=True, nargs="+",
                   help=("One or more paths to corpus files (.txt, .md, .csv, .json) "
                         "or directories containing any mix of those types. "
                         "Multiple paths are concatenated into a single corpus."))
    p.add_argument("--config_path", required=True, help="Path to user config JSON.")
    p.add_argument("--output_path", default="runs/recommendation.json",
                   help="Where to write the JSON recommendation. A .md report is written alongside.")
    p.add_argument("--verbose", action="store_true", help="Enable debug-level logging.")
    p.add_argument("--no_llm", action="store_true",
                   help="Skip the local LLM and use the deterministic heuristic. Useful for CI or when Ollama isn't installed.")
    p.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH), help="Config schema path.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    _configure_logging(args.verbose)

    log.info("Starting SmartEmbedAgent analysis.")
    log.info("Corpus: %s", args.corpus_path)
    log.info("Config: %s", args.config_path)

    total_steps = len(PIPELINE_STEPS)

    # Step 1: validate config.
    _step_announce(PIPELINE_STEPS[0], total_steps, 0)
    try:
        config, errors = load_and_validate(Path(args.config_path).expanduser(), Path(args.schema).expanduser())
    except FileNotFoundError as e:
        log.error("Config file not found: %s", e)
        return 2
    except json.JSONDecodeError as e:
        log.error("Config file is not valid JSON: %s", e)
        return 2

    if errors:
        log.error("Config validation failed (%d errors):", len(errors))
        for err in errors:
            log.error("  - %s", err)
        return 1

    pii_cfg = extract_pii_config(config)
    agent_cfg = extract_agent_settings(config)

    # Step 2: load corpus.
    _step_announce(PIPELINE_STEPS[1], total_steps, 1)
    try:
        corpus = load_corpus([Path(p).expanduser() for p in args.corpus_path])
        log.info("Loaded corpus from %d source(s): %d characters.", len(args.corpus_path), len(corpus))
    except (FileNotFoundError, ValueError) as e:
        log.error("Failed to load corpus: %s", e)
        return 2

    # Step 3 + 4: build agent and run pipeline.
    use_llm = not args.no_llm

    recommendation: Dict[str, Any]
    reasoning_trace: Optional[str] = None

    if not use_llm:
        log.warning("--no_llm specified. Running deterministic heuristic pipeline.")
        _step_announce(PIPELINE_STEPS[2], total_steps, 2)
        _step_announce(PIPELINE_STEPS[3], total_steps, 3)
        recommendation = run_pipeline_no_llm(corpus, user_config=pii_cfg)
    else:
        _step_announce(PIPELINE_STEPS[2], total_steps, 2)
        try:
            executor = build_agent(
                corpus=corpus,
                user_config=pii_cfg,
                cache_ttl_seconds=agent_cfg["search_cache_ttl_seconds"],
                verbose=agent_cfg["verbose_logging"] or args.verbose,
            )
        except Exception as e:
            log.error("Failed to initialize agent: %s", e)
            log.debug(traceback.format_exc())
            log.warning("Falling back to deterministic heuristic.")
            recommendation = run_pipeline_no_llm(corpus, user_config=pii_cfg)
        else:
            _step_announce(PIPELINE_STEPS[3], total_steps, 3)
            user_request = (
                "Analyze the loaded corpus. Run device_profiler, then pii_remover, then "
                "corpus_analyzer, then optionally web_search. Output the final structured "
                "JSON recommendation."
            )
            try:
                response = _run_with_retry(lambda: executor.invoke({"input": user_request}))
                output = response.get("output", "") if isinstance(response, dict) else str(response)
                reasoning_trace = output

                # The agent's prompt asks for JSON-only output; parse it. Fall
                # back to the heuristic if the model returns malformed JSON.
                try:
                    recommendation = json.loads(output)
                except (TypeError, json.JSONDecodeError):
                    log.warning("Agent output was not valid JSON; falling back to heuristic.")
                    log.debug("Raw agent output:\n%s", output)
                    recommendation = run_pipeline_no_llm(corpus, user_config=pii_cfg)
            except Exception as e:
                log.error("LLM API call failed after retries: %s", e)
                log.warning("Falling back to deterministic heuristic.")
                recommendation = run_pipeline_no_llm(corpus, user_config=pii_cfg)

    # Step 5: write outputs.
    _step_announce(PIPELINE_STEPS[4], total_steps, 4)
    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(recommendation, indent=2), encoding="utf-8")
    md_path = output_path.with_suffix(".md")
    md_path.write_text(render_markdown_report(recommendation, reasoning_trace), encoding="utf-8")

    log.info("Wrote JSON: %s", output_path)
    log.info("Wrote Markdown report: %s", md_path)
    log.info("Analysis complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
