"""
Config Validator
================

Validates user configuration JSON against `config/config_schema.json`.

Why a custom validator instead of jsonschema? jsonschema is excellent but
adds a dependency that the project otherwise doesn't need. The schema we
use is small enough that a focused validator fits in ~150 lines and
produces error messages tailored to our config shape.

Public API
----------
    validate_config(config: dict, schema: dict) -> List[str]
        Returns a list of error messages (empty if valid).

    load_and_validate(config_path, schema_path=None) -> Tuple[dict, List[str]]
        Convenience: loads JSON from disk, validates, returns
        (config, errors).

    extract_pii_config(config: dict) -> dict
        Maps the nested config to the flat shape `pii_remover.remove_pii`
        expects. Used by `main.py` so downstream modules keep their simple
        API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "config" / "config_schema.json"


def _validate_type(value: Any, expected: str, path: str, errors: List[str]) -> bool:
    type_map = {
        "object": dict,
        "array": list,
        "string": str,
        "number": (int, float),
        "boolean": bool,
        "integer": int,
        "null": type(None),
    }
    py_type = type_map.get(expected)
    if py_type is None:
        return True
    # bool is a subclass of int — reject when expecting number from a bool.
    if expected == "number" and isinstance(value, bool):
        errors.append(f"{path}: expected {expected}, got boolean")
        return False
    if not isinstance(value, py_type):
        errors.append(f"{path}: expected {expected}, got {type(value).__name__}")
        return False
    return True


def _validate_against_node(value: Any, node: Dict[str, Any], path: str, errors: List[str]) -> None:
    """Recursively validate `value` against a schema node."""
    expected_type = node.get("type")
    if expected_type and not _validate_type(value, expected_type, path, errors):
        return

    if expected_type == "object":
        required = node.get("required", [])
        properties = node.get("properties", {})
        additional_allowed = node.get("additionalProperties", True)

        for req in required:
            if req not in value:
                errors.append(f"{path}: missing required field '{req}'")

        for key, sub_value in value.items():
            if key.startswith("_"):
                continue  # `_comment` style keys are allowed everywhere
            if key in properties:
                _validate_against_node(sub_value, properties[key], f"{path}.{key}", errors)
            elif additional_allowed is False:
                errors.append(f"{path}: unknown field '{key}' (additionalProperties is false)")
        return

    if expected_type == "array":
        items_schema = node.get("items")
        if items_schema:
            for i, item in enumerate(value):
                _validate_against_node(item, items_schema, f"{path}[{i}]", errors)
        return

    enum = node.get("enum")
    if enum is not None and value not in enum:
        errors.append(f"{path}: value {value!r} not in allowed values {enum}")

    if expected_type == "number":
        minimum = node.get("minimum")
        maximum = node.get("maximum")
        excl_max = node.get("exclusiveMaximum")
        if minimum is not None and value < minimum:
            errors.append(f"{path}: value {value} below minimum {minimum}")
        if maximum is not None and value > maximum:
            errors.append(f"{path}: value {value} above maximum {maximum}")
        if excl_max is not None and value >= excl_max:
            errors.append(f"{path}: value {value} not below exclusive maximum {excl_max}")


def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Validate `config` against `schema`. Returns a list of error
    messages; an empty list means the config is valid."""
    errors: List[str] = []
    _validate_against_node(config, schema, "$", errors)
    return errors


def load_and_validate(
    config_path: Path,
    schema_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Load a JSON config from disk and validate it. Returns
    (config_dict, errors). Raises FileNotFoundError or JSONDecodeError
    on file-loading problems."""
    config_path = Path(config_path)
    schema_path = Path(schema_path or DEFAULT_SCHEMA_PATH)

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    return config, validate_config(config, schema)


# ---------------------------------------------------------------------------
# Adapters: nested config -> flat shape used by individual modules
# ---------------------------------------------------------------------------
def extract_pii_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the nested `pii_settings` block into the flat shape
    `pii_remover.remove_pii` expects."""
    pii = config.get("pii_settings", {}) or {}
    out: Dict[str, Any] = {
        "whitelist": pii.get("whitelist", []),
        "redaction_list": pii.get("custom_redaction_list", []),
        "ner_model": pii.get("ner_model_choice", "dslim/bert-base-NER"),
        "recognizer": pii.get("recognizer", "legacy"),
        "region_packs": pii.get("region_packs", []),
    }
    aggressiveness = pii.get("redaction_aggressiveness", "medium")
    if aggressiveness == "low":
        out["entity_types"] = []  # regex only
    elif aggressiveness == "medium":
        out["entity_types"] = ["PERSON", "LOCATION", "ORGANIZATION"]
    elif aggressiveness == "high":
        out["entity_types"] = ["PERSON", "LOCATION", "ORGANIZATION", "MISC"]
    return out


# Map the legacy `target_use_case` enum onto the new `task` vocabulary so
# old configs keep producing the same recommendation behavior. Values not
# in this map fall through to "retrieval" (the safe default).
_LEGACY_TASK_MAP = {
    "semantic_search": "retrieval",
    "retrieval":       "retrieval",
    "classification":  "classification",
    "clustering":      "clustering",
    "reranking":       "retrieval",  # reranking happens AFTER retrieval; the embedder is still retrieval-shaped
}


def extract_model_preferences(config: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the model-preference knobs into a flat dict the orchestrator
    consumes. Honors the new `task` field, falling back to the legacy
    `target_use_case` for back-compat with older configs."""
    prefs = config.get("model_preferences", {}) or {}
    task = prefs.get("task")
    if not task:
        legacy = prefs.get("target_use_case")
        task = _LEGACY_TASK_MAP.get(legacy, "retrieval") if legacy else "retrieval"
    return {
        "task": task,
        "max_model_size_gb": prefs.get("max_model_size_gb"),
        "preferred_model_families": list(prefs.get("preferred_model_families", []) or []),
        "prioritize_speed_over_accuracy": bool(prefs.get("prioritize_speed_over_accuracy", False)),
    }


def extract_agent_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """Pull agent runtime knobs into a flat dict consumed by main.py."""
    agent = config.get("agent_settings", {}) or {}
    return {
        "enable_web_search": bool(agent.get("enable_web_search", True)),
        "search_cache_ttl_seconds": int(agent.get("search_cache_days", 1) * 86400),
        "llm_model": agent.get("llm_model", "claude-sonnet-4-6"),
        "verbose_logging": bool(agent.get("verbose_logging", False)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Validate a SmartEmbedAgent config file.")
    parser.add_argument("config_path", help="Path to user config JSON.")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH), help="Schema path.")
    args = parser.parse_args()

    try:
        _, errors = load_and_validate(Path(args.config_path), Path(args.schema))
    except FileNotFoundError as e:
        print(f"ERROR: file not found: {e}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print(f"ERROR: invalid JSON in config: {e}", file=sys.stderr)
        return 2

    if errors:
        print(f"FAIL: {len(errors)} validation error(s):")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("OK: config is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
