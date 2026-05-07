"""
Tests for src/config_validator.py.

Verifies that valid configs pass cleanly, that invalid configs produce
specific error messages naming the offending field, and that the
nested -> flat adapter functions produce the right shape.
"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_validator import (
    DEFAULT_SCHEMA_PATH,
    extract_agent_settings,
    extract_pii_config,
    load_and_validate,
    validate_config,
)

SAMPLE_CONFIG_PATH = PROJECT_ROOT / "config" / "sample_config.json"


def _load_schema():
    with open(DEFAULT_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_sample():
    config, _ = load_and_validate(SAMPLE_CONFIG_PATH)
    return config


class TestSampleConfig(unittest.TestCase):
    def test_sample_config_validates(self):
        config, errors = load_and_validate(SAMPLE_CONFIG_PATH)
        self.assertEqual(errors, [], msg=f"Sample config produced errors: {errors}")

    def test_sample_config_has_all_top_level_groups(self):
        config = _load_sample()
        for key in ("pii_settings", "model_preferences", "hardware_constraints", "agent_settings"):
            self.assertIn(key, config)


class TestRequiredFields(unittest.TestCase):
    def test_missing_top_level_group_reports_clear_error(self):
        schema = _load_schema()
        config = _load_sample()
        del config["pii_settings"]
        errors = validate_config(config, schema)
        self.assertTrue(any("pii_settings" in e for e in errors))

    def test_missing_nested_field_reports_path(self):
        schema = _load_schema()
        config = _load_sample()
        del config["pii_settings"]["whitelist"]
        errors = validate_config(config, schema)
        self.assertTrue(any("whitelist" in e for e in errors))


class TestTypeValidation(unittest.TestCase):
    def test_wrong_type_for_array(self):
        schema = _load_schema()
        config = _load_sample()
        config["pii_settings"]["whitelist"] = "not an array"
        errors = validate_config(config, schema)
        self.assertTrue(any("whitelist" in e and "array" in e for e in errors))

    def test_wrong_type_for_boolean(self):
        schema = _load_schema()
        config = _load_sample()
        config["model_preferences"]["prioritize_speed_over_accuracy"] = "true"  # string, not bool
        errors = validate_config(config, schema)
        self.assertTrue(any("prioritize_speed_over_accuracy" in e for e in errors))

    def test_wrong_type_for_number(self):
        schema = _load_schema()
        config = _load_sample()
        config["model_preferences"]["max_model_size_gb"] = "two"
        errors = validate_config(config, schema)
        self.assertTrue(any("max_model_size_gb" in e for e in errors))


class TestEnumAndRange(unittest.TestCase):
    def test_invalid_enum_value(self):
        schema = _load_schema()
        config = _load_sample()
        config["pii_settings"]["redaction_aggressiveness"] = "extreme"
        errors = validate_config(config, schema)
        self.assertTrue(any("redaction_aggressiveness" in e and "extreme" in e for e in errors))

    def test_below_minimum(self):
        schema = _load_schema()
        config = _load_sample()
        config["model_preferences"]["max_model_size_gb"] = -1
        errors = validate_config(config, schema)
        self.assertTrue(any("max_model_size_gb" in e for e in errors))

    def test_above_maximum(self):
        schema = _load_schema()
        config = _load_sample()
        config["agent_settings"]["search_cache_days"] = 9999
        errors = validate_config(config, schema)
        self.assertTrue(any("search_cache_days" in e for e in errors))


class TestUnknownFields(unittest.TestCase):
    def test_unknown_field_in_strict_section(self):
        schema = _load_schema()
        config = _load_sample()
        config["pii_settings"]["misspelled_field"] = "value"
        errors = validate_config(config, schema)
        self.assertTrue(any("misspelled_field" in e for e in errors))

    def test_comment_keys_are_ignored(self):
        # `_comment` keys (any key starting with `_`) are documentation-only.
        schema = _load_schema()
        config = _load_sample()
        config["_extra_note"] = "this should be ignored"
        config["pii_settings"]["_internal_note"] = "also ignored"
        errors = validate_config(config, schema)
        self.assertEqual(errors, [])


class TestAdapters(unittest.TestCase):
    def test_extract_pii_config_shape(self):
        config = _load_sample()
        flat = extract_pii_config(config)
        self.assertIn("whitelist", flat)
        self.assertIn("redaction_list", flat)
        self.assertIn("ner_model", flat)
        self.assertIn("entity_types", flat)
        self.assertEqual(flat["redaction_list"], config["pii_settings"]["custom_redaction_list"])

    def test_aggressiveness_low_is_regex_only(self):
        config = _load_sample()
        config["pii_settings"]["redaction_aggressiveness"] = "low"
        self.assertEqual(extract_pii_config(config)["entity_types"], [])

    def test_aggressiveness_high_includes_misc(self):
        config = _load_sample()
        config["pii_settings"]["redaction_aggressiveness"] = "high"
        self.assertIn("MISC", extract_pii_config(config)["entity_types"])

    def test_extract_agent_settings_converts_days_to_seconds(self):
        config = _load_sample()
        config["agent_settings"]["search_cache_days"] = 2
        out = extract_agent_settings(config)
        self.assertEqual(out["search_cache_ttl_seconds"], 2 * 86400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
