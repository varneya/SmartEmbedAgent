"""
End-to-end integration tests for main.py.

These run the CLI entry point in --no_llm mode (so they work in CI without
an API key) and verify that the JSON + Markdown reports are written, the
exit code is zero on success, and that error paths produce non-zero exit
codes with informative messages.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import main, load_corpus, render_markdown_report  # noqa: E402


SAMPLE_CONFIG = PROJECT_ROOT / "config" / "sample_config.json"


class TestEndToEndPipeline(unittest.TestCase):
    def test_runs_on_general_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "rec.json"
            rc = main([
                "--corpus_path", str(PROJECT_ROOT / "data" / "sample_general.txt"),
                "--config_path", str(SAMPLE_CONFIG),
                "--output_path", str(out_json),
                "--no_llm",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(out_json.exists())
            self.assertTrue(out_json.with_suffix(".md").exists())
            data = json.loads(out_json.read_text())
            self.assertIn("recommended_models", data)
            self.assertIn("chunking_strategy", data)

    def test_short_corpus_no_chunking(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "rec.json"
            rc = main([
                "--corpus_path", str(PROJECT_ROOT / "data" / "sample_short.txt"),
                "--config_path", str(SAMPLE_CONFIG),
                "--output_path", str(out_json),
                "--no_llm",
            ])
            self.assertEqual(rc, 0)
            data = json.loads(out_json.read_text())
            self.assertFalse(data["chunking_strategy"]["needed"])

    def test_long_corpus_triggers_chunking(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_json = Path(tmp) / "rec.json"
            rc = main([
                "--corpus_path", str(PROJECT_ROOT / "data" / "sample_long.txt"),
                "--config_path", str(SAMPLE_CONFIG),
                "--output_path", str(out_json),
                "--no_llm",
            ])
            self.assertEqual(rc, 0)
            data = json.loads(out_json.read_text())
            self.assertTrue(data["chunking_strategy"]["needed"])


class TestErrorHandling(unittest.TestCase):
    def test_missing_corpus_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = main([
                "--corpus_path", str(Path(tmp) / "nonexistent.txt"),
                "--config_path", str(SAMPLE_CONFIG),
                "--output_path", str(Path(tmp) / "rec.json"),
                "--no_llm",
            ])
            self.assertNotEqual(rc, 0)

    def test_missing_config_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = main([
                "--corpus_path", str(PROJECT_ROOT / "data" / "sample_short.txt"),
                "--config_path", str(Path(tmp) / "nonexistent.json"),
                "--output_path", str(Path(tmp) / "rec.json"),
                "--no_llm",
            ])
            self.assertNotEqual(rc, 0)

    def test_invalid_config_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_cfg = Path(tmp) / "bad_config.json"
            bad_cfg.write_text(json.dumps({"pii_settings": "wrong shape"}))
            rc = main([
                "--corpus_path", str(PROJECT_ROOT / "data" / "sample_short.txt"),
                "--config_path", str(bad_cfg),
                "--output_path", str(Path(tmp) / "rec.json"),
                "--no_llm",
            ])
            self.assertNotEqual(rc, 0)


class TestCorpusLoaders(unittest.TestCase):
    def test_loads_txt(self):
        out = load_corpus(PROJECT_ROOT / "data" / "sample_general.txt")
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 100)

    def test_loads_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "docs.csv"
            csv_path.write_text("text\nfirst doc\nsecond doc\nthird doc\n")
            out = load_corpus(csv_path)
            self.assertIn("first doc", out)
            self.assertIn("third doc", out)

    def test_loads_json_list_of_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            jp = Path(tmp) / "docs.json"
            jp.write_text(json.dumps(["doc one", "doc two"]))
            out = load_corpus(jp)
            self.assertIn("doc one", out)

    def test_loads_json_list_of_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            jp = Path(tmp) / "docs.json"
            jp.write_text(json.dumps([{"text": "doc one"}, {"text": "doc two"}]))
            out = load_corpus(jp)
            self.assertIn("doc one", out)

    def test_loads_directory_of_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "a.txt").write_text("file A content")
            (d / "b.txt").write_text("file B content")
            out = load_corpus(d)
            self.assertIn("file A content", out)
            self.assertIn("file B content", out)


class TestMarkdownReport(unittest.TestCase):
    def test_renders_required_sections(self):
        rec = {
            "recommended_models": [
                {"name": "BGE-small", "rank": 1, "rationale": "small footprint"},
            ],
            "reasoning_explanation": "Test reasoning.",
            "chunking_strategy": {"needed": False, "rationale": "fits"},
            "fine_tuning_advice": "Not needed.",
            "hardware_fit_analysis": "Plenty of RAM.",
        }
        md = render_markdown_report(rec, reasoning_trace="agent ran tools in order")
        for header in [
            "Executive Summary",
            "Recommended Embedding Models",
            "Chunking Strategy",
            "Fine-Tuning Advice",
            "Hardware Fit Analysis",
            "Full Reasoning Trace",
        ]:
            self.assertIn(header, md)


if __name__ == "__main__":
    unittest.main(verbosity=2)
