"""
Unit tests for src/api/server.py.

These run against the FastAPI TestClient — no live HTTP server, no
sockets. Skipped automatically if FastAPI isn't installed (the API is an
optional extra), so the suite still passes on a default-install
environment.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


try:
    from fastapi.testclient import TestClient
    from src.api.server import app
    HAS_API = True
except ImportError:
    HAS_API = False


@unittest.skipUnless(HAS_API, "FastAPI extras not installed (pip install -r requirements-api.txt)")
class TestHealthAndIndex(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_healthz(self):
        r = self.client.get("/healthz")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        for key in ("ok", "version", "ollama_reachable", "default_model"):
            self.assertIn(key, body)
        self.assertTrue(body["ok"])

    def test_index_html_returns(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        # The page should reference the title and the form id we render against.
        self.assertIn("SmartEmbedAgent", r.text)
        self.assertIn('id="dropzone"', r.text)


@unittest.skipUnless(HAS_API, "FastAPI extras not installed")
class TestRecommendEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_recommend_with_corpus_text(self):
        r = self.client.post("/recommend", json={
            "corpus_text": "Customer feedback. Email me at alice@example.com.\n\n"
                           "We use embeddings for semantic search across our knowledge base.",
            "use_llm": False,
        })
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        for key in ("recommendation", "config_used", "used_llm", "markdown_report"):
            self.assertIn(key, body)
        rec = body["recommendation"]
        # Schema contract — the same fields as the CLI emits.
        for key in ("recommended_models", "reasoning_explanation", "chunking_strategy",
                    "fine_tuning_advice", "hardware_fit_analysis"):
            self.assertIn(key, rec)
        self.assertGreater(len(rec["recommended_models"]), 0)
        self.assertIn("name", rec["recommended_models"][0])
        # New data-scientist fields surface here too.
        self.assertIn("index_estimate", rec)
        self.assertIn("reranker_recommendation", rec)
        self.assertIn("language_profile", rec)

    def test_recommend_rejects_both_text_and_paths(self):
        r = self.client.post("/recommend", json={
            "corpus_text": "hi",
            "corpus_paths": ["/tmp/whatever.txt"],
        })
        self.assertEqual(r.status_code, 400)
        self.assertIn("not both", r.json()["detail"])

    def test_recommend_rejects_empty(self):
        r = self.client.post("/recommend", json={})
        self.assertEqual(r.status_code, 400)

    def test_recommend_404_on_missing_path(self):
        r = self.client.post("/recommend", json={
            "corpus_paths": ["/tmp/definitely_does_not_exist_12345.txt"],
        })
        self.assertEqual(r.status_code, 404)

    def test_recommend_with_known_corpus_path(self):
        sample = PROJECT_ROOT / "data" / "sample_short.txt"
        r = self.client.post("/recommend", json={
            "corpus_paths": [str(sample)],
            "use_llm": False,
        })
        self.assertEqual(r.status_code, 200, r.text)
        rec = r.json()["recommendation"]
        self.assertGreater(len(rec["recommended_models"]), 0)


@unittest.skipUnless(HAS_API, "FastAPI extras not installed")
class TestUploadEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_upload_single_file(self):
        sample_text = (
            "Customer feedback. The product is excellent for our use case.\n\n"
            "We had questions about the API rate limits and pricing.\n\n"
            "Reach me at alice@example.com or 555-123-4567."
        )
        files = [("files", ("notes.txt", sample_text.encode("utf-8"), "text/plain"))]
        r = self.client.post("/recommend/upload", files=files,
                             data={"use_llm": "false"})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIn("recommendation", body)
        self.assertGreater(len(body["recommendation"]["recommended_models"]), 0)
        # Notes string should report the upload count.
        self.assertTrue(any("uploaded" in n for n in body.get("notes", [])))

    def test_upload_rejects_no_files(self):
        # FastAPI returns 422 for missing required form-file (validation error).
        r = self.client.post("/recommend/upload", data={"use_llm": "false"})
        self.assertEqual(r.status_code, 422)


@unittest.skipUnless(HAS_API, "FastAPI extras not installed")
class TestTaskOverride(unittest.TestCase):
    """`task` flows through both /recommend and /recommend/upload."""

    def setUp(self):
        self.client = TestClient(app)

    def test_recommend_with_task_clustering(self):
        r = self.client.post("/recommend", json={
            "corpus_text": "alice@example.com is great. The product works well for our team.",
            "use_llm": False,
            "task": "clustering",
        })
        self.assertEqual(r.status_code, 200, r.text)
        rec = r.json()["recommendation"]
        self.assertEqual(rec["task"], "clustering")
        # Clustering should not surface a reranker model.
        self.assertIsNone(rec["reranker_recommendation"]["name"])

    def test_recommend_with_task_deduplication(self):
        r = self.client.post("/recommend", json={
            "corpus_text": "Customer feedback. Email me at alice@example.com.",
            "use_llm": False,
            "task": "deduplication",
        })
        self.assertEqual(r.status_code, 200, r.text)
        rec = r.json()["recommendation"]
        self.assertEqual(rec["task"], "deduplication")
        # Symmetric task → no prompt prefixes carried.
        for m in rec["recommended_models"]:
            self.assertEqual(m["embed_prefix"], "")
            self.assertEqual(m["query_prefix"], "")

    def test_unknown_task_falls_back_with_note(self):
        r = self.client.post("/recommend", json={
            "corpus_text": "hello",
            "use_llm": False,
            "task": "fortune_telling",
        })
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["recommendation"]["task"], "retrieval")
        self.assertTrue(any("Unknown task" in n for n in body["notes"]),
                        f"expected an Unknown-task note in {body['notes']}")

    def test_upload_with_task_form_field(self):
        files = [("files", ("notes.txt", b"Quick text for clustering.\n", "text/plain"))]
        r = self.client.post("/recommend/upload", files=files,
                             data={"use_llm": "false", "task": "clustering"})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["recommendation"]["task"], "clustering")


@unittest.skipUnless(HAS_API, "FastAPI extras not installed")
class TestPathAllowlist(unittest.TestCase):
    """`corpus_paths` must be under an allowed root. Defeats the
    'public server can read /etc/passwd' class of issues."""

    def setUp(self):
        self.client = TestClient(app)

    def test_path_outside_allowed_roots_returns_403(self):
        # /etc/passwd is a classic 'reach for arbitrary file' target. Even
        # if the file exists and is readable, the allowlist must reject.
        r = self.client.post("/recommend", json={
            "corpus_paths": ["/etc/passwd"],
            "use_llm": False,
        })
        self.assertEqual(r.status_code, 403)
        self.assertIn("not under any allowed root", r.json()["detail"].lower()
                      if isinstance(r.json().get("detail"), str)
                      else str(r.json()["detail"]).lower())

    def test_traversal_attempt_resolved_then_rejected(self):
        # /tmp/../etc/passwd resolves to /etc/passwd, which is rejected.
        r = self.client.post("/recommend", json={
            "corpus_paths": ["/tmp/../etc/passwd"],
            "use_llm": False,
        })
        self.assertEqual(r.status_code, 403)

    def test_path_inside_tmp_is_allowed(self):
        # Write a small temp corpus under /tmp (an allowed root) and confirm
        # the allowlist permits it through to load_corpus.
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", dir="/tmp", delete=False) as tf:
            tf.write("Customer feedback. Email me at alice@example.com.\n")
            path = tf.name
        try:
            r = self.client.post("/recommend", json={
                "corpus_paths": [path],
                "use_llm": False,
            })
            self.assertEqual(r.status_code, 200, r.text)
        finally:
            Path(path).unlink(missing_ok=True)


@unittest.skipUnless(HAS_API, "FastAPI extras not installed")
class TestUploadSizeCap(unittest.TestCase):
    """Upload endpoint must reject files larger than the configured cap
    so a single request can't fill the disk."""

    def setUp(self):
        self.client = TestClient(app)

    def test_upload_within_limit_succeeds(self):
        # 1 MB synthetic file — well under the 100 MB default cap.
        payload = ("Customer feedback. " * 50_000).encode("utf-8")[:1024 * 1024]
        files = [("files", ("notes.txt", payload, "text/plain"))]
        r = self.client.post("/recommend/upload", files=files,
                             data={"use_llm": "false"})
        self.assertEqual(r.status_code, 200, r.text)

    def test_upload_over_limit_returns_413(self):
        # Set a tiny cap so we don't have to fabricate a 100MB file.
        import os
        prev = os.environ.get("SMARTEMBED_MAX_UPLOAD_MB")
        os.environ["SMARTEMBED_MAX_UPLOAD_MB"] = "1"   # 1 MB cap
        try:
            payload = b"X" * (2 * 1024 * 1024 + 1)     # 2 MB + 1 byte
            files = [("files", ("big.txt", payload, "text/plain"))]
            r = self.client.post("/recommend/upload", files=files,
                                 data={"use_llm": "false"})
            self.assertEqual(r.status_code, 413, r.text)
            self.assertIn("exceed", r.json()["detail"].lower())
        finally:
            if prev is None:
                os.environ.pop("SMARTEMBED_MAX_UPLOAD_MB", None)
            else:
                os.environ["SMARTEMBED_MAX_UPLOAD_MB"] = prev


@unittest.skipUnless(HAS_API, "FastAPI extras not installed")
class TestMarkdownDemo(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_markdown_demo(self):
        r = self.client.get("/recommend/markdown")
        self.assertEqual(r.status_code, 200)
        # Markdown report has known section headers.
        self.assertIn("# SmartEmbedAgent Recommendation Report", r.text)
        self.assertIn("## Executive Summary", r.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
