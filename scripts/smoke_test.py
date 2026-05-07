"""
Smoke test — quickly verifies a fresh install can run end to end.

Runs the deterministic pipeline on the general sample corpus, checks that
the output JSON has the required schema, and reports the final
recommendation. Exit code 0 = success.

Run from the project root:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import main  # noqa: E402


REQUIRED_KEYS = {
    "recommended_models",
    "reasoning_explanation",
    "chunking_strategy",
    "fine_tuning_advice",
    "hardware_fit_analysis",
}


def smoke() -> int:
    print("=" * 60)
    print("SmartEmbedAgent — smoke test")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "smoke.json"
        rc = main([
            "--corpus_path", str(PROJECT_ROOT / "data" / "sample_general.txt"),
            "--config_path", str(PROJECT_ROOT / "config" / "sample_config.json"),
            "--output_path", str(out_path),
            "--no_llm",
        ])
        if rc != 0:
            print(f"\n[FAIL] main.py exited with code {rc}")
            return rc

        if not out_path.exists():
            print(f"\n[FAIL] expected JSON output at {out_path}")
            return 1

        data = json.loads(out_path.read_text())
        missing = REQUIRED_KEYS - set(data.keys())
        if missing:
            print(f"\n[FAIL] missing keys: {missing}")
            return 1

        if not data["recommended_models"]:
            print("\n[FAIL] recommended_models list is empty")
            return 1

        top = data["recommended_models"][0]
        print()
        print(f"[OK] Top recommendation: {top.get('name')}")
        print(f"     Rationale: {top.get('rationale')}")
        print(f"     Chunking needed: {data['chunking_strategy']['needed']}")
        print()
        print("[OK] Smoke test passed.")
        return 0


if __name__ == "__main__":
    raise SystemExit(smoke())
