"""
Aggregate test entry point.

The per-module test files are the source of truth:
    - test_device_profiler.py
    - test_pii_remover.py
    - test_corpus_analyzer.py

This file exists only to keep `python -m unittest tests.test_modules` working
for callers that haven't switched to the per-module imports yet.
"""

from tests.test_corpus_analyzer import *  # noqa: F401, F403
from tests.test_device_profiler import *  # noqa: F401, F403
from tests.test_pii_remover import *      # noqa: F401, F403
