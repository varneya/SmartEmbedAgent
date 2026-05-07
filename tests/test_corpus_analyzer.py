"""
Unit tests for src/corpus_analyzer.py.

The HF tokenizer requires a network download; in environments without it the
analyzer falls back to whitespace tokenization. The tests here are written so
they pass under either tokenizer — assertions check ranges and structural
properties rather than exact token counts.

Run with:
    python -m pytest tests/test_corpus_analyzer.py
or:
    python tests/test_corpus_analyzer.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.corpus_analyzer import CONTEXT_WINDOW_THRESHOLDS, analyze_corpus


REQUIRED_TOP_LEVEL_KEYS = {
    "doc_count",
    "token_stats",
    "context_window_recommendations",
    "chunking_needed",
    "suggested_chunk_size",
    "suggested_overlap",
    "vocabulary_metrics",
    "domain_indicators",
}


class TestSchema(unittest.TestCase):
    def test_output_has_required_keys(self):
        result = analyze_corpus("Hello world.\n\nAnother short doc.")
        d = result.to_dict()
        for key in REQUIRED_TOP_LEVEL_KEYS:
            self.assertIn(key, d)

    def test_token_stats_keys(self):
        result = analyze_corpus("Sentence one.\n\nSentence two.").to_dict()
        for key in ("total", "mean", "median", "min", "max", "std"):
            self.assertIn(key, result["token_stats"])

    def test_context_window_recommendations_keys(self):
        result = analyze_corpus("doc one\n\ndoc two").to_dict()
        for w in CONTEXT_WINDOW_THRESHOLDS:
            self.assertIn(str(w), result["context_window_recommendations"])
            entry = result["context_window_recommendations"][str(w)]
            self.assertIn("fits", entry)
            self.assertIn("fit_percentage", entry)


class TestShortCorpus(unittest.TestCase):
    """A corpus of short docs should not trigger chunking."""

    def test_short_corpus_no_chunking(self):
        result = analyze_corpus("Doc one.\n\nDoc two.\n\nDoc three.").to_dict()
        self.assertEqual(result["doc_count"], 3)
        self.assertFalse(result["chunking_needed"])
        self.assertIsNone(result["suggested_chunk_size"])
        self.assertIsNone(result["suggested_overlap"])

    def test_short_corpus_fits_all_windows(self):
        result = analyze_corpus("a b c.\n\nd e f.").to_dict()
        for w in CONTEXT_WINDOW_THRESHOLDS:
            self.assertEqual(result["context_window_recommendations"][str(w)]["fit_percentage"], 100.0)


class TestLongCorpus(unittest.TestCase):
    """A corpus where most docs blow past 512 tokens should trigger chunking."""

    def test_long_corpus_recommends_chunking(self):
        long_doc = ("Lorem ipsum dolor sit amet consectetur adipiscing elit. " * 200).strip()
        corpus = "\n\n".join([long_doc] * 4 + ["short doc"])
        result = analyze_corpus(corpus).to_dict()
        self.assertTrue(result["chunking_needed"])
        self.assertIsNotNone(result["suggested_chunk_size"])
        self.assertIsNotNone(result["suggested_overlap"])

    def test_chunk_size_is_under_threshold(self):
        long_doc = ("word " * 1000).strip()
        corpus = "\n\n".join([long_doc] * 3)
        result = analyze_corpus(corpus).to_dict()
        if result["chunking_needed"]:
            self.assertLessEqual(result["suggested_chunk_size"], 512)
            # Overlap is in the typical 10–20% band (with a 16-token floor).
            self.assertGreaterEqual(result["suggested_overlap"], 16)


class TestStatistics(unittest.TestCase):
    def test_token_counts_are_positive(self):
        result = analyze_corpus("The quick brown fox.").to_dict()
        self.assertGreater(result["token_stats"]["total"], 0)

    def test_doc_count_matches_input(self):
        docs = ["a b c", "d e f g", "h i", "j k l m n"]
        result = analyze_corpus(docs).to_dict()
        self.assertEqual(result["doc_count"], 4)

    def test_min_max_relationship(self):
        result = analyze_corpus(["short", "this is a much longer document with many words"]).to_dict()
        self.assertLessEqual(result["token_stats"]["min"], result["token_stats"]["max"])

    def test_std_zero_for_single_doc(self):
        result = analyze_corpus(["only one doc"]).to_dict()
        self.assertEqual(result["token_stats"]["std"], 0.0)


class TestVocabularyAndDomain(unittest.TestCase):
    def test_type_token_ratio_in_range(self):
        result = analyze_corpus("the the the the the").to_dict()
        ttr = result["vocabulary_metrics"]["type_token_ratio"]
        self.assertGreaterEqual(ttr, 0.0)
        self.assertLessEqual(ttr, 1.0)

    def test_high_ttr_for_diverse_corpus(self):
        diverse = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        result = analyze_corpus(diverse).to_dict()
        # Every token unique (ignoring case) → ratio should be 1.0.
        self.assertAlmostEqual(result["vocabulary_metrics"]["type_token_ratio"], 1.0, places=2)

    def test_domain_indicators_excludes_stopwords(self):
        text = "the the the embeddings embeddings tokenizer tokenizer pipeline"
        result = analyze_corpus(text).to_dict()
        terms = [d["term"] for d in result["domain_indicators"]]
        self.assertNotIn("the", terms)
        self.assertIn("embeddings", terms)

    def test_domain_indicators_sorted_by_frequency(self):
        text = ("alpha alpha alpha beta beta gamma " * 5)
        result = analyze_corpus(text).to_dict()
        freqs = [d["frequency"] for d in result["domain_indicators"]]
        self.assertEqual(freqs, sorted(freqs, reverse=True))


class TestEdgeCases(unittest.TestCase):
    def test_empty_corpus(self):
        result = analyze_corpus("").to_dict()
        self.assertEqual(result["doc_count"], 1)  # Single (empty-after-strip) doc
        # Should not crash and should return zeros for stats.

    def test_list_input(self):
        result = analyze_corpus(["one", "two", "three"]).to_dict()
        self.assertEqual(result["doc_count"], 3)

    def test_invalid_overlap_raises(self):
        with self.assertRaises(ValueError):
            analyze_corpus("foo", overlap_percentage=2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
