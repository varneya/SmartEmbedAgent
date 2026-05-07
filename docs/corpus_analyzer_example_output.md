# `corpus_analyzer.analyze_corpus()` — Example Output

Sample input is three documents: a short sentence, a long Lorem ipsum paragraph (~400 whitespace-tokens), and a short note. Run with the default tokenizer (`bert-base-uncased`); falls back to whitespace tokenization if `transformers` isn't installed.

## Output

```json
{
  "doc_count": 3,
  "token_stats": {
    "total": 411,
    "mean": 137.0,
    "median": 9.0,
    "min": 2,
    "max": 400,
    "std": 187.66
  },
  "context_window_recommendations": {
    "512":  { "fits": 3, "fit_percentage": 100.0 },
    "1024": { "fits": 3, "fit_percentage": 100.0 },
    "2048": { "fits": 3, "fit_percentage": 100.0 },
    "4096": { "fits": 3, "fit_percentage": 100.0 },
    "8192": { "fits": 3, "fit_percentage": 100.0 }
  },
  "chunking_needed": false,
  "suggested_chunk_size": null,
  "suggested_overlap": null,
  "vocabulary_metrics": {
    "unique_tokens": 16,
    "total_tokens": 411,
    "type_token_ratio": 0.0389
  },
  "domain_indicators": [
    { "term": "lorem",        "frequency": 50 },
    { "term": "ipsum",        "frequency": 50 },
    { "term": "dolor",        "frequency": 50 },
    { "term": "consectetur",  "frequency": 50 },
    { "term": "adipiscing",   "frequency": 50 },
    { "term": "embeddings",   "frequency": 1 }
  ],
  "tokenizer_used": "whitespace",
  "notes": [
    "Long-tail outliers present (max=400 tokens). Consider per-document chunking decisions or truncation."
  ]
}
```

## Interpreting the fields

`token_stats` is the unconditional descriptive statistics. The high standard deviation (187.66) relative to the median (9) is the immediate flag here — this corpus has one very long doc and two short ones. The `notes` field surfaces this as an outlier warning.

`context_window_recommendations` shows what fraction of documents fit within each common context-window size. Compact embedding models top out around 512 tokens; 4096+ is the territory of OpenAI / Voyage / Cohere long-context models. If the 512 row drops below 90%, chunking is recommended.

`vocabulary_metrics` reports type-token ratio. Low TTR (as here, 0.04) signals heavy repetition — typical of templated or boilerplate text. Higher TTR (0.5+) means lexically diverse content.

`domain_indicators` is a fast-and-loose proxy for domain terminology. Stopwords are filtered, then top-K tokens by frequency are returned. For real domain analysis, consider TF-IDF against a reference corpus.

`tokenizer_used` records which tokenizer was active (HF model name, tiktoken encoding, or the whitespace fallback) so the consumer knows how to interpret token counts.

## Test results

18 tests run, all pass:

```
test_chunk_size_is_under_threshold                       ... ok
test_long_corpus_recommends_chunking                     ... ok
test_short_corpus_fits_all_windows                       ... ok
test_short_corpus_no_chunking                            ... ok
test_output_has_required_keys                            ... ok
test_token_stats_keys                                    ... ok
test_context_window_recommendations_keys                 ... ok
test_doc_count_matches_input                             ... ok
test_min_max_relationship                                ... ok
test_std_zero_for_single_doc                             ... ok
test_token_counts_are_positive                           ... ok
test_domain_indicators_excludes_stopwords                ... ok
test_domain_indicators_sorted_by_frequency               ... ok
test_high_ttr_for_diverse_corpus                         ... ok
test_type_token_ratio_in_range                           ... ok
test_empty_corpus                                        ... ok
test_invalid_overlap_raises                              ... ok
test_list_input                                          ... ok
```
