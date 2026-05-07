# Sample Agent Reasoning Trace — Medical Corpus

This trace captures one verbose run of the LangChain agent against `data/sample_medical.txt` using the configuration in `config/sample_config.json`. It illustrates the deterministic-tool / LLM-judgment split that the architecture is built around.

Command:

```bash
python main.py \
  --corpus_path data/sample_medical.txt \
  --config_path config/sample_config.json \
  --output_path runs/medical.json \
  --verbose
```

Trace (lightly trimmed for readability):

```
[15:42:11] INFO smart_embed_agent: [1/5] Validating user config ...
[15:42:11] INFO smart_embed_agent: Config OK.
[15:42:11] INFO smart_embed_agent: [2/5] Loading corpus ...
[15:42:11] INFO smart_embed_agent: Loaded corpus: 4982 characters.
[15:42:11] INFO smart_embed_agent: [3/5] Initializing agent ...
[15:42:11] INFO smart_embed_agent: [4/5] Running agent reasoning pipeline ...

> Entering new AgentExecutor chain...

[Tool call] device_profiler({})
[Tool result] {
  "total_ram_gb": 16.0,
  "available_ram_gb": 9.42,
  "ram_percentage_used": 41.13,
  "gpu_available": false,
  "gpu_name": "None",
  "gpu_memory_gb": 0.0,
  "compute_device": "cpu"
}

[Agent reasoning]
The host is CPU-only with 16 GB of RAM. That immediately rules out GPU-bound
recommendations like BGE-large or e5-large; I'll keep the candidate pool
focused on small-to-medium models that run comfortably on CPU. RAM is
adequate for any of those.

[Tool call] pii_remover({})
[Tool result] {
  "summary": {
    "NAME": 4,
    "PHONE": 1,
    "EMAIL": 0
  },
  "total_redactions": 5,
  "cleaned_preview": "Patient encounter note for [REDACTED_NAME], DOB 1965-04-12, MRN 8821049. Seen by [REDACTED_NAME] at Cedar Hills Medical Center on 2026-04-03..."
}

[Agent reasoning]
Five redactions in a corpus this small is high — patient names and a phone
number, exactly the profile you'd expect from clinical notes. This is
privacy-sensitive content. Hosted embedding APIs (OpenAI, Cohere, Voyage)
should be ruled out for this workload regardless of their accuracy
advantage; the data shouldn't leave the user's environment.

[Tool call] corpus_analyzer({})
[Tool result] {
  "doc_count": 10,
  "token_stats": { "mean": 287.4, "median": 281, "max": 612, "std": 91.3 },
  "context_window_recommendations": {
    "512":  { "fits": 9, "fit_percentage": 90.0 },
    "1024": { "fits": 10, "fit_percentage": 100.0 },
    "2048": { "fits": 10, "fit_percentage": 100.0 }
  },
  "chunking_needed": false,
  "vocabulary_metrics": {
    "unique_tokens": 312,
    "total_tokens": 645,
    "type_token_ratio": 0.484
  },
  "domain_indicators": [
    { "term": "patient", "frequency": 11 },
    { "term": "medication", "frequency": 7 },
    { "term": "lumbar", "frequency": 5 },
    { "term": "diabetes", "frequency": 4 },
    { "term": "hypertension", "frequency": 3 }
  ]
}

[Agent reasoning]
9 of 10 documents fit within 512 tokens, so chunking isn't strictly required.
The domain indicators ("patient", "lumbar", "diabetes") confirm this is
clinical text. TTR of 0.48 is moderate, suggesting some lexical diversity
but enough domain concentration that a domain-tuned model could plausibly
help.

I should check whether there are domain-tuned medical embedding models that
have been recently benchmarked. This is a freshness-sensitive question, so
web search is justified.

[Tool call] web_search({"query": "PubMedBERT vs BGE-small medical retrieval 2026 benchmark"})
[Tool result] {
  "cached": false,
  "result": "...PubMedBERT and clinical-bert variants continue to outperform
  general-purpose embedders on PubMed-derived retrieval benchmarks by
  4-7 percentage points on nDCG@10, but at meaningful inference-cost
  premium..."
}

[Agent reasoning]
The benchmark data confirms that domain-tuned medical embedders outperform
general-purpose models, but the gap is modest (4-7%) and the inference cost
is higher. For a small 10-document corpus, the cost premium of running a
medical-specific model on CPU may not be justified — the user can probably
get acceptable retrieval quality with a general-purpose CPU model and
revisit if quality is poor.

I'll recommend BGE-small as the primary (fast, CPU-friendly, private) with
PubMedBERT as a domain-tuned alternative if retrieval quality is poor in
testing.

[Final answer]
{
  "recommended_models": [
    {
      "name": "BAAI/bge-small-en-v1.5",
      "rank": 1,
      "rationale": "Best primary choice. 512-token context fits 90% of documents without chunking, runs comfortably on CPU, and is open-source so no PII leaves the host. ~130 MB on disk."
    },
    {
      "name": "pritamdeka/PubMedBERT-mnli-snli-stsb",
      "rank": 2,
      "rationale": "Domain-tuned alternative if BGE-small retrieval quality is poor in evaluation. Trained on PubMed; literature shows 4-7% nDCG@10 advantage on medical retrieval benchmarks. Slower on CPU."
    },
    {
      "name": "sentence-transformers/all-MiniLM-L6-v2",
      "rank": 3,
      "rationale": "Smallest CPU-friendly fallback. 256-token context window will force chunking on the longer documents, but model is ~80 MB and inference is very fast."
    }
  ],
  "reasoning_explanation": "Privacy-sensitive medical corpus (5 redactions across 10 short documents) on CPU-only hardware. This rules out hosted APIs entirely. P95 token count fits within compact-model context windows, so a small open-source embedder is the right primary choice. Domain-tuned models offer a modest accuracy win but their inference cost may not be justified for a corpus this size — recommend evaluating BGE-small first.",
  "chunking_strategy": {
    "needed": false,
    "chunk_size_tokens": null,
    "overlap_tokens": null,
    "rationale": "Document p95 length (~400 tokens) fits within compact-model context windows. The single 612-token outlier can be truncated without significant information loss."
  },
  "fine_tuning_advice": "Not recommended at current corpus size. With only 10 documents, fine-tuning data would be insufficient. Revisit when the corpus grows past several hundred labeled query-document pairs.",
  "hardware_fit_analysis": "BGE-small at ~130 MB is comfortable on a 16 GB CPU host; inference batch size can be modest without memory concerns. PubMedBERT (~440 MB) is also feasible; the constraint is latency, not memory."
}

[15:42:18] INFO smart_embed_agent: [5/5] Generating recommendation reports ...
[15:42:18] INFO smart_embed_agent: Wrote JSON: runs/medical.json
[15:42:18] INFO smart_embed_agent: Wrote Markdown report: runs/medical.md
[15:42:18] INFO smart_embed_agent: Analysis complete.
```

## What this illustrates

**Tools as ground truth, agent as judgment.** The three deterministic tool calls (`device_profiler`, `pii_remover`, `corpus_analyzer`) produce the factual data that grounds the recommendation. The agent's contribution is the synthesis — recognizing that a corpus of clinical notes belongs in a different model class than a corpus of marketing text, and weighing the modest accuracy gain of a domain model against the cost of running it.

**Web search is selective.** The agent didn't call `web_search` until it had a specific question whose answer changes over time ("are there recent benchmarks for medical retrieval embedders?"). For a generic corpus, this fourth tool call wouldn't have fired.

**The output is structured.** The agent's final message is a single JSON object matching the documented schema, not free-form text. This is enforced by the system prompt.
