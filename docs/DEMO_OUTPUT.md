# Demo Output

Below is a real end-to-end run of SmartEmbedAgent, captured from the deterministic heuristic path so it's reproducible without an API key. The LLM-driven path produces output in the same schema with a richer `reasoning_explanation`.

## Command

```bash
python main.py \
  --corpus_path data/sample_medical.txt \
  --config_path config/sample_config.json \
  --output_path runs/demo_medical.json \
  --verbose
```

## Terminal output

```
[15:58:06] INFO smart_embed_agent: Starting SmartEmbedAgent analysis.
[15:58:06] INFO smart_embed_agent: Corpus: data/sample_medical.txt
[15:58:06] INFO smart_embed_agent: Config: config/sample_config.json
[15:58:06] INFO smart_embed_agent: [1/5] Validating user config ...
[15:58:06] INFO smart_embed_agent: [2/5] Loading corpus ...
[15:58:06] INFO smart_embed_agent: Loaded corpus: 3299 characters.
[15:58:06] INFO smart_embed_agent: [3/5] Initializing agent ...
[15:58:06] INFO smart_embed_agent: [4/5] Running agent reasoning pipeline ...
[pii_remover] PII removal complete: 7 redactions across 3 categories.
[15:58:06] INFO smart_embed_agent: [5/5] Generating recommendation reports ...
[15:58:06] INFO smart_embed_agent: Wrote JSON: runs/demo_medical.json
[15:58:06] INFO smart_embed_agent: Wrote Markdown report: runs/demo_medical.md
[15:58:06] INFO smart_embed_agent: Analysis complete.
```

## Recommendation (`runs/demo_medical.json`)

```json
{
  "recommended_models": [
    {
      "name": "sentence-transformers/all-MiniLM-L6-v2",
      "rank": 1,
      "rationale": "Context window 256, tiny (~80 MB). CPU-friendly. Open-source / on-device — privacy-preserving."
    },
    {
      "name": "sentence-transformers/all-mpnet-base-v2",
      "rank": 2,
      "rationale": "Context window 384, medium (~420 MB). CPU-friendly. Open-source / on-device — privacy-preserving."
    },
    {
      "name": "BAAI/bge-small-en-v1.5",
      "rank": 3,
      "rationale": "Context window 512, small (~130 MB). CPU-friendly. Open-source / on-device — privacy-preserving."
    }
  ],
  "reasoning_explanation": "Detected 7 PII redactions (privacy-sensitive). Corpus has 11 documents averaging 287 tokens. No chunking required. Selected model balances corpus size, hardware, and privacy.",
  "chunking_strategy": {
    "needed": false,
    "chunk_size_tokens": null,
    "overlap_tokens": null,
    "rationale": "Documents fit comfortably within compact-model context windows."
  },
  "fine_tuning_advice": "Not necessary as a first pass. The corpus is lexically diverse enough that a strong general-purpose embedding model should perform well; revisit only if retrieval quality is poor.",
  "hardware_fit_analysis": "Total RAM: 16 GB. CPU-only. Top recommendation 'sentence-transformers/all-MiniLM-L6-v2' is tiny (~80 MB) — fits comfortably."
}
```

## Markdown report (`runs/demo_medical.md`, abbreviated)

```markdown
# SmartEmbedAgent Recommendation Report

## Executive Summary

**Top recommendation:** `sentence-transformers/all-MiniLM-L6-v2`

Detected 7 PII redactions (privacy-sensitive). Corpus has 11 documents averaging 287 tokens. No chunking required. Selected model balances corpus size, hardware, and privacy.

## Recommended Embedding Models

### 1. `sentence-transformers/all-MiniLM-L6-v2`
Context window 256, tiny (~80 MB). CPU-friendly. Open-source / on-device — privacy-preserving.

### 2. `sentence-transformers/all-mpnet-base-v2`
Context window 384, medium (~420 MB). CPU-friendly. Open-source / on-device — privacy-preserving.

### 3. `BAAI/bge-small-en-v1.5`
Context window 512, small (~130 MB). CPU-friendly. Open-source / on-device — privacy-preserving.

## Chunking Strategy

- **Required:** no

Documents fit comfortably within compact-model context windows.

## Fine-Tuning Advice

Not necessary as a first pass. The corpus is lexically diverse enough that a strong general-purpose embedding model should perform well; revisit only if retrieval quality is poor.

## Hardware Fit Analysis

Total RAM: 16 GB. CPU-only. Top recommendation 'sentence-transformers/all-MiniLM-L6-v2' is tiny (~80 MB) — fits comfortably.
```

## How the LLM path differs

When `ANTHROPIC_API_KEY` is set, the agent's `reasoning_explanation` would weigh additional factors the heuristic doesn't:

> The corpus is medical, with 7 PII redactions including patient names, MRN-like identifiers, and contact information. This is privacy-sensitive in a regulated sense (HIPAA-adjacent), so I am ruling out hosted embedding APIs entirely — even where the policy permits, the operational complexity of attestations and BAAs is rarely worth it for an internal tool.
>
> Among on-device options, MiniLM is the safe default but has a 256-token window; medical documents are often longer than that. BGE-small at 512 tokens is a better fit for this corpus's p95 length and only adds ~50 MB. I'd recommend BGE-small as the primary choice with MiniLM as a fallback if latency becomes a concern.
>
> Specifically for medical text, consider `pritamdeka/PubMedBERT-mnli-snli-stsb` if your downstream task is research literature retrieval — it outperforms BGE on MedNLI and similar benchmarks. For clinical notes (which this corpus appears to be), the gap closes and BGE remains competitive.
>
> Skip fine-tuning at this stage. With only 11 documents you don't have enough material for a fine-tuning pass to outperform the base model; retrieval quality should be evaluated against a baseline first.

That's the kind of layered judgment the deterministic path can't match — and what justifies the LLM in the loop.

## Reproducing this demo

The deterministic outputs above are reproducible bit-for-bit on any machine with the project installed. The LLM-driven outputs vary by run; the structured fields (`recommended_models`, `chunking_strategy`, etc.) stay schema-stable but the prose in `reasoning_explanation` and `fine_tuning_advice` will differ.
