# SmartEmbedAgent Recommendation Report

## Executive Summary

**Top recommendation:** `sentence-transformers/all-MiniLM-L6-v2`

Detected 0 PII redactions (standard privacy). Corpus has 11 documents averaging 41.09 tokens. No chunking required. Selected model balances corpus size, hardware, and privacy.

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

Total RAM: 3.81 GB. CPU-only. Top recommendation 'sentence-transformers/all-MiniLM-L6-v2' is tiny (~80 MB) — fits comfortably.
