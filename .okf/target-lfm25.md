---
type: Target Model
title: 'Liquid AI LFM2.5-2.6B'
description: 'The first concrete model target for llmopt on Apple Silicon.'
resource: https://huggingface.co/LiquidAI/LFM2.5-2.6B
tags: [target, lfm2.5, apple-silicon, inference]
status: stable
generated: { by: codex/gpt-5, at: '2026-08-20T11:24:21Z' }
sources:
  - id: lfm25-config
    resource: https://huggingface.co/LiquidAI/LFM2.5-2.6B/raw/main/config.json
    title: LFM2.5-2.6B config.json
  - id: lfm25-card
    resource: https://huggingface.co/LiquidAI/LFM2.5-2.6B
    title: LFM2.5-2.6B model card
---

# Shape and topology

The checked-in target descriptor mirrors the supplied model configuration:

| Field | Value |
|---|---:|
| hidden size | 2048 |
| intermediate size | 10752 |
| hidden layers | 30 |
| convolution blocks | 22 |
| full-attention/GQA blocks | 8 |
| attention heads | 32 |
| key/value heads | 8 |
| vocabulary | 128000 |
| maximum position embeddings | 131072 |
| convolution cache length | 3 |
| checkpoint dtype | bfloat16 |

The first model-shaped compiler probe is the projection family with matrix
shapes `(rows, 2048) x (2048, 10752)` and bias `(1, 10752)`. The target
descriptor is in [lib/lfm25.ml](../lib/lfm25.ml). The complete forward probe
uses the Transformers checkpoint and PyTorch MPS; custom model-specific OCaml
lowering remains separate from that runtime path.[^lfm25-config]

# Provenance

The model card describes LFM2.5-2.6B as a hybrid on-device model and identifies
the 30-layer composition and context size.[^lfm25-card]

[^lfm25-config]: Official `config.json` for `LiquidAI/LFM2.5-2.6B`.
[^lfm25-card]: Official Hugging Face model card for `LiquidAI/LFM2.5-2.6B`.
