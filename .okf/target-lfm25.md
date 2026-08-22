---
type: Target Model
title: 'Liquid AI LFM2.5-350M'
description: 'The primary model target for llmopt on Apple Silicon.'
resource: https://huggingface.co/LiquidAI/LFM2.5-350M
tags: [target, lfm2.5, apple-silicon, inference]
status: stable
generated: { by: codex/gpt-5, at: '2026-08-20T11:24:21Z' }
sources:
  - id: lfm25-config
    resource: https://huggingface.co/LiquidAI/LFM2.5-350M/raw/main/config.json
    title: LFM2.5-350M config.json
  - id: lfm25-card
    resource: https://huggingface.co/LiquidAI/LFM2.5-350M
    title: LFM2.5-350M model card
---

# Shape and topology

The checked-in target descriptor mirrors the supplied model configuration:

| Field | Value |
|---|---:|
| hidden size | 1024 |
| intermediate size | 6656 |
| hidden layers | 16 |
| convolution blocks | 10 |
| full-attention/GQA blocks | 6 |
| attention heads | 16 |
| key/value heads | 8 |
| vocabulary | 65536 |
| maximum position embeddings | 128000 |
| convolution cache length | 3 |
| checkpoint dtype | bfloat16 |

The first model-shaped compiler probe is the projection family with matrix
shapes `(rows, 1024) x (1024, 6656)` and bias `(1, 6656)`. The target
descriptor is in [lib/lfm25.ml](../lib/lfm25.ml). The complete forward probe
uses the Transformers checkpoint and PyTorch MPS; custom model-specific OCaml
lowering remains separate from that runtime path.[^lfm25-config]

# Provenance

The model card describes LFM2.5-350M as a compact hybrid on-device model and identifies
the 16-layer composition and context size.[^lfm25-card]

[^lfm25-config]: Official `config.json` for `LiquidAI/LFM2.5-350M`.
[^lfm25-card]: Official Hugging Face model card for `LiquidAI/LFM2.5-350M`.
