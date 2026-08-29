---
type: Probe Model
title: 'Liquid AI LFM2.5-350M probe'
description: 'An explicit hybrid recurrent/attention compiler and runtime probe, not a product default.'
resource: https://huggingface.co/LiquidAI/LFM2.5-350M
tags: [probe, lfm2.5, hybrid, apple-silicon, inference]
status: stable
generated: { by: codex/gpt-5, at: '2026-08-29T13:05:00+07:00' }
sources:
  - id: lfm25-config
    resource: https://huggingface.co/LiquidAI/LFM2.5-350M/raw/main/config.json
    title: LFM2.5-350M config.json
  - id: lfm25-card
    resource: https://huggingface.co/LiquidAI/LFM2.5-350M
    title: LFM2.5-350M model card
---

# Role

LFM2.5-350M probes the compiler's hybrid recurrent/attention coverage and the
native runtime's mixed persistent-state lifecycle. Its checked-in constants
are confined to `Lfm25.Config.probe_350m`, `Lfm25_probe`, dedicated probe
binaries, tests, and historical experiment receipts. The generic pipeline,
Model Program linker, generation loop, and server do not select this profile.

# Shape and topology

The checked-in probe descriptor mirrors the supplied model configuration:

| Field | Value |
|---|---:|
| hidden size | 1024 |
| declared intermediate size | 6656 |
| effective SwiGLU feed-forward size | 4608 |
| hidden layers | 16 |
| convolution blocks | 10 |
| full-attention/GQA blocks | 6 |
| attention heads | 16 |
| key/value heads | 8 |
| vocabulary | 65536 |
| maximum position embeddings | 128000 |
| convolution cache length | 3 |
| checkpoint dtype | bfloat16 |

The configuration enables `block_auto_adjust_ff_dim`. Transformers applies
integer `2/3` scaling and rounds upward to `block_multiple_of=256`, so the
declared 6656 becomes an executable width of 4608. Captured `w1`/`w3`
projections therefore use `[4608,1024]` weights and `w2` uses `[1024,4608]`.
The probe descriptor in [lib/lfm25.ml](../lib/lfm25.ml) records both values;
probe fixtures consume the effective width.[^lfm25-config]

# Provenance

The model card describes LFM2.5-350M as a compact hybrid on-device model and identifies
the 16-layer composition and context size.[^lfm25-card]

[^lfm25-config]: Official `config.json` for `LiquidAI/LFM2.5-350M`.
[^lfm25-card]: Official Hugging Face model card for `LiquidAI/LFM2.5-350M`.
