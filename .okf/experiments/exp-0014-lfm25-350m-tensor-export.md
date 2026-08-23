---
type: Experiment
title: 'Complete LFM2.5-350M Q8 tensor export'
description: 'Bind every Dynamo-lifted model tensor in the captured forward to one streaming safetensors archive and validate it from OCaml.'
tags: [experiment, lfm25, q8, fx, safetensors, ocaml, package]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T17:34:00Z' }
sources:
  - id: capture
    resource: /python/llmopt_backend/__init__.py
    title: FX static-input classification and bindings
  - id: writer
    resource: /python/llmopt_backend/tensor_archive.py
    title: Streaming safetensors writer
  - id: validator
    resource: /lib/serving_validation.ml
    title: FX-to-archive binding validator
  - id: result
    resource: /_artifacts/lfm25-350m-q8-package-2026-08-24/result.json
    title: Bounded model probe result
  - id: package
    resource: /_artifacts/lfm25-350m-q8-package-2026-08-24/graphs/graph-0000/package.json
    title: Emitted serving package
---

# Question

Can one real Q8 capture distinguish runtime request values from lifted model
state, stream all static tensors into one archive without staging a second
whole model on CPU, and produce bindings the OCaml package checker accepts?

# Implementation

Dynamo's `_dynamo_static_input_type` marker classifies lifted placeholder
state; `get_attr` nodes are static. Every other placeholder is runtime-bound.
Aliased tensors share a canonical key. The manifest carries only that binding;
the archive remains authoritative for tensor storage metadata.

The writer computes the safetensors header from shape/dtype metadata, then
copies at most one tensor from MPS to CPU, writes it, and releases that staging
view before continuing. OCaml checks every tensor key, dtype, shape, offset,
and payload extent before accepting a serving package.

# Model observation

The single probe used `LiquidAI/LFM2.5-350M`, Q8 weight-only conversion, one
eager forward, one compile/forward, and one compiled measurement forward. The
preflight reported 58% of 25.77 GB system memory free, 49 GiB disk available,
and no competing model process; post-run memory was 53% free.

The capture converted 92 linear modules and emitted:

* 1,115 FX and IR plan nodes;
* 241 tensor-store bindings and one runtime-bound `input_ids` value;
* 241 archive tensors: 148 F16, 92 I8, and 1 F32;
* 422,104,704 tensor payload bytes;
* a 422,144,400-byte `weights.safetensors`; and
* a `serving` package accepted by `llmopt-package-check`.

Eager and compiled logits were bit-exact (`max_abs=0`, `mean_abs=0`). The run
did not execute ERS, generation, or needle retrieval. Its one-sample forward
times are retained in the result artifact but do not support a speed claim.

# Remaining boundary

The current package describes a no-cache forward only. Its textual plan has
379 typed/lowered nodes and 736 opaque nodes. Decode/KV-state graph capture,
machine-executable invocation serialization, native full-model dispatch,
tokenization, and request serving remain open.
