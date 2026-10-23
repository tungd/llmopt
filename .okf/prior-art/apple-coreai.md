---
type: Prior Art
title: 'Apple Core AI: PyTorch-to-Apple-Silicon AOT export via proprietary IR and pre-conversion'
description: 'Apple Core AI (coreai-torch, coreai-optimization, coreai-models, CoreAIKit) bridges PyTorch ExportedPrograms to Apple Silicon CPU/GPU/ANE via pre-export compression and opaque system compilation.'
tags: [prior-art, apple, coreai, pytorch, aot, quantization, metal, ane]
status: stable
stale_after: '2027-10-23'
generated: { by: antigravity/2.0, at: '2026-10-23T16:52:00+07:00' }
verified: { by: human:tung, at: '2026-10-23T16:47:51+07:00' }
sources:
  - id: coreai-torch
    resource: https://github.com/apple/coreai-torch
    title: Apple Core AI PyTorch conversion and custom lowering library
  - id: coreai-opt
    resource: https://github.com/apple/coreai-optimization
    title: Apple Core AI PyTorch model compression and palettization toolkit
  - id: coreai-models
    resource: https://github.com/apple/coreai-models
    title: Apple Core AI model export recipes and Swift runtime utilities
  - id: local-model-program
    resource: /decisions/model-program-boundary.md
    title: llmopt Model Program execution contract
  - id: local-canonical-tile32
    resource: /decisions/canonical-tile32-affine-and-hadamard-attention-rotation.md
    title: Canonical Tile32_Affine and Hadamard attention rotation decision
---

# What it is

Apple's **Core AI** ecosystem (`apple/coreai-torch`, `apple/coreai-optimization`,
`apple/coreai-models`, and the Swift `CoreAIKit` runtime) is Apple's on-device
generative AI compilation and execution stack for Apple Silicon (iPhone, iPad,
Mac, Vision Pro).[^coreai-torch][^coreai-models] It succeeds Core ML for LLM and
vision-transformer workloads by converting `torch.export.ExportedProgram` graphs
into `.aimodel` bundles executed across CPU, Metal GPU, and the Apple Neural
Engine (ANE).

# Comparison with `llmopt`

| Dimension | Apple Core AI (`apple/coreai-*`) | `llmopt` |
| :--- | :--- | :--- |
| **Graph Acquisition** | `torch.export.ExportedProgram` via `TorchConverter`; frequently requires rewriting blocks with `coreai_torch.composite_ops` (`coreai-models`).[^coreai-torch] | Unmodified PyTorch `torch.compile` (Dynamo/FX) $\to$ binary `graph.llmopt` (`LLMOPTFX` v2).[^local-model-program] |
| **Compiler Transparency** | Closed-source system compiler lowering `AIProgram` to Metal/ANE; custom ops require hand-written `TorchMetalKernel`. | Inspectable algebraic OCaml compiler (`Coordinate_map`, `Applicative_expr`, `Epilogue`, `Contraction_group`) with explicit JSON optimization receipts. |
| **Quantization Workflow** | Requires running `coreai-opt` (PTQ/QAT/palettization) inside Python *before* exporting to `.aimodel`.[^coreai-opt] Cannot ingest community GGUFs. | Ingests standard GGUF / Unsloth Dynamic distributions and transcodes them offline into canonical `Tile32_Affine` tiles.[^local-canonical-tile32] |
| **Serving vs. In-App SDK** | Swift in-app framework (`CoreAIKit`) with stateful KV handles for single-user edge apps. | Native OCaml + Metal multi-request serving engine (`llmopt-serve`) with SGLang-derived radix prefix caching and SRPT continuous batching. |

# Key lessons for `llmopt`

1. **Zero-Rewrite Capture is a Decisive Advantage**: `coreai-models` requires per-family Python export recipes and composite-op substitutions for attention, RoPE, and KV caching.[^coreai-models] `llmopt` must preserve its invariant that unmodified PyTorch models compile directly from captured FX topology.
2. **Pre-Compilation Weight Normalization Works**: Core AI avoids a runtime kernel zoo by normalizing weights into a small set of hardware-aligned compressed representations before runtime load. `llmopt` achieves the same runtime simplicity—without giving up GGUF compatibility—by transcoding heterogeneous GGUF block formats into canonical `Tile32_Affine` blocks at link time.[^local-canonical-tile32]
3. **Layer-Wise Parity Debugging**: Core AI ships an intermediate tensor debugger comparing PyTorch reference activations against device outputs. Maintaining automated per-stage parity checks in `llmopt` isolates numerical drift immediately.

[^coreai-torch]: Apple `coreai-torch` repository and converter documentation.
[^coreai-opt]: Apple `coreai-optimization` model compression library.
[^coreai-models]: Apple `coreai-models` export recipes and Swift utilities.
[^local-model-program]: llmopt decision - Model Program boundary.
[^local-canonical-tile32]: llmopt decision - Canonical Tile32_Affine and Hadamard attention rotation.
