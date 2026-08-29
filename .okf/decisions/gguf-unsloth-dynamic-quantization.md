---
type: Decision
title: 'Target GGUF and Unsloth Dynamic (UD) Quantization with AOT Code Specialization'
description: 'Adopt GGUF and Unsloth Dynamic (UD) quant schemes as the primary weight format, using AOT code specialization to bake dequantization parameters directly into specialized Metal megakernels with zero runtime dispatch penalty.'
tags: [decision, quantization, gguf, unsloth, metal, compiler, dequantization, aot]
status: stable
generated: { by: human:tung, at: '2026-08-29T01:53:30+07:00' }
sources:
  - id: requested-direction
    resource: /direction.md
    title: Project direction and Unsloth UD analysis
  - id: unsloth-dynamic-3
    resource: https://unsloth.ai/docs/basics/dynamic-3.0-ggufs
    title: Unsloth Dynamic 3.0 GGUFs Documentation
  - id: weight-archive
    resource: /lib/weight_archive.ml
    title: Weight archive binary format
  - id: model-program
    resource: /lib/model_program.ml
    title: Model Program root execution contract
---

# Decision

`llmopt` retains **PyTorch Dynamo FX (`torch.compile(model, backend=llmopt_backend)`)** as the authority for graph topology and execution planning, and targets **GGUF with Unsloth Dynamic / UD quant schemes** as the primary weight-distribution format. `weights.llmopt` remains an internal package/archive representation, not the product's fixed quantization policy.

This gives users flexible options:
1. **PyTorch Capture Authority**: Captures model computation directly from Python/PyTorch (`AutoModelForCausalLM`, Dynamo FX). Operator topology, parameter use, entrypoints, and state roles come from that capture and the compilation session.
2. **GGUF / Unsloth Dynamic Weights**: Uses pre-quantized mixed-precision GGUF tensors (`UD-Q4_K_XL`, `UD-Q8_K_XL`, `Q4_K_M`, `Q8_0`) while retaining the captured graph as the executable architecture.
3. **Structured Quantized Block Descriptors**: Formal superblock and block-quant descriptors (`Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q5_0`, `F16`, `BF16`, `F32`) in `Ir.Dtype` and `Weight_archive.Dtype`.
4. **Two Dequantization Kernel Families**:
   - **Legacy Block-32 Family**: `Q8_0`, `Q5_0`, `Q4_0` (32-element blocks with FP16 scale).
   - **K-Quant Superblock-256 Family**: `Q4_K`, `Q5_K`, and `Q6_K` (256-element superblocks with 8 sub-blocks of 32 elements, 6-bit scales and mins).
5. **AOT Code Specialization**: Bake each captured parameter's declared GGUF quant block layout, sub-block extraction, and tile dimensions directly into specialized Metal MSL kernels, eliminating runtime architecture or quant-format dispatch.
6. **Offline AOT Transcoding for Exotic Types**: Rare non-uniform codebook types (e.g. `IQ4_XS`, which accounts for only ~6% of tensors in `UD-Q4_K_XL`) are transcoded offline to `Q5_K` at package assembly time, avoiding GPU codebook kernel bloat.
7. **Deterministic Bit-Exact Verification**: Verify dequantization correctness by comparing `llmopt` dequantized tensors byte-for-byte against `llama.cpp`'s reference dequantization on real GGUF binaries.

# Graph authority and tensor binding

GGUF `general.architecture` and family-specific metadata may be inspected and
recorded as provenance, but never select compiler passes, schedule rules,
runtime classes, or model adapters. This differs from llama.cpp's
architecture-ID dispatch: llmopt already has the executable graph.

The compilation session binds each lifted FX parameter to a GGUF tensor using
an explicit name map plus shape and dtype/quant-descriptor validation. Missing,
duplicate, or shape-incompatible mappings are errors. The importer does not
reconstruct the network from GGUF tensor names or architecture metadata.

# Context & Problem

The current `llmopt` quantization stack has three structural limitations:

- **Uniform 4-Bit Policy**: `quantization.py` replaces every `nn.Linear` (including `lm_head` and sensitive attention projections) with symmetric W4A16 group-64 quantization, which cannot preserve UD's per-tensor sensitivity choices.
- **Disconnected Assets**: Tokenizer and chat metadata are not yet linked from GGUF into the explicit Model Program processor contract.
- **Closed ABI**: `kernel_ir.mli` and `weight_archive.ml` assume a fixed 64-element group size, preventing ingestion of industry-standard 256-element superblocks or asymmetric scales/mins.

# Architecture & Scope

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GGUF / UD INGESTION PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. FX Capture: Defines operators, parameter use, entrypoints, and state     │
│ 2. GGUF Parser: Reads tokenizer metadata and quantized tensor descriptors   │
│ 3. Tensor Binder: Explicit FX parameter -> GGUF tensor map + validation     │
│ 4. Offline Transcoder: Converts IQ4_XS tensors to Q5_K                      │
│ 5. AOT Compiler: Maps each tensor to specialized Metal MSL megakernels     │
│    • Q8_0 (Block-32, 1 scale)                                               │
│    • Q4_K (Superblock-256, 4-bit nibbles + sub-block scales/mins)           │
│    • Q5_K (Superblock-256, 5-bit nibbles + high plane + scales/mins)        │
│    • Q6_K (Superblock-256, 6-bit nibbles + high plane + int8 scales)        │
│ 6. Bit-Exact Unit Tests: Validates dequant against llama.cpp reference      │
│ 7. Model Program & Serving: Bakes into prebaked Metal ICB serving bundle   │
└─────────────────────────────────────────────────────────────────────────────┘
```

# Phased Target Tiers

- **Tier 1 (Diagnostic)**: `UD-Q8_K_XL` (`Q8_0` + `F16`/`F32`). Near-lossless, isolates compiler optimizations from quantization errors.
- **Tier 2 (High Efficiency)**: `UD-Q4_K_XL` (`Q4_K`, `Q5_K`, `Q6_K`, `IQ4_XS->Q5_K`). Validates maximum bandwidth efficiency on 1–3B models (`Qwen3-0.6B`, `Llama-3.2-1B`, `SmolLM2-135M`).
