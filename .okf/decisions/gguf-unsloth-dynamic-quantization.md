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

`llmopt` retains **PyTorch Dynamo FX (`torch.compile(model, backend=llmopt_backend)`)** as its primary graph capture and execution planning frontend, while adding first-class **GGUF** (with Unsloth Dynamic / UD quant schemes) as an alternative high-performance weight distribution format alongside native safetensors / `weights.llmopt`.

This gives users flexible options:
1. **PyTorch Capture Architecture**: Captures arbitrary model architectures directly from Python/PyTorch (`AutoModelForCausalLM`, Dynamo FX) with automated state binding, KV-cache planning, and pass optimizations.
2. **Dual Weight Store Formats**:
   - **Safetensors / `weights.llmopt`**: Direct weight archives generated from PyTorch checkpoints.
   - **GGUF / Unsloth Dynamic (UD)**: Pre-quantized mixed-precision GGUF files (`UD-Q4_K_XL`, `UD-Q8_K_XL`, `Q4_K_M`, `Q8_0`) for superior quantization quality and zero-conversion deployment.
3. **Structured Quantized Block Descriptors**: Formal superblock and block-quant descriptors (`Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q5_0`, `F16`, `BF16`, `F32`) in `Ir.Dtype` and `Weight_archive.Dtype`.
4. **Two Dequantization Kernel Families**:
   - **Legacy Block-32 Family**: `Q8_0`, `Q5_0`, `Q4_0` (32-element blocks with FP16 scale).
   - **K-Quant Superblock-256 Family**: `Q4_K`, `Q5_K`, and `Q6_K` (256-element superblocks with 8 sub-blocks of 32 elements, 6-bit scales and mins).
5. **AOT Code Specialization**: Bake quant block layouts, sub-block extraction, and tile dimensions directly into compile-time specialized Metal MSL megakernels (`w4a16_dual_swiglu`, `w4a16_down_add`, `w4a16_lm_head`), eliminating dynamic loop strides and runtime branches.
6. **Offline AOT Transcoding for Exotic Types**: Rare non-uniform codebook types (e.g. `IQ4_XS`, which accounts for only ~6% of tensors in `UD-Q4_K_XL`) are transcoded offline to `Q5_K` at package assembly time, avoiding GPU codebook kernel bloat.
7. **Deterministic Bit-Exact Verification**: Verify dequantization correctness by comparing `llmopt` dequantized tensors byte-for-byte against `llama.cpp`'s reference dequantization on real GGUF binaries.

# Context & Problem

The current `llmopt` quantization stack has three structural limitations:

- **Uniform 4-Bit Policy**: `quantization.py` replaces every `nn.Linear` (including `lm_head` and sensitive attention projections) with symmetric W4A16 group-64 quantization, degrading model perplexity compared to sensitivity-aware mixed-precision formats.
- **Frontend Asset Reconstruction**: Tokenizer assets, Jinja chat templates, and architectural metadata must be manually supplied or inferred. GGUF carries these assets in-file.
- **Closed ABI**: `kernel_ir.mli` and `weight_archive.ml` assume a fixed 64-element group size, preventing ingestion of industry-standard 256-element superblocks or asymmetric scales/mins.

# Architecture & Scope

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GGUF / UD INGESTION PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. GGUF Parser: Reads metadata, KV config, tokenizer, tensor headers        │
│ 2. Offline Transcoder: Converts IQ4_XS tensors to Q5_K                      │
│ 3. AOT Compiler: Maps each tensor to specialized Metal MSL megakernels     │
│    • Q8_0 (Block-32, 1 scale)                                               │
│    • Q4_K (Superblock-256, 4-bit nibbles + sub-block scales/mins)           │
│    • Q5_K (Superblock-256, 5-bit nibbles + high plane + scales/mins)        │
│    • Q6_K (Superblock-256, 6-bit nibbles + high plane + int8 scales)        │
│ 4. Bit-Exact Unit Tests: Validates dequant against llama.cpp reference      │
│ 5. Model Program & Serving: Bakes into prebaked Metal ICB serving bundle   │
└─────────────────────────────────────────────────────────────────────────────┘
```

# Phased Target Tiers

- **Tier 1 (Diagnostic)**: `UD-Q8_K_XL` (`Q8_0` + `F16`/`F32`). Near-lossless, isolates compiler optimizations from quantization errors.
- **Tier 2 (High Efficiency)**: `UD-Q4_K_XL` (`Q4_K`, `Q5_K`, `Q6_K`, `IQ4_XS->Q5_K`). Validates maximum bandwidth efficiency on 1–3B models (`Qwen3-0.6B`, `Llama-3.2-1B`, `SmolLM2-135M`).
