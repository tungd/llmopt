---
type: Experiment
title: 'Tensor-layout quantization and Linear storage classification'
description: 'Replace repeated quant-format tables and semantic W4 assumptions with one tensor-layout authority and classified Linear storage.'
tags: [experiment, compiler, ir, tensor-layout, quantization, gguf, ud, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-29T23:37:06+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-2-2026-08-29.json
    title: Slice 2 benchmark receipt
  - id: ir
    resource: /lib/ir.ml
    title: Tensor layout and Linear storage authority
  - id: archive
    resource: /lib/weight_archive.ml
    title: Shared archive quant format
---

# Compiler change

`Ir.Tensor_layout` now owns dense and block-quantized physical layout, including
all GGUF block element and byte geometry. `Ir.Value.layout` exposes that
representation to passes and planners. `Weight_archive.Dtype.quant_type` is the
same type as `Ir.Dtype.quant_type`, so archive validation no longer maintains a
constructor-by-constructor translation table.

`Ir.Linear_storage.classify` represents the three storage forms currently
visible at the graph seam: dense tensors, GGUF block-quantized tensors, and
separately scaled groupwise packed W4 tensors. `Kernel_ir.Primitive.Linear`
carries this layout, while its operation remains semantic `m/n/k/bias` Linear.
SwiGLU resource accounting and serving workspace planning now ask the tensor
layout for physical bytes rather than repeating quant-format tables.

# Informational comparison

This representation slice does not change executable package commands or
dispatches. Fresh two-token, no-cache timings are:

| Probe | LLMOpt median | llama.cpp median | Ratio |
|---|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `137.146592 ms` | `8.7125415 ms` | `15.7413x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `63.612938 ms` | `23.0834165 ms` | `2.7558x` |

The llama.cpp samples in this run show wider variance than slice 1; the receipt
retains every sample instead of interpreting the fluctuation as a compiler
effect.

# Validation

The full OCaml and Python suites pass. New regressions verify that IR and weight
archives share one quant format type, Q4_K physical bytes come from the layout,
GGUF weights classify as block-quantized storage, and the retained probe W4
representation classifies as groupwise packed storage with a separate scale.

Evidence is the exhaustive format tests, shared-type compilation, full test
result, and fresh timing receipt. The interpretation is that later Metal tactic
selection can now query a closed layout value instead of branching on semantic
operator names.
