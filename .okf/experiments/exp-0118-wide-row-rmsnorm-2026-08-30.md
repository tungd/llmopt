---
type: Experiment
title: 'Shape-selected wide-row RMSNorm'
description: 'Assign a whole Metal threadgroup to low-row, wide RMSNorm tensors selected from captured shape and dtype.'
tags: [experiment, compiler, tactics, metal, rmsnorm, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T12:45:27+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-16-2026-08-30.json
    title: Slice 16 benchmark receipt
  - id: metal
    resource: /lib/metal.ml
    title: Wide-row RMSNorm kernels
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Captured-shape tactic selection
---

# Structural change

The Metal runtime now assigns one 256-thread threadgroup to each final-axis
RMSNorm row when the captured tensor has fewer than eight rows and at least
eight SIMD-widths per row. The threadgroup reduces across its eight SIMD
groups and cooperatively writes the normalized row. Other shapes retain the
existing tactic that assigns one SIMD group to each row.

Selection consumes captured row count, final-axis width, input/weight dtype,
and target SIMD/threadgroup geometry. It uses no model or tensor name, GGUF
architecture identifier, or quantization format. Both standalone RMSNorm and
the semantic RMSNorm-residual operation use the same shape rule.

# Full-model result

| Probe | Slice 15 LLMOpt | Slice 16 LLMOpt | Selected wide sites | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `12.584567 ms` | `11.730433 ms` | `55` | `7.993271 ms` | `1.467539x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `25.668025 ms` | `22.013426 ms` | `176` | `17.202084 ms` | `1.279695x` |

The dispatch counts remain `1098` for Qwen and `1044` for Gemma. Qwen's
median changes by `-0.854134 ms` (`-6.787154%`) and Gemma's by
`-3.654599 ms` (`-14.237944%`). Float16 logits change because reduction order
changes: Qwen mean/maximum absolute differences are
`0.001557522/0.011718750`, and Gemma's are
`0.009134801/0.072265625`; both two-row argmax sequences are unchanged.

# Validation

The OCaml and 49-test Python suites, ABI round trip, Metal compilation, both
package checks, zero-opaque audits, native 101-kernel primitive fixture, output
comparisons, and fresh `llama-bench -p 2 -n 0 -r 10` runs pass.

Evidence is the retained packages, outputs, source selection rule, and timing
receipt; attributing the full timing change to improved RMSNorm occupancy is
an inference supported by unchanged command/dispatch counts and the selected
kernel inventory.
