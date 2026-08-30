---
type: Experiment
title: 'Single-consumer RMSNorm-residual fusion'
description: 'Fuse a captured RMSNorm and its sole residual-add consumer into one typed SIMD dispatch.'
tags: [experiment, compiler, fusion, metal, rmsnorm, residual, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T12:38:27+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-15-2026-08-30.json
    title: Slice 15 benchmark receipt
  - id: fusion
    resource: /lib/pass_fuse_rms_norm.ml
    title: RMSNorm-residual topology fusion
  - id: metal
    resource: /lib/metal.ml
    title: Fused RMSNorm-residual SIMD kernel
---

# Structural change

The RMSNorm pass now replaces a semantic RMSNorm and its sole tensor-add
consumer with `Rms_norm_add`. It requires producer identity, no graph-output
escape, equal captured tensor shapes, Float16 input/residual/output, and a
Float32 norm weight. The Metal tactic reduces and scales each final-axis row,
then adds the captured residual in the same dispatch.

Selection uses no model/tensor names, weight format, GGUF architecture, or
llama.cpp architecture ID. This fills the same general compiler vocabulary as
llama.cpp's `kernel_rms_norm_mul_add_f32`; the graph topology remains the
authority in LLMOpt.

# Full-model result

| Probe | Slice 14 LLMOpt | Slice 15 LLMOpt | Dispatch change | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `12.590408 ms` | `12.584567 ms` | `1098 -> 1098` | `8.1262295 ms` | `1.548635x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `26.156545 ms` | `25.668025 ms` | `1150 -> 1044` | `17.2815835 ms` | `1.485282x` |

Gemma selects 106 sites, removes 211 schedule commands and 106 dispatches,
and improves by `0.488520 ms`. Its Float16 logits change by mean absolute
`0.013817219` and maximum `0.109375`, while both row argmax IDs remain
`84904,148465`. Qwen selects zero sites and remains byte exact.

# Validation

The OCaml and 49-test Python suites, ABI round trip, Metal compilation, both
package checks, zero-opaque audits, full-model output comparisons, and fresh
`llama-bench -p 2 -n 0 -r 10` runs pass.

Evidence is the retained package, output, source comparison, and timing
records; latency attribution is an inference from Gemma's 106 selected sites
and Qwen's zero-site control.
