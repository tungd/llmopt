---
type: Experiment
title: 'Full-SIMD two-column Q4_K Linear tactic'
description: 'Replace the two-row Q4_K sub-SIMD tactic with a full-lane two-column implementation selected from captured geometry and physical layout.'
tags: [experiment, compiler, tactics, metal, q4-k, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T11:14:18+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-10-2026-08-30.json
    title: Slice 10 benchmark receipt
  - id: metal
    resource: /lib/metal.ml
    title: Q4_K tactic registry and Metal implementation
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Shape-derived tactic dispatch
---

# Structural change

The previous `m=2` Q4_K tactic divided one SIMD group into four independent
eight-lane reductions. The new tactic assigns all 32 lanes to two output
columns for one input row, partitions Q4_K superblocks across four lane
quarters, and uses packed 16-bit quant and scale extraction before one
SIMD-wide reduction. Two SIMD groups share a threadgroup.

Selection consumes only the captured `m=2` Linear shape, Q4_K block layout,
Float16 input/output dtypes, and SIMD32 target capability. It reads no model
name, tensor name, GGUF architecture, or llama.cpp architecture ID. The prior
four-column, paired-row, and generic kernels remain fallback tactics.

# Full-model result

| Probe | Slice 9 LLMOpt | Slice 10 LLMOpt | Change | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `14.506459 ms` | `14.441490 ms` | `-0.064969 ms` (`-0.447863%`) | `7.6178545 ms` | `1.895742x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `30.664444 ms` | `28.633475 ms` | `-2.030969 ms` (`-6.623205%`) | `16.6469585 ms` | `1.720042x` |

Command and dispatch counts stay at 2,723/1,272 for Qwen and 3,997/1,635
for Gemma. Qwen changes from slice 9 by mean absolute `0.0034628396` and
maximum `0.025390625`; Gemma changes by `0.0120636337` and `0.078125`.
Both retain their two previous row argmax IDs.

# Validation

The complete OCaml suite, all 49 Python tests, Xcode Metal compilation, native
454-dispatch fixture, both package checks, zero-opaque audits, full-model output
comparisons, and fresh `llama-bench -p 2 -n 0 -r 10` runs pass. The receipt
retains every measured sample.

Evidence is the tactic-selection regression, valid typed package inventories,
native output comparisons, and timing samples; attributing Gemma's reduction
to the new Q4_K implementation is an inference from the only executable graph
change.
