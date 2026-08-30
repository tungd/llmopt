---
type: Experiment
title: 'Topology-fused Q4_K gated Linear'
description: 'Fuse independent same-layout gate/up Linear branches and their sole activation-product consumer into one semantic operation.'
tags: [experiment, compiler, fusion, tactics, metal, linear, q4-k, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T12:58:54+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-17-2026-08-30.json
    title: Slice 17 benchmark receipt
  - id: fusion
    resource: /lib/pass_fuse_swiglu_ffn.ml
    title: Gated Linear topology fusion
  - id: metal
    resource: /lib/metal.ml
    title: Q4_K gated Linear tactics
---

# Structural change

The compiler now replaces two unbiased Linear projections with a shared input
and one sole gated-activation product consumer with `Gated_linear`. The rule
requires producer and sole-consumer identity, no graph-output escape, matching
captured `m/n/k`, compatible Float16 values, and equal supported weight
layouts. The semantic operation carries SiLU, GELU, or sigmoid activation.

The first executable tactic covers equal Q4_K weights. One 32-lane SIMD group
computes both projections for one captured row/column, using the same
eight-lane-per-superblock decomposition as the established short-row Q4_K
Linear tactic. Selection uses no model/tensor names or GGUF architecture ID.
Schedule ABI v25 and package ABI v23 carry the general operation; older ABI
reads are intentionally not retained.

# Full-model result

| Probe | Slice 16 LLMOpt | Slice 17 LLMOpt | Sites / dispatch change | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `11.730433 ms` | `11.627913 ms` | `14 / -28` | `7.911646 ms` | `1.469721x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `22.013426 ms` | `21.449566 ms` | `30 / -60` | `17.350833 ms` | `1.236227x` |

Qwen removes 42 schedule commands and changes by `-0.102520 ms`; Gemma
removes 90 commands and changes by `-0.563860 ms`. Float16 logit mean/maximum
absolute differences versus slice 16 are `0.002864912/0.019531250` and
`0.014898818/0.099670410`; both two-row argmax sequences are unchanged.

A controlled full-SIMD-per-row/column tactic was rejected after measuring
`12.676477 ms` for Qwen and `31.209588 ms` for Gemma. It preserved the
semantic fusion but discarded the faster sub-SIMD Q4_K execution structure.

# Validation

The OCaml and 49-test Python suites, ABI round trip, Metal compilation, both
package checks, zero-opaque audits, native 101-kernel primitive fixture, output
comparisons, and fresh `llama-bench -p 2 -n 0 -r 10` runs pass.

Evidence is the retained packages, source topology rule and tactic, controlled
rejection, outputs, and timing receipt; attributing the retained delta to
projection fusion is an inference supported by the exact selected-site and
dispatch changes.
