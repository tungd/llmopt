---
type: Experiment
title: 'Gated Linear layout extension'
description: 'Extend the topology-selected gated Linear operation from Q4_K to Q5_K and IQ4_XS tactics.'
tags: [experiment, compiler, fusion, tactics, metal, linear, q5-k, iq4-xs, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T13:05:03+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-18-2026-08-30.json
    title: Slice 18 benchmark receipt
  - id: fusion
    resource: /lib/pass_fuse_swiglu_ffn.ml
    title: Layout capability selection
  - id: metal
    resource: /lib/metal.ml
    title: Q5_K and IQ4_XS gated Linear tactics
---

# Structural change

The semantic `Gated_linear` operation now accepts captured two-row Q5_K and
IQ4_XS gate/up pairs in addition to Q4_K. The unchanged fusion rule requires a
shared input, sole gated-product consumer, no graph-output escape, matching
shapes, and equal supported layouts. Q5_K retains its four-lane superblock
distribution; IQ4_XS shares activation loads across its two projections and
two rows. No model/tensor names or architecture identifiers participate.

# Full-model result

| Probe | Slice 17 LLMOpt | Slice 18 LLMOpt | Added sites / dispatch change | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `11.627913 ms` | `11.500001 ms` | `10 / -20` | `7.829771 ms` | `1.468753x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `21.449566 ms` | `21.452665 ms` | `5 / -10` | `17.1705835 ms` | `1.249385x` |

Qwen removes another 30 commands and changes by `-0.127912 ms`. Gemma
removes 15 commands and changes by `+0.003099 ms`. Float16 logit mean/maximum
absolute differences versus slice 17 are `0.005456710/0.037109375` and
`0.008841064/0.067382812`; both two-row argmax sequences are unchanged.

# Validation

The OCaml and 49-test Python suites, ABI round trip, Metal compilation, both
package checks, zero-opaque audits, native 101-kernel primitive fixture, output
comparisons, and fresh `llama-bench -p 2 -n 0 -r 10` runs pass.

Evidence is the retained packages, kernel inventory, outputs, and timing
receipt. The Qwen improvement and flat Gemma observation are reported
separately rather than treating dispatch reduction as a latency proxy.
