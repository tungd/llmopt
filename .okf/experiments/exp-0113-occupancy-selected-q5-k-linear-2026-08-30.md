---
type: Experiment
title: 'Occupancy-selected full-SIMD Q5_K Linear tactic'
description: 'Use full-lane Q5_K execution only for the captured four-superblock inner dimension where lane cohorts remain balanced.'
tags: [experiment, compiler, tactics, metal, q5-k, occupancy, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T11:23:30+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-11-2026-08-30.json
    title: Slice 11 benchmark receipt
  - id: metal
    resource: /lib/metal.ml
    title: Q5_K tactic registry and Metal implementation
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Ordered tactic fallback
---

# Structural change

The full-lane Q5_K implementation partitions four physical superblocks across
four SIMD lane cohorts and computes one output column per SIMD group. Its
tactic predicate requires captured `m=2`, Q5_K storage, Float16 input/output,
SIMD32 support, and exactly four 256-element superblocks (`k=1024`). Other
inner dimensions keep the prior four-column sub-SIMD tactic.

This is a cost/occupancy predicate over graph shape and physical quant layout.
It reads no model name, tensor name, GGUF architecture, or llama.cpp
architecture ID.

# Controlled selection

Selecting full-SIMD Q5_K at every aligned block count measured Qwen at
`13.445497 ms` but Gemma at `29.587030 ms`. Adding the analogous Q6_K candidate
measured `14.234543 ms` and `29.796600 ms`. The final four-superblock predicate
selects the new kernel for Qwen's `k=1024` Q5_K Linears and zero Gemma sites;
the Q6_K candidate is not retained.

# Full-model result

| Probe | Slice 10 LLMOpt | Slice 11 LLMOpt | Change | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `14.441490 ms` | `13.383031 ms` | `-1.058459 ms` (`-7.329292%`) | `8.0035625 ms` | `1.672134x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `28.633475 ms` | `28.946519 ms` | `+0.313044 ms` (`+1.093280%`) | `17.6487085 ms` | `1.640149x` |

Qwen changes from slice 10 by mean absolute `0.0032285259` and maximum
`0.0234375`, while row argmax IDs remain `760,16`. Gemma selects no new Q5_K
site and its output remains byte exact with row argmax IDs `84904,148465`;
its timing change is an observation, not attributed to changed execution.

# Validation

The complete OCaml suite, all 49 Python tests, Xcode Metal compilation, native
454-dispatch fixture, both package checks, zero-opaque audits, full-model output
comparisons, and fresh `llama-bench -p 2 -n 0 -r 10` runs pass. A tactic
regression covers both the selected four-block geometry and the unselected
six-block fallback.

Evidence is the shape-selected package kernel inventory, native outputs, and
retained timing samples; the performance attribution is an inference from the
single Qwen execution change and Gemma's zero selected sites.
