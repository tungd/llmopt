---
type: Experiment
title: 'Token-major Gated Delta input absorption'
description: 'Feed captured token-major tensors directly into semantic Gated Delta and remove exclusive layout-conversion chains.'
tags: [experiment, compiler, fusion, layout, metal, gated-delta, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T13:16:15+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-19-2026-08-30.json
    title: Slice 19 benchmark receipt
  - id: fusion
    resource: /lib/pass_fuse_gated_delta.ml
    title: Exclusive layout-chain absorption
  - id: metal
    resource: /lib/metal.ml
    title: Token-major mixed-dtype Gated Delta tactic
  - id: schedule
    resource: /lib/serving_schedule.ml
    title: Typed schedule contract
---

# Structural change

The existing semantic `Gated_delta` operation now absorbs five exclusive
`transpose(1,2) -> contiguous -> cast(f32)` producer chains when captured
shapes prove that they only convert token-major inputs for that recurrence.
The direct kernel accepts Float16 query/key/value/beta tensors, retains the
Float32 gate and recurrence state, and writes the same token-major Float16
output. Selection uses producer/user topology, shape, dtype, and SIMD width;
it contains no model, tensor, or architecture identifiers.

# Full-model result

| Probe | Slice 18 LLMOpt | Slice 19 LLMOpt | Dispatch change | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `11.500001 ms` | `10.674000 ms` | `1050 -> 888` | `7.879479 ms` | `1.354658x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `21.452665 ms` | `21.584868 ms` | `974 -> 974` | `17.4471875 ms` | `1.237155x` |

All 18 Qwen recurrence sites match. The package removes 324 schedule commands
and 162 Metal dispatches, and its median changes by `-0.826001 ms`. Gemma has
no matching recurrence and its fresh observation changes by `+0.132203 ms`.
Both outputs are byte exact with slice 18 and preserve both row argmax IDs.

# Validation

The OCaml and 49-test Python suites, typed schedule round trip, generated Metal
compilation, both package checks, zero-opaque audits, native 101-kernel
primitive fixture, output comparisons, and fresh
`llama-bench -p 2 -n 0 -r 10` runs pass.

Evidence is the retained packages, outputs, tests, and timing receipt. The
Qwen latency reduction is measured; the compiler rule's applicability to
other captured graphs is structural, while their performance effect remains
an assumption until measured.
