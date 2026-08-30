---
type: Experiment
title: 'Topology-fused L2 normalization'
description: 'Recover final-axis L2 normalization from captured producer/user topology and execute each row in one SIMD kernel.'
tags: [experiment, compiler, fusion, metal, l2-norm, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T12:05:08+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-12-2026-08-30.json
    title: Slice 12 benchmark receipt
  - id: fusion
    resource: /lib/pass_fuse_rms_norm.ml
    title: Normalization topology fusion
  - id: metal
    resource: /lib/metal.ml
    title: SIMD L2 normalization kernel
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Semantic L2 dispatch
---

# Structural change

The normalization pass now follows graph producers backward from a final
multiply and recognizes `mul(x,x) -> sum(last,keepdim) -> add(epsilon) ->
rsqrt -> mul(x,inverse)`. It requires each intermediate to have one consumer,
rejects graph-exposed intermediates, and checks shape, dtype, axis, and epsilon.
Interleaved independent branches are therefore matched by topology rather than
node adjacency.

The replacement is a typed `L2_norm` primitive serialized in the serving
schedule and lowered to one SIMD-group Metal reduction per row. Neither the
pass nor the runtime reads a model name, tensor name, GGUF architecture, or
llama.cpp architecture ID.

# Full-model result

| Probe | Slice 11 LLMOpt | Slice 12 LLMOpt | Dispatch change | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `13.383031 ms` | `12.558937 ms` | `1272 -> 1128` | `7.931209 ms` | `1.583483x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `28.946519 ms` | `28.671980 ms` | `1635 -> 1635` | `17.3268545 ms` | `1.654771x` |

All 36 Qwen normalization chains become 36 semantic dispatches, eliminating
144 dispatches. Qwen changes from slice 11 by mean absolute `0.0051494651` and
maximum `0.0454711914`, while both row argmax IDs remain `760,16`. Gemma has no
matching topology, remains byte exact with slice 11, and preserves argmax IDs
`84904,148465`; its timing change is an observation rather than an attributed
effect.

# Validation

The complete OCaml and Python suites, Xcode Metal compilation, native Metal
fixture, schedule serialization regression, both package checks, zero-opaque
audits, full-model output comparisons, and fresh
`llama-bench -p 2 -n 0 -r 10` runs pass.

Evidence is the compiled package inventories and retained timing/output
records; the dispatch-overhead attribution is an inference from the captured
five-operation chain becoming one semantic dispatch.
