---
type: Experiment
title: 'SIMD-group batched matmul tactic'
description: 'Replace scalar Float32 batched matmul with captured-shape-selected SIMD matrix tiles and an unaligned tiled fallback.'
tags: [experiment, compiler, metal, tactics, batched-matmul, simdgroup, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T04:16:26+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-7-2026-08-30.json
    title: Slice 7 benchmark receipt
  - id: metal
    resource: /lib/metal.ml
    title: SIMD8 and tiled batched-matmul implementations
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Captured-shape tactic selection and launch geometry
---

# Diagnosis

A controlled generated-package differential replaces selected kernel families
with no-op bodies while retaining the same 2,407 dispatches. Against Qwen's
`49.884439 ms` intact median, no-op batched matmul measures `37.662983 ms`, a
`12.221456 ms` differential across 145 sites. No-op all fine-grained primitives
measures `21.782517 ms`; movement is `41.793108 ms`, elementwise is
`43.432474 ms`, construction is `42.273998 ms`, and cast is `45.843482 ms`.
These diagnostics overlap and are not additive.

# Compiler change

The Metal backend now selects an `8x8` SIMD-group matrix tactic for aligned
Float32 `Batched_matmul` shapes. One 32-lane SIMD group owns each output tile,
and the kernel retains broadcast batch-coordinate mapping from the semantic
operation. Unaligned dimensions select a `16x16` threadgroup-tiled fallback.

Selection uses the typed operation, captured `m/n/k`, batch broadcast geometry,
and target SIMD width. It does not read a model name, tensor name, GGUF
architecture, or llama.cpp architecture ID. Qwen selects the SIMD tactic at 144
sites and the fallback once. Gemma contains no `Batched_matmul` site.

# Full-model result

| Probe | Slice 6 LLMOpt | Slice 7 LLMOpt | Change | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `49.884439 ms` | `43.148398 ms` | `-6.736041 ms` (`-13.503291%`) | `7.903979 ms` | `5.459073x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `42.235494 ms` | `39.534926 ms` | observed `-2.700568 ms`; no matching operation | `17.2000205 ms` | `2.298539x` |

Both outputs are byte exact to slice 6. Command and dispatch counts are
unchanged because the optimization replaces implementation tactics below the
semantic schedule. Gemma's timing change is not attributed to this slice.

# Validation

The native fixture selects and executes both the aligned SIMD8 tactic and the
unaligned tiled fallback, with exact expected outputs. The OCaml suite, all 49
Python tests, Xcode Metal compilation, full native 454-dispatch fixture, and
zero-opaque package checks pass. Fresh llama.cpp runs use
`llama-bench -p 2 -n 0 -r 10`; the receipt retains every sample.

Evidence is the family differential, typed dispatch selection, native tactic
coverage, full-model byte comparison, and fresh benchmark receipt; attributing
Qwen's measured reduction to the batched-matmul tactic is an inference from the
only executable change.
