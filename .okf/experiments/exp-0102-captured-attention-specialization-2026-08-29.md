---
type: Experiment
title: 'Captured-shape SIMD attention specialization'
description: 'Graph-driven Metal lowering specializes fused attention for captured head widths and records full-forward changes for Gemma and Qwen against their prior packages and llama.cpp.'
tags: [experiment, fx, compiler, metal, attention, gguf, ud, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-29T19:10:17+07:00' }
sources:
  - id: receipt
    resource: /bench/results/gguf-attention-specialization-2026-08-29.json
    title: Captured-attention tuning receipt
  - id: metal
    resource: /lib/metal.ml
    title: Graph-driven Metal source and ABI lowering
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Schedule-derived attention dispatch
  - id: baseline
    resource: /bench/results/gguf-full-model-comparison-2026-08-29.json
    title: Same-day llama.cpp full-forward receipt
---

# Diagnosis

`Ir.Primitive.Attention` already represents fused scaled dot-product attention.
The performance defect was below that fusion boundary: the Metal backend had a
SIMD online-softmax kernel only for head dimension 64. Captured Gemma attention
uses dimensions 256 and 512, so all 35 attention commands selected the scalar
kernel, which recomputed each query-key score for every output dimension.

A controlled Gemma no-op differential measured `527.723074 ms` for the intact
package and `68.799019 ms` with attention dispatches replaced by no-ops, an
attention-family delta of `458.924055 ms`. Replacing every Linear family with
no-ops measured `474.267960 ms`, a `53.455114 ms` delta. These dispatch-family
deltas are diagnostic observations and are not additive.

# Compiler change

Metal lowering now reads the final logical dimension from every captured
float16 attention query, sorts the unique SIMD-compatible dimensions, and emits
one fixed-width SIMD online-softmax kernel plus ABI entry per observed width.
Gemma therefore packages `h256` and `h512`; Qwen packages `h256`. The runtime
forms the exact kernel name from the schedule's head dimension and falls back to
the scalar kernel only when the package does not declare that specialization.

No model name or GGUF architecture identifier participates in this path. FX
shape metadata remains the specialization authority, and the fixed chunk count
lets Metal compile the query and result arrays and their loops for the captured
width.

# Full-forward result

The controlled sequence ran each baseline package immediately before its new
package with three warmups and ten measured two-token, no-cache forwards:

| Probe | Baseline | Captured-width SIMD | Change | llama.cpp context | New ratio |
|---|---:|---:|---:|---:|---:|
| Gemma-4-E2B-it UD-Q4_K_XL | `483.045578 ms` | `56.972980 ms` | `8.4785x`; `-426.072598 ms` | `17.8397915 ms` | `3.1936x` |
| Qwen3.5-0.8B UD-Q4_K_XL | `177.618027 ms` | `129.416585 ms` | `1.3725x`; `-48.201442 ms` | `7.4071875 ms` | `17.4718x` |

The llama.cpp medians are the same-day values from the full-model receipt; they
were not rerun in this controlled baseline/specialized sequence. Command and
dispatch counts do not change: Gemma remains at 3,999 commands and 1,637 native
dispatches, while Qwen remains at 19,176 commands and 9,193 dispatches.

# Numerical observation

Both specialized outputs are fully finite and preserve their baseline argmax
IDs. Gemma has 341,613/524,288 elements bit exact to the scalar-attention
package, maximum absolute difference `0.0732421875`, and mean absolute
difference `0.00503994`; both produce argmax IDs `84904,148465`. Qwen has
264,492/496,640 exact elements, maximum absolute difference `0.029296875`, and
mean absolute difference `0.00300820`; both produce argmax IDs `1,458`.

The native primitive package separately executes generated `h256` attention
with exact expected output. Both full packages compile with Xcode Metal,
validate with zero opaque commands, and execute through the native OCaml
runtime.

# Remaining profile

Evidence is the controlled package timing, no-op differential, and binary-logit
comparison. The interpretation is that attention was Gemma's dominant backend
kernel defect, while Qwen's smaller gain and unchanged 9,193-dispatch schedule
leave its captured unrolled gated-delta recurrence as the larger remaining
compiler target.
