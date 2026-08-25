---
type: Experiment
title: 'Q8 multiplied-input down-projection fusion'
description: 'Absorb the materialized SwiGLU product into each Q8 down projection while preserving float16 multiplication and residual-add rounding.'
tags: [experiment, compiler, ocaml, metal, q8, fusion, swiglu, residual, prefill, decode, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T06:57:09Z' }
sources:
  - id: pass
    resource: /lib/passes.ml
    title: Alias-safe multiplied-input rewrite
  - id: emitter
    resource: /lib/metal.ml
    title: Product-loading Q8 GEMM and GEMV kernels
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Typed dual-input and residual dispatch
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-multiplied-input-fusion-2026-08-25.txt
    title: Offline model replan and static fixture evidence
---

# Optimization boundary

Each of the sixteen feed-forward layers materializes a float16 product between
the SiLU gate projection and the parallel up projection. The following Q8 down
projection reads that product once and immediately adds the layer residual.

`Passes.fuse_q8_mul_add` removes only the product node. The two upstream Q8
projections remain independent, while `Q8_linear_mul_add` receives their
float16 outputs directly, multiplies each pair while loading the reduction
tile, performs the Q8 down projection, rounds its result to float16, and adds
the residual. A second consumer of the product prevents the rewrite.

# ABI and lowering

Schedule/package ABI v11 carries the typed operation while retaining schedule
v1-v10 and package v2-v10 reads. The Metal ABI binds left input, right input,
Q8 weight, scale, optional bias, residual, output, and parameters in that
order. Separate tiled GEMM and one-row GEMV entries cover prefill and decode;
the runtime validates equal input metadata before dispatch.

# Saved-model structure

| Stage | Commands before | Commands after | Fused products | Standalone mul before/after | Kernel entries |
|---|---:|---:|---:|---:|---:|
| Prefill | 824 | 808 | 16 | 62 / 46 | 60 |
| Decode | 878 | 862 | 16 | 72 / 56 | 58 |

Both packages remain zero-opaque and validate all 241 tensor bindings against
the same 422,137,216-byte binary tensor archive. Materialized allocations fall
from 474 to 458 for prefill and from 496 to 480 for decode. Workspace high-water
falls from 1,153,792 to 1,098,496 bytes for the captured prefill template and
from 271,360 to 262,144 bytes for decode.

The persistent artifact pair is
`_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi11-mul-add-v1-2026-08-25`.
Both stages use archive inode `241927011`; Q8-group-64 and selectable FP16
package-pair validation pass, and both generated metallibs compile.

# Evidence boundary

The 153-command fixture contains materialized and fused multiplied-input paths
and byte-exact output assertions. Its 57-entry MSL and ABI-v11 package compile
and validate, but the executable was not launched. No model request, token,
cache, needle, latency, or ERS measurement ran; the latest valid native ERS
remains `0.11381808711306604`.
