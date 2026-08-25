---
type: Experiment
title: 'Q8 linear-residual epilogue fusion'
description: 'Fuse all 32 same-shape Q8 projection and residual-add pairs in each LFM stage while retaining the materialized float16 operation order.'
tags: [experiment, compiler, ocaml, metal, q8, fusion, residual, prefill, decode, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T06:38:20Z' }
sources:
  - id: pass
    resource: /lib/passes.ml
    title: Alias and shape-safe Q8 residual rewrite
  - id: emitter
    resource: /lib/metal.ml
    title: Fused Q8 GEMM and GEMV residual epilogues
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Typed residual-buffer dispatch
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-linear-add-fusion-2026-08-25.txt
    title: Offline model replan and static parity fixture
---

# Optimization boundary

Every hybrid layer has two residual boundaries: the mixer output projection
and the feed-forward down projection. In both cases a Q8 result is materialized
to float16, read by a standalone add kernel, and written back to the workspace.
The preserved prefill and decode graphs each contain 32 such sole-consumer
pairs.

# Typed rewrite and semantics

`Passes.fuse_q8_add` accepts only an adjacent, same-shape, same-dtype tensor add
whose Q8 operand has no other consumer. Broadcast adds remain separate. The
fused operation appends the residual tensor to the original Q8 inputs and
retains the add node's output identity.

The float16 Metal epilogue applies an optional bias, explicitly rounds that Q8
result to `half`, then performs the residual addition. This preserves the
materialized Q8-linear then pointwise-add operation order. Float32 uses the
same typed family without the half conversion.

# ABI and lowering

Schedule ABI v10 adds `Q8_linear_add`; package ABI v10 adds distinct fused
GEMM/GEMV entry tags while retaining schedule v1-v9 and package v2-v9 reads.
LFM sequence specialization rewrites the fused `m`, `n`, and `k` dimensions.
The runtime validates that the residual metadata equals the output before
binding its extra Metal buffer.

# Saved-model structure

| Stage | Commands before | Commands after | Fused residuals | Remaining adds | Kernel entries |
|---|---:|---:|---:|---:|---:|
| Prefill | 856 | 824 | 32 | 15 | 56 |
| Decode | 910 | 878 | 32 | 15 | 54 |

Both packages remain zero-opaque, retain the sixteen Q8-SiLU epilogues from
the prior pass, validate all 241 bindings, and compile/link with Xcode Metal.
Materialized allocations fall from 506 to 474 for prefill and from 528 to 496
for decode. Workspace high-water remains 1,153,792 and 271,360 bytes because
other live intervals dominate it; unreused aligned bytes fall to 8,577,536 and
1,938,688.

The persistent artifact pair is
`_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi10-add-v1-2026-08-25`.
Both package stages hard-link inode `241927011`, and Q8-group-64 plus selectable
FP16 pair validation pass.

# Evidence boundary

The current 144-command small fixture compiles and statically compares fused
Q8+SiLU and Q8+residual outputs with their materialized references. It was not
launched after the one prior device attempt. No model or ERS run occurred; the
latest valid native ERS remains `0.11381808711306604`.
