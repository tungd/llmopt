---
type: Experiment
title: 'Shared LFM prefill/decode capture and recurrent state lowering'
description: 'Capture fixed use-cache prefill and one-token decode graphs once, share one binary tensor archive, and type their recurrent cache operations in schedule v8.'
tags: [experiment, lfm25, fx, ocaml, schedule, metal, prefill, decode, kv-cache]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T20:30:25Z' }
sources:
  - id: capture-session
    resource: /python/llmopt_backend/__init__.py
    title: Multi-graph capture session and static alias canonicalization
  - id: capture-probe
    resource: /bench/lfm25_capture_decode.py
    title: Bounded 350M prefill/decode capture probe
  - id: planner
    resource: /lib/fx_plan.ml
    title: Recurrent prefill/decode FX lowering
  - id: schedule
    resource: /lib/serving_schedule.ml
    title: Schedule-v8 validation and codec
  - id: prefill-package
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-replan-v8-2026-08-24/prefill/package.llmopt
    title: Offline-replanned prefill package
  - id: decode-package
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-replan-v8-2026-08-24/decode/package.llmopt
    title: Offline-replanned decode package
---

# Capture

Before launch, `memory_pressure -Q` reported 58% system-wide memory free, the
local `LiquidAI/LFM2.5-350M` checkpoint was complete, and no competing model
process was present. The one approved attempt loaded the Q8-transformed model
and captured both Dynamo specializations:

| Graph | FX nodes | Runtime inputs | Static tensors | Outputs |
|---|---:|---:|---:|---:|
| six-token prefill | 1,155 | 1 | 241 | 23 |
| one-token decode | 1,195 | 23 | 241 | 13 |

The decode inputs comprise token IDs, ten `[1,1024,3]` float16 ShortConv
states, and six key/value pairs with `[1,8,6,64]` float16 shape. The prefill
outputs contain the ten recurrent states, six key/value pairs, and logits; the
decode outputs contain six updated key/value pairs and logits while recurrent
states are updated by explicit copies.

The capture session writes the first graph's static tensors once. Later graphs
are permitted to use a storage-identical subset; their binding names are
canonicalized to the sealed archive keys. The root, prefill, and decode paths
were hard links to inode `241927011`, so the 422,137,216-byte archive occupied
one physical copy during capture.

# Failure and correction

The attempt reached both compiler invocations and then failed in its
post-capture package checker because no `kernel.metallib` had been produced.
The graph began with a supported non-Q8 primary operation, and `Metal.lower`
therefore omitted the 92 Q8-linear kernels even though they were present in the
same graph. Q8 source and ABI entries are now additive components. A synthetic
mixed matmul/Q8 test proves both families are emitted together.

Eager and compiled parity comparisons ran in the process before this failure,
but the script version wrote its result only after package validation. No
result file survived, so this experiment makes no retained parity claim. The
model/device attempt was not repeated.

# Schedule-v8 lowering

The preserved manifests exposed the cache vocabulary precisely:

* prefill: ten pure-crop pads, ten float16 `zeros_like` fills, ten state copies,
  twelve empty tensor literals, and twelve concatenations using those empty
  literals as identities;
* decode: ten rolls, ten trailing-slice updates, ten state copies, and ten
  trailing-axis sums.

Empty tensor literals do not become invalid zero-sized tile values; the
frontend records them as compile-time concat identities. Slice assignment is
functional SSA (`Update_slice`), followed by an explicit ordered copy into the
runtime-owned recurrent state. Roll, update, copy, and sum have exact CPU
reference coverage and survive the version-8 binary schedule round trip.

# Offline evidence

No model load or device dispatch was used for the corrected replans:

```text
planned 1155 FX nodes into 872 IR nodes
valid serving package: 14 kernels, 872 commands, 0 opaque, tensor-store=241

planned 1195 FX nodes into 926 IR nodes
valid serving package: 10 kernels, 926 commands, 0 opaque, tensor-store=241
```

Both generated MSL files compile with `xcrun metal`, link with `xcrun
metallib`, and reference hard links to the same binary archive. The full Ninja
suite passes 29 Python tests plus OCaml, LLVM, Metal, and package checks.

# Boundary

This proves capture structure, one-copy static storage, typed IR/schedule
coverage, Metal source compilation, and package/archive validation for the
fixed prefill/decode specializations. It does not prove package execution,
physical KV allocation or quantization, tokenizer/generation behavior, retained
logit/token parity, radix-cache reuse, needle retrieval, latency, or ERS.
