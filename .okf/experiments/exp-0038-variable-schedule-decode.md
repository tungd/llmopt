---
type: Experiment
title: 'Variable-length LFM schedules and repeated native decode'
description: 'Specialize captured LFM2.5 schedules to request lengths, execute three consecutive native decode steps, and preserve eager-Q8 greedy tokens.'
tags: [experiment, ocaml, metal, serving, scheduling, radix-cache, kv-cache, q8, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-24T00:52:42Z' }
sources:
  - id: specialization
    resource: /lib/serving_schedule.ml
    title: Target-specific schedule specialization
  - id: engine
    resource: /lib/serving_engine.ml
    title: Variable-length native serving coordinator
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Runtime-selected schedule execution
  - id: probe
    resource: /bin/lfm_serving_smoke.ml
    title: Repeated native decode probe
  - id: reference
    resource: /bench/lfm25_reference_tokens.py
    title: Bounded eager-Q8 reference probe
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-native-dynamic-parity-2026-08-24.txt
    title: Four-token native and eager evidence
  - id: prefill-package
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi8-dynamic-v1-2026-08-24/prefill/package.llmopt
    title: Captured prefill template package
  - id: decode-package
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi8-dynamic-v1-2026-08-24/decode/package.llmopt
    title: Captured decode template package
---

# Schedule transformation

`Serving_schedule.Lfm25` treats the captured six-token prefill and six-token
decode-past schedules as typed templates. It substitutes request sequence
dimensions and scalar bounds, then rebuilds every SSA output shape in
topological order from the transformed inputs and operation semantics. Static
tensor-archive inputs retain their declared shapes. Opaque operations are
rejected rather than copied into a native schedule.

The transformation covers dynamic matmul and Q8-linear dimensions, ranges,
pointwise integer bounds, normalized indices, the final three-token ShortConv
state, attention masks, cache update slices, and decode total length. The
prefill entry rejects requests shorter than that three-token recurrent window.
The
float32 matmul and linear Metal entries now receive `m`, `n`, and `k` as
fixed-width dispatch parameters instead of embedding captured dimensions in
MSL source.

`Serving_engine.prefill` specializes the prefill schedule to the actual token
array. Every decode step specializes the decode schedule to the matched radix
prefix, unpacks exactly that prefix's physical KV state and recurrent
checkpoint, appends one new physical slot, and inserts the extended sequence.

# Offline real-package evidence

The ABI-v8 packages retain 872 prefill commands, 926 decode commands, zero
opaque operations, and the shared 422,137,216-byte binary tensor archive.
Offline specialization and liveness planning completed for these request
shapes without opening a Metal device:

| Schedule | Specialized length | Workspace |
|---|---:|---:|
| Prefill | 13 | 2,500,608 bytes |
| Prefill | 128 | 23,482,368 bytes |
| Prefill | 4,096 | 759,300,096 bytes |
| Decode past | 1 | 198,656 bytes |
| Decode past | 127 | 2,269,952 bytes |
| Decode past | 4,095 | 71,371,520 bytes |

The Ninja-built OCaml test suite covers preservation of static dimensions,
dynamic Q8 and matmul dimensions, scalar and range rewriting, shape
re-inference, liveness planning, and malformed lengths. Xcode Metal compiled
the regenerated parameterized kernels. A separately memory-checked small
device probe dispatched the 129-command primitive package and retained 39/39
exact outputs.

# Native repeated-decode observation

System memory was 61% free with no Torch or model process before the one
180-second-supervised Apple M4 Pro launch. The native runtime mapped the binary
archive, executed one prefill plus three consecutive decode steps, and exited
normally:

```text
tokens: 19130,11040,11207,1414
load-seconds: 0.028177
prefill-seconds: 0.395947
decode-seconds: 0.344855
decode-steps: 3
prefill-kernels: 522
decode-kernels: 1632
decode-cached-prefixes: 6,7,8
radix-cached-tokens: 9
radix-hits: 3
radix-misses: 0
kv-used-tokens: 9
kv-used-checkpoints: 4
```

One separately memory-checked eager PyTorch MPS process loaded the local 350M
checkpoint, applied the same 92-module Q8 rewrite, and produced exactly
`19130,11040,11207,1414`. This establishes four-token greedy parity and proves
that runtime schedule specialization, growing physical KV state, recurrent
checkpoints, and radix ownership remain coherent across repeated decode.

# Boundary

The prefill template has offline shape and workspace evidence at 13, 128, and
4,096 tokens; this experiment did not launch those longer prefill schedules.
It also does not yet connect native tokenization and typed chat construction to
generation, expose an HTTP request loop, execute FP16 KV at model scale, run
needle retrieval, or record ERS.
