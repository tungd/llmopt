---
type: Experiment
title: 'Physical Q8 and FP16 Metal KV cache'
description: 'Allocate native token and recurrent-checkpoint pools and execute package-declared cache pack and unpack kernels.'
tags: [experiment, ocaml, metal, serving, kv-cache, q8, fp16]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T23:17:07Z' }
sources:
  - id: layout
    resource: /lib/kv_cache.ml
    title: Typed cache geometry and physical byte accounting
  - id: emitter
    resource: /lib/metal.ml
    title: Q8 and FP16 cache kernel emitter
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Native physical cache owner and dispatch
  - id: probe
    resource: /bin/ocaml_metal_smoke.ml
    title: Exact cache round-trip probe
  - id: prefill
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi7-cache-2026-08-24/prefill/package.llmopt
    title: Preserved prefill package with cache kernels
  - id: decode
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi7-cache-2026-08-24/decode/package.llmopt
    title: Preserved decode package with cache kernels
---

# Physical representation

`Kv_cache.Layout` retains LFM attention-layer, KV-head, head-dimension, and
recurrent-state geometry. FP16 stores two bytes per element. Grouped Q8 stores
signed int8 values followed by one FP16 scale per group; Q8 groups cannot cross
an attention head or recurrent-layer checkpoint. The default group size is 64.

For LFM2.5-350M this gives:

| Format | Bytes per token | Bytes per recurrent checkpoint |
|---|---:|---:|
| FP16 | 12,288 | 61,440 |
| Q8 group 64 | 6,336 | 31,680 |

`Metal_runtime.Cache` allocates token and checkpoint `MTLBuffer` pools from the
typed capacity configuration. It validates the serving package's declared
cache policy and resolves exact package entries for attention-key,
attention-value, and recurrent-checkpoint transfers.

# Generated kernels and package ABI

Package ABI v7 adds the typed `Cache` operation and retains reads for ABI v2
through v6. A serving compile adds eight entry points: pack and unpack for
attention and recurrent checkpoints in both FP16 and Q8. Q8 packing computes a
scale per 64-element group, rounds that scale to its stored FP16 representation,
and uses the same rounded value for quantization and later dequantization.

Offline binary recompilation of the preserved model graphs produced:

| Graph | Commands | Kernels | Opaque | Tensor bindings | Workspace |
|---|---:|---:|---:|---:|---:|
| Prefill | 872 | 46 | 0 | 241 | 1,153,792 bytes |
| Decode | 926 | 44 | 0 | 241 | 271,360 bytes |

Both ABI-v7 MSL programs compile and both packages validate against hard links
to the same 422,137,216-byte `weights.llmopt` archive. Neither recompile loaded
the model or opened a Metal device, and neither output directory contains JSON.

# Exact device probe

The first physical-cache probe exposed a probe-order defect: OCaml evaluated an
effectful list literal right-to-left, so unpack commands ran before pack
commands and observed zeroed pools. The probe now binds each dispatch with an
explicit sequential `let`; a future mismatch reports the first differing byte.

Before the corrected fixed launch, system-wide memory was 53% free and no model
or Torch process was resident. One Apple M4 Pro run executed the existing Q8
linear schedule and twelve cache dispatches:

```text
device: Apple M4 Pro
stage: serving
dispatch: ocaml-metal-schedule
kernel: llmopt_q8_linear
cache-formats: q8-group-64,f16
cache-dispatches: 12
q8-pools: 528 token bytes, 132 checkpoint bytes
f16-pools: 1024 token bytes, 256 checkpoint bytes
attention: exact
checkpoint: exact
```

Separate key and value segments occupied two non-contiguous token slots. Both
formats round-tripped those segments and one recurrent checkpoint exactly.

# Boundary

Physical allocation and format conversion now execute in native OCaml plus
Metal. The model schedule, radix-prefix leases, and physical pool are not yet
coordinated by one serving engine, so this experiment records no model token,
cache-hit, needle, latency, or ERS result.
