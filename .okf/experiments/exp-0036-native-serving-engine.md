---
type: Experiment
title: 'Native LFM prefill, decode, and radix coordination'
description: 'Execute the complete fixed LFM2.5-350M Q8 schedules from OCaml while binding physical cache state to radix ownership.'
tags: [experiment, ocaml, metal, serving, radix-cache, kv-cache, q8, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-24T00:07:09Z' }
sources:
  - id: engine
    resource: /lib/serving_engine.ml
    title: Native prefill and decode coordinator
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Shared Metal context and binary archive loader
  - id: smoke
    resource: /bin/lfm_serving_smoke.ml
    title: Fixed native model probe
  - id: evidence
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi8-engine-2026-08-24/native-q8-smoke.txt
    title: Native Q8 probe output
  - id: reference-probe
    resource: /bench/lfm25_reference_tokens.py
    title: Eager Q8 token reference probe
  - id: reference-evidence
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi8-engine-2026-08-24/eager-q8-reference.txt
    title: Eager Q8 token and logits reference
  - id: prefill
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi8-engine-2026-08-24/prefill/package.llmopt
    title: ABI-v8 prefill package
  - id: decode
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi8-engine-2026-08-24/decode/package.llmopt
    title: ABI-v8 decode package
---

# Coordinator

`Serving_engine` validates the fixed LFM2.5 package pair before opening Metal.
It derives the six attention and ten recurrent bindings from
`Lfm25.Config.layer_types`, checks every runtime input and named output, and
requires the prefill length to equal the decode package's past length.

Prefill executes the complete binary schedule, reserves physical token slots
and one recurrent checkpoint, packs all attention and ShortConv state, and
inserts the token sequence into the mandatory radix cache. Decode obtains an
exact leased prefix, unpacks its state, executes one token, appends only the new
key/value position, stores the updated recurrent checkpoint, inserts the
extended prefix, and releases the lease. Reservation failures and pre-insert
execution failures release their logical ownership.

Package ABI v8 appends `source_items` and `source_offset` to the cache-transfer
parameters. This lets decode pack position six from each seven-position output
without rewriting the six shared radix-owned slots. ABI-v2 through ABI-v7
packages remain readable, while this coordinator requires ABI v8.

`Metal_runtime.load_packages` validates all packages first, opens one Metal
context, and reuses one mapped archive buffer when tensor-store paths resolve
to the same device and inode. The prefill and decode `weights.llmopt` paths in
this experiment are hard links to the same 422,137,216-byte binary archive.

# Static package evidence

The JSON-free ABI-v8 package pair validates for Q8-group-64 and FP16 policies:

| Stage | Commands | Kernel entries | Opaque | Tensor bindings | Workspace |
|---|---:|---:|---:|---:|---:|
| Prefill | 872 | 46 | 0 | 241 | 1,153,792 bytes |
| Decode | 926 | 44 | 0 | 241 | 271,360 bytes |

# Native Q8 observation

Before the one fixed launch, system-wide memory was 57% free, no Torch/model
process was resident, and both package paths had the same archive inode. The
180-second-supervised process completed normally in 1.5 seconds wall time;
memory was 56% free afterward.

```text
device: Apple M4 Pro
format: q8-group-64
input: 1,2,3,4,5,6
tokens: 19130,11040
load-seconds: 0.030475
prefill-seconds: 0.432272
decode-seconds: 0.119545
prefill-kernels: 522
decode-kernels: 544
decode-cached-prefix: 6
radix-cached-tokens: 7
radix-hits: 1
radix-misses: 0
kv-used-tokens: 7
kv-used-checkpoints: 2
```

One separately memory-checked eager PyTorch MPS run loaded the same local 350M
checkpoint, applied the same 92-module Q8 rewrite, and produced tokens
`19130,11040` for the same prefill/decode sequence. The native result therefore
has exact two-token greedy parity. The eager reference also records full-logit
SHA-256 values, but the native probe did not retain its logits, so exact logit
parity is not established.

These are single observations, not an ERS result. Together they prove that the
complete fixed schedules, physical grouped-Q8 cache, recurrent checkpoints,
and radix lease execute together in native OCaml plus Metal while preserving
the sampled tokens. They do not exercise FP16 physical storage at model scale,
tokenize text, generate beyond the fixed one-token decode shape, accept
benchmark requests, run needle retrieval, or produce ERS.
