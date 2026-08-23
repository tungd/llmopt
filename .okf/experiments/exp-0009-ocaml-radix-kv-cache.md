---
type: Experiment
title: 'OCaml radix and KV serving cache'
description: 'Implement the serving-owned prefix cache and selectable FP16/Q8 KV storage policy for the future OCaml Metal runtime.'
tags: [experiment, ocaml, serving, radix-cache, kv-cache, q8, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T16:03:35Z' }
sources:
  - id: kv-cache
    resource: /lib/kv_cache.ml
    title: KV format, accounting, and slot allocator
  - id: radix-cache
    resource: /lib/radix_cache.ml
    title: Compressed radix prefix cache
  - id: serving-cache
    resource: /lib/serving_cache.ml
    title: Serving cache owner and LFM2.5 configuration
  - id: tests
    resource: /test/test.ml
    title: OCaml cache behavior tests
  - id: sglang-radix
    resource: https://github.com/sgl-project/sglang/blob/d1af3c89233c475fc1bf11939d86787e6cddd58c/python/sglang/srt/mem_cache/radix_cache.py
    title: SGLang RadixCache reference revision
  - id: sglang-mamba-radix
    resource: https://github.com/sgl-project/sglang/blob/d1af3c89233c475fc1bf11939d86787e6cddd58c/python/sglang/srt/mem_cache/mamba_radix_cache.py
    title: SGLang hybrid recurrent-cache reference revision
  - id: sglang-license
    resource: https://github.com/sgl-project/sglang/blob/d1af3c89233c475fc1bf11939d86787e6cddd58c/LICENSE
    title: SGLang Apache 2.0 license
---

# Question

Can the serving-side prefix and KV-cache ownership be implemented natively in
OCaml while retaining SGLang's compressed-radix reuse and the checkpoint rule
required by LFM2.5's hybrid attention/ShortConv state?

# Design

`Radix_cache` stores token sequences as compressed edges with one KV slot per
edge token. Exact and branching inserts retain canonical slots, while redundant
new allocations are returned to the owning allocator. A prefix match returns a
lease that protects its path from LRU leaf eviction until the scheduler releases
it. Namespaces isolate otherwise identical token sequences, and page-sized
child keys preserve branches that diverge within later pages.

The hybrid rule follows SGLang's recurrent-cache specialization: recurrent
state exists only at materialized node endpoints. Splitting a compressed edge
does not manufacture a checkpoint, and lookup therefore returns the deepest
matched prefix for which a valid checkpoint exists. The cache structure is
generic over slot/checkpoint identifiers; `Serving_cache` binds both to the
owned `Kv_cache` allocator.[^sglang-radix] [^sglang-mamba-radix]

The mutable tree and allocator are abstract behind module interfaces and have
one owner: the OCaml serving scheduler. This keeps mutation local while leases
make active-prefix lifetime explicit.

# KV format

`Kv_cache.Format` supports FP16 and grouped Q8. Q8 accounts for one signed byte
per value and one FP16 scale per group. With the LFM2.5-350M defaults (6 GQA
layers, 8 KV heads, head dimension 64, and 10 recurrent ShortConv layers), the
calculated storage is:

| Format | Bytes per token KV | Bytes per recurrent checkpoint |
|---|---:|---:|
| FP16 | 12,288 | 61,440 |
| Q8, group size 64 | 6,336 | 31,680 |

Q8 is the default serving policy, but FP16 remains an explicit configuration.
These values are layout and capacity accounting; this slice does not yet bind
the slot identifiers to Metal buffers or implement device quantization.

# Verification

```sh
ninja -f ninja.build ocaml-test
```

The tests cover both byte layouts, branch ownership, exact prefix reuse,
checkpoint-safe edge splitting, insertion at an internal checkpoint,
namespace isolation, lock-protected eviction, allocator reclamation, and
two-token page keys.

# Boundary

The radix/KV ownership layer is implemented in OCaml and is not configurable
away. The current executable Metal path still loads generated libraries through
Python and the PyTorch MPS bridge. Moving package loading, physical KV buffers,
Q8 quantize/dequantize operations, command submission, and the serving loop
into OCaml is the next runtime integration boundary.

[^sglang-radix]: SGLang `RadixCache`, pinned to revision `d1af3c89233c475fc1bf11939d86787e6cddd58c`.
[^sglang-mamba-radix]: SGLang `MambaRadixCache`, pinned to the same revision.
