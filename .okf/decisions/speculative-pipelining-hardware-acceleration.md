---
type: Decision
title: 'Hardware Bitfield and Speculative Runtime Primitives'
description: 'Retains the implemented bitfield, Split-K, tree-attention, and speculative-cache primitives while retracting the unsupported end-to-end pipelining claim.'
tags: [decision, compiler, speculative-decoding, bitfield-extract, split-k, apple-silicon, metal]
status: deprecated
generated: { by: 'process:codex', at: '2026-08-31T15:30:00+07:00' }
sources:
  - id: local-metal
    resource: /lib/metal.ml
    title: Metal Shading Language generator and registered kernels
  - id: local-kernel-abi
    resource: /lib/kernel_abi.ml
    title: Shape-aware Linear tactic registry
  - id: local-serving-cache
    resource: /lib/serving_cache.ml
    title: Speculative cache metadata primitives
  - id: gemma-mtp-contract
    resource: /decisions/gemma4-mtp-runtime-contract.md
    title: Corrected Gemma 4 MTP executable contract
---

# Correction

The original decision combined several implemented primitives with an
unimplemented end-to-end speculative serving path and assigned it an unsupported
throughput expectation. The latter claims are retracted.

# Retained implementation evidence

- Superblock K-quant shaders use `metal::extract_bits` at the inspected unpack
  sites.
- Wide Linear tactics include four-way SIMDgroup, intra-threadgroup Split-K
  variants. They are not cross-threadgroup reductions.
- A compact-bitmask tree-attention kernel and runtime dispatch are registered.
  Registration and isolated tests do not prove target multi-token verification.
- Radix and serving cache modules provide tested reservation, partial commit,
  and rollback metadata primitives. They are not connected to physical Gemma
  MTP execution.

# Retracted claims

- `Serving_queue` never transitioned requests through the claimed speculative
  states, and its `2.5x` score multiplier had no measured basis.
- The removed `Serving_engine.Speculative_pipeline` treated the Gemma assistant
  as an independent token-ID model, used incorrect verification indexing, and
  did not implement asynchronous Metal execution.
- `bench/reproduce_slice31.py` measures static two-token full forwards. It is
  not a speculative decoding benchmark and cannot validate MTP throughput.

The replacement contract and remaining implementation sequence are defined in
[Gemma 4 MTP Requires a Target-Coupled Model Program Contract](gemma4-mtp-runtime-contract.md).
