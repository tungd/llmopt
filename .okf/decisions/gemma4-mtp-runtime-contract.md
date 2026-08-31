---
type: Decision
title: 'Gemma 4 MTP Requires a Target-Coupled Model Program Contract'
description: 'Defines the executable target, cache, assistant, proposal, verification, and acceptance contract required for Gemma 4 MTP in LLMOpt.'
tags: [decision, gemma4, mtp, speculative-decoding, model-program, kv-cache]
status: stable
generated: { by: 'process:codex', at: '2026-08-31T15:30:00+07:00' }
sources:
  - id: model-program
    resource: /lib/model_program.ml
    title: Current Model Program state and entrypoint contract
  - id: serving-cache
    resource: /lib/serving_cache.ml
    title: Current serving cache policy
  - id: serving-engine
    resource: /lib/serving_engine.ml
    title: Serving execution and greedy speculative acceptance
  - id: baseline-receipt
    resource: /bench/results/llama-cpp-gemma4-12b-mtp-2026-08-31.json
    title: Gemma 4 12B llama.cpp sequential and MTP receipt
  - id: publisher-mtp
    resource: https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF/blob/main/MTP/README.md
    title: Publisher MTP usage notes
---

# Context

The Gemma 4 assistant GGUF is not an independent autoregressive draft model. Its
four next-token steps are conditioned on target-model state: the target hidden
state, the target token embedding, position information, and the target's shared
full-attention and sliding-attention KV state. Treating it as a second ordinary
token-ID `Serving_engine.t` does not implement Gemma MTP.

The target is also outside the current uniform cache contract. Its 48 attention
layers mix 40 sliding-attention layers with 8 KV heads and 256-wide heads and
8 global-attention layers with 1 KV head and 512-wide heads. The current Model
Program records one `kv_heads` and one `head_dim`, while the serving Q8 cache
requires `head_dim=64`.

# Decision

Gemma 4 MTP is one linked Model Program with explicit target and assistant
entrypoints, not two interchangeable language-model engines.

1. Target prefill/decode returns logits, the final 3840-wide hidden state, and
   updated heterogeneous persistent KV state.
2. Each assistant step consumes the current target token embedding concatenated
   with the target hidden state, plus position and the target's shared full and
   sliding KV state. Its GGUF projections map the 7680-wide coupled input into
   the 1024-wide four-layer assistant and project back to target hidden width.
3. The assistant proposes up to four tokens sequentially. These proposals are a
   target-coupled MTP sequence, not four independent draft-model calls.
4. The target verifies the proposal in one `K+1` prediction window. Greedy
   acceptance emits the longest matching draft prefix followed by the target
   correction token, or the target bonus token when every proposal matches.
5. Cache reservation and rollback are applied only around an executable target
   verification result. Metadata-only reservations do not constitute a working
   speculative runtime.

No queue priority multiplier or throughput expectation is encoded in this
contract. Performance is reported from measured sequential and MTP campaigns.

# Current implementation boundary

| Capability | Current evidence | State |
|---|---|---|
| Greedy acceptance rule | `Serving_engine.Speculative_acceptance` plus mismatch, partial-match, full-match, and invalid-window tests | implemented |
| Speculative slot metadata | `Serving_cache.commit_speculative` and `rollback_speculative` tests | primitive only |
| Tree-attention Metal entrypoint | Registered generated kernel and runtime dispatch | primitive only |
| Heterogeneous 256/512-head persistent cache | ABI-v3 per-layer geometry, exact Q8 regions, ABI-v2 reads, and mixed-layout transaction tests | implemented |
| Functional target prefill/decode capture | Meta-device FX signatures and complete GGUF binding audit; no `Serving_package` artifact | capture only |
| Target-coupled assistant entrypoint | ABI-v4 serializes the pinned hidden/shared-KV name boundary and package path; no assistant package or Metal binding | metadata only |
| End-to-end server integration | No MTP request state or executable proposal/verification path | missing |

# Consequences

- GGUF inspection verifies tensor storage and shapes; it does not establish an
  executable Model Program.
- The runtime work has functional graph capture, heterogeneous KV layout, and
  linked target/assistant metadata. It still needs executable package lowering
  and runtime binding before serving integration.
- The corrected external baseline remains an observation: `31.14 tok/s`
  sequential versus `18.55 tok/s` MTP, a `0.5956968529x` throughput ratio, with
  exact `92/137` draft acceptance and identical generated output.
