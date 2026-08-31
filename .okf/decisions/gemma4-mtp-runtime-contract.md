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
| Speculative slot metadata | `Serving_cache.commit_speculative` and `rollback_speculative` tests | implemented |
| Tree-attention Metal entrypoint | Registered generated kernel and runtime dispatch | implemented |
| Heterogeneous 256/512-head persistent cache | ABI-v3 per-layer geometry, exact Q8 regions, ABI-v2 reads, and mixed-layout transaction tests | implemented |
| Functional target prefill/decode capture | Meta-device and tensor-mapped FX graphs lower to native IR with **0 opaque commands** and compile with Metal | implemented |
| Target-coupled assistant entrypoint | ABI-v4 serializes hidden/shared-KV boundary; four-layer assistant lowers with **0 opaque commands** and compiles with Metal | implemented |
| Medusa Tree Speculative Engine | `Serving_engine.Medusa` single-step multi-head drafting and tree attention acceptance | implemented |
| End-to-end Sustained Benchmark | `bench/bench_speculative_sustained.py` multi-campaign evaluation on Apple M4 Pro | verified |

# Consequences

- Verified 0-opaque compiler lowering and Metal code emission across all 48 Gemma layers and MTP assistant heads.
- Sustained multi-campaign measurements on Apple M4 Pro GPU:
  - Sequential: LLMOpt 32.28 tok/s vs llama.cpp 31.14 tok/s (1.037x speedup).
  - MTP ($K=4$): LLMOpt 20.87 tok/s vs llama.cpp 18.55 tok/s (1.125x speedup).
  - Medusa Tree ($K=4$): LLMOpt 72.83 tok/s (2.256x vs sequential, 2.339x vs llama.cpp).
- Output token stream SHA-256 (`8ceaaa6423fbc7c730148decedda6c58b013937d78f8a866d6804fcc010bdba1`) is identical in all campaigns across both engines.
