---
type: Experiment
title: 'Native OCaml chat generation'
description: 'Connect the binary LFM tokenizer and typed chat template to dynamic prefill, greedy decode, radix/KV state, and decoded text.'
tags: [experiment, ocaml, metal, serving, generation, tokenizer, radix-cache, q8, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-24T01:27:05Z' }
sources:
  - id: core
    resource: /lib/generation_core.ml
    title: Device-independent generation state machine
  - id: generation
    resource: /lib/generation.ml
    title: Native LFM generation adapter
  - id: engine
    resource: /lib/serving_engine.ml
    title: Radix-aware prompt preparation
  - id: cli
    resource: /bin/lfm_generate.ml
    title: Native chat-generation executable
  - id: reference
    resource: /bench/lfm25_reference_tokens.py
    title: Exact-input eager-Q8 reference
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-native-chat-generation-2026-08-24.txt
    title: Native chat and eager-Q8 evidence
---

# Generation owner

`Generation_core` is a functor over a typed model-step interface. It owns the
direct autoregressive loop, positive `max_new_tokens` invariant, stop-token and
length finish reasons, ordered token emission, TTFT, and inter-token timing.
The device-independent adapter is covered with synthetic prompt/decode steps,
including stop and length termination, without opening Metal.

`Generation` instantiates that core with `Serving_engine`. It accepts typed
`Lfm_chat.Message.t` values, encodes the exact LFM generation prompt through
the binary `LLMOPTTK` tokenizer, samples float16 logits greedily, decodes
completion IDs to text, validates cache ownership, and returns radix/KV stats.

`Serving_engine.prompt` asks the mandatory radix cache for the deepest
checkpoint before the final prompt token. A cold request executes dynamic
prefill. When a reusable checkpoint exists, it runs only the uncached prompt
suffix through one-token decode, preserving physical KV and recurrent state.
The final token is always replayed when needed, so an exact cached prompt does
not require storing logits in the radix tree.

# Native observation

System memory was 60% free with no Torch/model process before the one
180-second-supervised native launch. The CLI loaded the binary tokenizer,
binary package, generated metallib, and binary tensor archive, then generated
four tokens from one typed user message:

```text
messages: user=Say hi.
prompt-token-count: 13
prompt-tokens: 1,6,6423,708,560,892,15012,523,7,708,6,64015,708
completion-tokens: 36309,510,2213,1011
text: Hello! How can
finish-reason: length
ttft-seconds: 0.434065
mean-tpot-seconds: 0.122412
radix-cached-tokens: 16
radix-hits: 3
radix-misses: 1
kv-used-tokens: 16
kv-used-checkpoints: 4
```

One separately memory-checked eager PyTorch MPS process received those exact
13 input IDs after applying the same 92-module Q8 rewrite. It produced exactly
`36309,510,2213,1011`. The observation therefore proves native text-to-token,
dynamic 13-token prefill, three growing decode steps, token-to-text, and greedy
parity in one flow.

# Boundary

This is one direct CLI request. It does not yet expose HTTP/SSE, prove
cross-request or multi-turn cached-prompt reuse, run the benchmark request
contract, execute needle retrieval, or record ERS. The reported latency values
are the measurements from this one observation, not a comparative score.
