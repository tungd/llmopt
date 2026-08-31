---
type: Experiment
title: 'Gemma 4 12B target-coupled MTP functional capture'
description: 'Records meta-device FX capture of explicit target prefill/decode cache state and the target-coupled Gemma 4 assistant step, with complete pinned-GGUF tensor-binding audits.'
tags: [experiment, gemma4, mtp, fx, functional-cache, gguf, model-program]
status: stable
generated: { by: 'process:codex', at: '2026-08-31T15:55:00+07:00' }
sources:
  - id: receipt
    resource: /bench/results/gemma4-mtp-functional-capture-2026-08-31.json
    title: Functional graph signatures and binding receipt
  - id: capture
    resource: /bench/gemma4_mtp_capture.py
    title: Meta-device functional capture and GGUF binding audit
  - id: tests
    resource: /python/tests/test_gemma4_mtp_capture.py
    title: Geometry, mapping, cache, and digest regressions
  - id: contract
    resource: /decisions/gemma4-mtp-runtime-contract.md
    title: Target-coupled Gemma 4 MTP runtime contract
---

# Probe

The pinned Google text config was instantiated on PyTorch's `meta` device with
Float16 activation semantics; no checkpoint weights or device model were
loaded. A traceable functional cache accepted every layer key/value tensor as a
graph input and returned every updated key/value tensor as a graph output.
Target attention masks and position IDs were also explicit inputs.

The assistant graph consumed a 7680-wide concatenation of target token
embedding and target hidden state, plus target shared full-attention and
sliding-attention KV tensors. It returned vocabulary logits and a projected
3840-wide target hidden state.

# Result

| Entrypoint | FX nodes | Static tensors | Runtime inputs | Tensor outputs |
|---|---:|---:|---:|---:|
| Target prefill, 2 tokens | 6,423 | 669 | 100 | 102 |
| Target decode, 2 cached + 1 token | 6,423 | 669 | 100 | 102 |
| Target-coupled assistant step | 424 | 50 | 8 | 2 |

The target signatures expose 96 layer cache tensors plus tokens, positions, and
two masks. Their outputs contain logits, target hidden state, two shared KV
pairs, and all 96 updated layer cache tensors. The target GGUF audit binds all
666 weight/state tensors other than `rope_freqs.weight`; the assistant audit
binds all 48 corresponding tensors. In both captures, the GGUF rope tensor is
replaced by deterministic full/sliding inverse-frequency constants derived
from the pinned config. Binding SHA-256 digests and all individual mappings are
in the structured receipt.

# Boundary

This establishes functional graph and tensor-binding contracts. It does not
yet create a `model.llmopt` package, generalize the OCaml cache ABI, execute the
graphs through Metal, or measure LLMOpt generation.
