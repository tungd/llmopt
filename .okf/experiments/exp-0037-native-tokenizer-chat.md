---
type: Experiment
title: 'Binary LFM tokenizer and typed chat encoding'
description: 'Replace Transformers tokenization and Jinja rendering in the serving path with a versioned binary archive and native OCaml implementation.'
tags: [experiment, ocaml, serving, tokenizer, bpe, chat, binary, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-24T00:28:36Z' }
sources:
  - id: archive-writer
    resource: /python/llmopt_backend/tokenizer_archive.py
    title: Binary tokenizer archive writer
  - id: tokenizer
    resource: /lib/tokenizer.ml
    title: Native byte-level BPE tokenizer
  - id: chat
    resource: /lib/lfm_chat.ml
    title: Typed text-only LFM chat encoder
  - id: probe
    resource: /bench/lfm25_tokenizer_parity.py
    title: Transformers differential probe
  - id: evidence
    resource: /_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi8-engine-2026-08-24/tokenizer-parity.txt
    title: Token and chat parity output
---

# Binary boundary

The offline exporter reads the upstream Hugging Face `tokenizer.json` and
writes `tokenizer.llmopt` with magic `LLMOPTTK`, ABI version 1, LFM profile 1, fixed-width
little-endian token records, UTF-8 token bytes, added/special flags, and an
ordered merge table. The LFM2.5-350M archive contains 64,402 tokens and 63,683
merges in 2,284,649 bytes. The native OCaml runtime reads only this binary
archive.

`Tokenizer.t` owns the validated token tables, a longest-match added-token
trie, the GPT-2 byte-to-Unicode mapping, the exact LFM Unicode pre-tokenizer,
ranked BPE merging, and inverse decoding. Its interface keeps the parsed state
abstract and reports malformed archives and unknown symbols as `result`
errors.

# Typed chat path

`Lfm_chat` represents system, user, assistant, and tool roles as a variant and
message values as an abstract type. It emits BOS, message boundary tokens,
role/content text, and the assistant generation prompt directly into token
IDs. The current boundary intentionally covers the text-only request shape
used by the benchmark. It also matches the LFM template's empty initial system
message and historical assistant-thinking handling without evaluating Jinja or
parsing JSON.

# Differential evidence

The offline probe compares `_build/bin/llmopt-tokenize` against the exact local
LFM2.5-350M tokenizer files. All 10 text cases match token IDs and decoded text,
covering contractions, Vietnamese, long digit runs, whitespace/newlines,
punctuation, added tokens, emoji, and chat markers. All 6 typed chat cases
match Transformers token-for-token, including multi-turn assistant thinking,
an empty initial system message, and a tool response.

This probe loads neither model weights nor a Metal device. It establishes the
text-to-token boundary only; variable-length compiled graphs, a generation
loop, HTTP request handling, needle retrieval, and ERS remain separate work.
