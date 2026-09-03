---
type: Experiment
title: 'Gemma 4 postprocessed last-token projection execution'
description: 'Specialization projects one final Gemma prefill row through its postprocessed logits chain; the corrected artifact completed 128 Metal-backed autoregressive tokens and matched the five retained llama.cpp prefix IDs.'
tags: [experiment, gemma-4, prefill, last-token-projection, autoregressive-decode, module-cohesion]
status: stable
generated: { by: codex/gpt-5.6, at: '2026-09-03T10:30:55+07:00' }
sources:
  - id: attempt-receipt
    resource: /bench/results/uoc-gemma4-last-token-projection-attempt-2026-09-03.txt
    title: Gemma postprocessed last-token projection attempt
  - id: completion-receipt
    resource: /bench/results/uoc-gemma4-last-token-projection-completion-2026-09-03.txt
    title: Corrected Gemma postprocessed last-token projection execution
  - id: prior-parity
    resource: /bench/results/uoc-final-constructive-gemma4-parity-2026-09-03.txt
    title: Final constructive Gemma matched-policy parity
  - id: specialization
    resource: /lib/serving_schedule.ml
    title: Sequence specialization implementation
  - id: layout-model
    resource: /lib/layout_model.ml
    title: Typed logical-axis provenance model
---

# Result

Prefill specialization now follows the declared `logits` output backward
through a sole-consumer, shape-preserving pointwise chain. For the Gemma tail
`Linear -> Div -> Tanh -> Mul -> Output`, it inserts a typed final-row `Index`
immediately before the Linear when the captured graph has no identity index to
rewrite. The downstream pointwise commands consequently operate on
`[1,1,262144]`, while `all_token_outputs=true` retains the full token axis.

`Layout_model` is now the independent `lib/layout_model.ml` compilation unit
with only `canonicalize_attention` exposed. `pass_fuse_linear_bias.ml` retains
the legacy Linear/QKV matchers and delegates typed layout canonicalization to
that unit; every relevant Ninja compile and link target carries the new module.

The first final-artifact execution attempt stopped during specialization because
the first implementation required an existing identity index. It produced no
Metal prompt dispatch, logits, latency, or completion sequence.[^attempt-receipt]
After the direct-input correction, the exact ABI-v7 artifact passes static
specialization with one prefill head row, 48 paged-Q8 decode attentions, and all
337 primitive regions per entrypoint intact.

The OCaml suite, all 79 Python tests, the full build, and the 1,125-command,
135-kernel native schedule fixture pass. The corrected device execution then
completed all 128 requested autoregressive tokens with TTFT `8.761896 s`, mean
inter-token time `221.304664 ms`, median inter-token time `180.555105 ms`, and
total throughput `3.471884 tok/s`.[^completion-receipt]

The first five completion IDs are
`100,45518,107,14947,135383`, exactly equal to the five IDs retained for the
pinned same-GGUF llama.cpp sequential stream. That llama.cpp receipt stores only
a generated-text hash beyond this prefix, not the complete token IDs or text,
so positions 6 through 128 cannot be compared exactly from retained evidence.

The earlier `10.581340 s` measurement was a top-k probe including logits
readback and ranking. The new generation TTFT is `1.819444 s` lower, but the
different measurement modes prevent attributing that delta solely to the
last-token projection.

Evidence: the focused regression proves one-row and explicit all-token shapes,
and the corrected device run completes 128 tokens with five retained reference
IDs matching; assumption: none for those facts, and no all-128 parity assumption
is made.

[^attempt-receipt]: Gemma postprocessed last-token projection attempt.
[^completion-receipt]: Corrected Gemma postprocessed last-token projection execution.
