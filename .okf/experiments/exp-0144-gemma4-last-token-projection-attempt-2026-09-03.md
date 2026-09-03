---
type: Experiment
title: 'Gemma 4 postprocessed last-token projection attempt'
description: 'Specialization now projects one final Gemma prefill row through its postprocessed logits chain; the sole device attempt stopped before dispatch, so corrected latency and sustained parity remain unmeasured.'
tags: [experiment, gemma-4, prefill, last-token-projection, autoregressive-decode, module-cohesion]
status: stable
generated: { by: codex/gpt-5.6, at: '2026-09-03T09:47:12+07:00' }
sources:
  - id: attempt-receipt
    resource: /bench/results/uoc-gemma4-last-token-projection-attempt-2026-09-03.txt
    title: Gemma postprocessed last-token projection attempt
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

The one final-artifact execution attempt stopped during specialization because
the first implementation required an existing identity index. It produced no
Metal prompt dispatch, logits, latency, or completion sequence. After the
direct-input correction, the exact ABI-v7 artifact passes static specialization
with one prefill head row, 48 paged-Q8 decode attentions, and all 337 primitive
regions per entrypoint intact.[^attempt-receipt]

The OCaml suite, all 79 Python tests, the full build, and the 1,125-command,
135-kernel native schedule fixture pass. No second model-execution attempt was
made, so neither a corrected prefill time nor 128-token sequential parity is
reported.

Evidence: the focused regression proves one-row and explicit all-token shapes,
and the exact final artifact statically specializes to one head row; assumption:
the corrected but unexecuted device path preserves the prior token `100` result.

[^attempt-receipt]: Gemma postprocessed last-token projection attempt.
