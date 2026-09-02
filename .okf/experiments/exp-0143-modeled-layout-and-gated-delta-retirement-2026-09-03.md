---
type: Experiment
title: 'Modeled layout and gated-delta pass retirement'
description: 'GQA and token-major layouts use typed axis provenance; padded gated-delta executes from primitive assignment regions; four ad hoc pass modules are removed.'
tags: [experiment, compiler, constructive-lowering, gqa, gated-delta, pass-retirement]
status: stable
generated: { by: codex/gpt-5.6, at: '2026-09-03T12:00:00+07:00' }
sources:
  - id: compiler-decision
    resource: /decisions/constructive-primitive-loop-lowering-and-gpu-queue-compilation.md
    title: Constructive assignment lowering decision
  - id: assignment-model
    resource: /lib/kernel_ir.ml
    title: Primitive assignment and fused tactic model
  - id: layout-model
    resource: /lib/pass_fuse_linear_bias.ml
    title: Typed logical-axis provenance model
  - id: final-attempt
    resource: /bench/results/uoc-final-constructive-gemma4-attempt-2026-09-03.txt
    title: Post-retirement Gemma attempt and specialization correction
---

# Result

`Kernel_ir.Primitive` now represents batched matrix multiplication, right-zero
padding, slice updates, and triangular recurrence. Constructive lowering keeps
those operations in legal dependency components, identifies query, key, value,
gate, and beta roles from shapes and consumer relations, extracts the exact
backward assignment slice, serializes it, and dispatches the existing
captured-width gated-delta Metal kernel directly.

One logical-axis model composes transpose, full-slice/new-axis indexing,
expansion, reshape/view, contiguous movement, and dtype casts. Attention and
gated-delta consumers query that provenance for grouped-head replication,
token-to-head movement, and flattened token-major output. The compiler no
longer runs fixed GQA, attention-output, value-transpose, or gated-delta graph
sequence matchers.

The first post-retirement 37-token execution stopped before Metal dispatch:
prefill specialization changed GQA reshape value `776` from `5,607,424` input
bytes to `303,104` output bytes. The captured token count and fixed GQA group
factor were both two, and provenance had crossed the shared cache output before
generic `2 -> 37` substitution. Layout tracing now stops at shared dependency
values, preserving cache-producing transposes while removing the replicated
GQA suffix. The corrected prefill has 5,273 rather than 5,753 commands, and the
new package checker successfully constructs the exact 37-token specialized
workspace plan.

The retired `pass_fuse_rms_norm`, `pass_fuse_rms_rope`,
`pass_fuse_short_conv`, and `pass_fuse_gated_delta` modules, public aliases,
and Ninja link dependencies are removed. Focused OCaml coverage proves direct
assignment selection, schedule round-trip, and Metal source selection for
RMSNorm, RMSNorm-RoPE, ShortConv-SiLU, and padded gated-delta. The complete
repository test target passes together with all 79 Python tests. The corrected
ABI-v7 bundle retains 337 primitive regions per entrypoint: 48 gated-Linear,
96 RMSNorm-RoPE, and 193 RMSNorm.

Evidence: the compiler and tests build without the four retired modules, the
selected gated-delta command contains a deserialized
`Kernel_ir.Fused_tactic.Gated_delta`, and the corrected package passes static
`2 -> 37` specialization; assumption: its model token remains unknown because
the one post-retirement execution stopped before dispatch and was not retried.
