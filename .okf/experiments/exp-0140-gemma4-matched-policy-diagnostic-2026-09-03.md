---
type: Experiment
title: 'Gemma 4 matched-policy diagnostic and constructive-transition correction'
description: 'Records the first same-prompt llama.cpp and LLMOpt top-logit comparison with GGUF suppression applied, the remaining arithmetic divergence, and the restoration of legacy passes after rejecting receipt-only Kernel_ir regions as a completed compiler transition.'
tags: [experiment, gemma-4, llama-cpp, logits, token-suppression, arithmetic-parity, constructive-lowering, correction]
status: stable
generated: { by: codex/gpt-5.6, at: '2026-09-03T03:15:43+07:00' }
sources:
  - id: matched-policy-receipt
    resource: /bench/results/uoc-gemma4-matched-policy-attempt-2026-09-03.txt
    title: Exact commands and top-10 logits from both runtimes
  - id: compiler-decision
    resource: /decisions/constructive-primitive-loop-lowering-and-gpu-queue-compilation.md
    title: Constructive assignment lowering and AOT GPU command-queue decision
  - id: prior-repair
    resource: /experiments/exp-0139-unified-compiler-gemma4-repair-2026-09-02.md
    title: Prior Gemma repair receipt whose transition claim is corrected here
---

# Scope

This experiment completes the previously abandoned matched-policy diagnostic.
It uses the same pinned GGUF, exact 37 prompt token IDs, and suppression set in
both runtimes. It also audits whether the attempted compiler migration actually
made `Kernel_ir` the backend execution input.

# Execution correction

The first builder invocation exited at stale `llmopt-pipeline` argument parsing
after capturing the graphs and augmented weight archive. The project was then
rebuilt, the retained inputs were packaged, and the Model Program validated as
ABI v7 with 48 attention layers and suppressed token IDs `258882,258883`.

Two unused `torch._C._set_grad_enabled` capture-control nodes had survived as
opaque runtime commands. FX planning now removes those inference-capture
controls before IR construction. Gemma's postprocessed logits chain also makes
the optional last-token projection rewrite inapplicable, so specialization
keeps the all-token logits calculation and sampling reads its final row.

# Matched-policy result

Both runtimes evaluated the exact same 37 token IDs.

| Runtime | Raw argmax | Policy argmax | Top logit |
|---|---:|---:|---:|
| llama.cpp | 100 | 100 | 28.4734364 |
| LLMOpt | 258882 | 236888 | 20.234375 |

The raw and policy top-10 sets are disjoint across runtimes. Applying the GGUF
suppression policy changes LLMOpt's selected token from `258882` to `236888`,
but llama.cpp selects `100` before and after the same mask. Token suppression is
therefore implemented and observable, while the first-token arithmetic result
still diverges.

LLMOpt reported `0.083030 s` package load time and `7.500555 s` for the logits
probe. These timings are observations from this diagnostic, not a throughput
comparison.

# Compiler-transition correction

The attempted follow-up migration copied the old operation-family sequence
matchers into `assignment_*_tactics` modules, ran them before a region-membership
check, and printed `Kernel_ir` bindings only in `plan.txt`. Backend code
generation and runtime dispatch still consumed macro `Ir.Op` operations. That
was a facade over the old architecture, not operator erasure or constructive
backend lowering.

Those renamed matchers and their deletion of the six `pass_fuse_*` modules were
removed. The legacy passes are restored as the executable path. Consequently,
the zero-legacy and unified-transition wording in exp-0139 is not evidence that
`Kernel_ir` drove code generation; it described rewrite accounting around a
macro-IR backend.

The next compiler slice must make assignment bindings the input to emitted
kernels and scheduled dispatches before any covered legacy matcher is retired.
The matched-policy arithmetic result above is authoritative for the current
legacy-backed Gemma bundle, not for a future assignment-backed bundle.

Evidence: llama.cpp and LLMOpt produced the recorded top-10 logits under the
same prompt IDs, GGUF, and suppression set; assumption: final logits alone
localize the remaining mismatch to model arithmetic after policy application,
but not to a particular intermediate operator or layer.
