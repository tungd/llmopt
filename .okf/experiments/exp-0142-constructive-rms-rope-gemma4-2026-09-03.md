---
type: Experiment
title: 'Constructive RMSNorm and RoPE Gemma execution'
description: 'Dependency-sliced primitive RMSNorm and RMSNorm-RoPE regions execute the pinned Gemma 4 12B model with the same next token as llama.cpp.'
tags: [experiment, gemma-4, constructive-lowering, rmsnorm, rope, model-execution]
status: stable
generated: { by: codex/gpt-5.6, at: '2026-09-03T06:00:00+07:00' }
sources:
  - id: execution-receipt
    resource: /bench/results/uoc-constructive-rms-rope-gemma4-2026-09-03.txt
    title: Constructive RMSNorm and RoPE model execution receipt
  - id: compiler-decision
    resource: /decisions/constructive-primitive-loop-lowering-and-gpu-queue-compilation.md
    title: Constructive assignment lowering decision
---

# Result

The compiler normalizes raw cast, pointwise, reduction, index, movement, and
ShortConv assignments into `Kernel_ir`. Tactic selection searches legal maximal
dependency components and extracts the exact inner dependency slice for each
executable region, preserving upstream projections outside the selected slice.

The captured Gemma graphs produce 48 gated-Linear, 96 RMSNorm-RoPE, and 193
standalone RMSNorm regions in both prefill and decode. Schedule serialization
reads back 337 primitive regions per entrypoint, and Metal runtime selects their
kernels from deserialized bindings rather than macro `Ir.Op` replacements.

On the pinned GGUF and exact 37-token prompt, the probe completed in `9.887640
s` and selected token `100`, equal to the recorded llama.cpp policy token. The
ranked IDs agree through rank eight and then differ. This establishes the
probed next-token equality, not bit-exact logit equality.

Evidence: a valid ABI-v7 package executed 337 deserialized assignment regions
per entrypoint and returned token `100`; assumption: the later top-logit ordering
difference is accumulated floating-point ordering rather than a semantic
constant mismatch.
