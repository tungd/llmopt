---
type: Experiment
title: 'Typed LFM position and mask construction'
description: 'Remove the final opaque operations from the saved no-cache prefill graph with schedule-v7 primitives and compiled Metal kernels.'
tags: [experiment, lfm25, fx, ocaml, schedule, metal, mask, position]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T20:00:03Z' }
sources:
  - id: planner
    resource: /lib/fx_plan.ml
    title: FX position and mask lowering
  - id: schedule
    resource: /lib/serving_schedule.ml
    title: Schedule-v7 codecs and validation
  - id: metal
    resource: /lib/metal.ml
    title: Generated position and mask Metal kernels
  - id: artifact
    resource: /_artifacts/lfm25-350m-q8-v8-mask-position-replan-2026-08-24/package.llmopt
    title: Replanned LFM2.5-350M no-cache prefill package
---

# Captured boundary

The saved 1,115-node manifest had 11 opaque commands after embedding lowering:
five static int64 ranges, one prepended int64 difference, one boolean-to-int64
cumulative sum, one scalar boolean `new_ones`, two rank-two gathers with
broadcast rank-four index tensors, and one unused
`torch._C._log_api_usage_once` call.

# Lowering

The executable operations now have typed shape rules and IR primitives. Their
configurations survive schedule v7 and are checked again when the binary
schedule is decoded. CPU references cover the exact range, difference,
cumulative-sum, fill, and gather results. Generated correctness-first MSL adds
one kernel entry for each primitive family.

Only the exact captured telemetry form is elided: one static string argument,
no keywords, and no tensor result. It is not a general dead-code elimination
rule. The scalar `new_ones` command has no false dependency on its receiver,
whose only role in PyTorch is to supply placement.

# Evidence

The focused checks report:

```text
ninja -f ninja.build ocaml-test mask-position-smoke
llmopt tests passed
validate emitted LFM mask and position Metal source
```

Offline replanning reused the saved manifest and the JSON-free 241-tensor
archive. It did not load the model, allocate MPS tensors, or dispatch a device
kernel:

```text
planned 1115 FX nodes into 834 IR nodes
valid serving package: 11 kernels, 834 commands, 0 opaque, tensor-store=241
package.llmopt   94,877 bytes
kernel.metal     12,443 bytes
kernel.metallib  49,342 bytes
weights.llmopt   422,137,216 bytes
```

The generated MSL compiled with the installed Xcode Metal compiler and linked
with `metallib`.

# Boundary

This establishes zero opaque commands for the saved six-token no-cache prefill
graph. It does not cover decode/KV-state graphs, native interpretation of the
834-command stream, physical Q8 KV storage, tokenization, generation, parity,
needle retrieval, latency, or ERS.
