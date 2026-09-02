---
type: Experiment
title: 'Primitive-assignment-backed Gemma execution'
description: 'Records the first packaged Gemma run whose fused gated-linear commands preserve primitive bindings through schedule serialization and derive runtime execution from those bindings.'
tags: [experiment, gemma-4, constructive-lowering, primitive-assignment, schedule, metal, model-execution]
status: stable
generated: { by: codex/gpt-5.6, at: '2026-09-03T05:10:00+07:00' }
sources:
  - id: execution-receipt
    resource: /bench/results/uoc-assignment-backed-gemma4-2026-09-03.txt
    title: Assignment-backed package and model execution receipt
  - id: matched-policy
    resource: /experiments/exp-0140-gemma4-matched-policy-diagnostic-2026-09-03.md
    title: Legacy-backed matched-policy diagnostic
  - id: compiler-decision
    resource: /decisions/constructive-primitive-loop-lowering-and-gpu-queue-compilation.md
    title: Constructive assignment lowering and AOT GPU command-queue decision
---

# Executable boundary

Accepted Linear/add and dual-Linear/activation/multiply regions are now
materialized as validated `Kernel_ir` programs. A backend tactic is selected
from primitive slot dataflow, not from adjacency in the source AST. The Metal
emitter removes covered macro nodes from its graph scan and obtains the
required quantized tactics from these regions.

Serving Schedule ABI 28 serializes the complete primitive program: boundary
slots, bindings, output metadata, effects, resources, and result storage. Shape
specialization reconstructs the program with specialized values and Linear
dimensions. At execution, Metal runtime derives the fused operation's inputs,
output, dimensions, storage, and activation from the deserialized primitive
program; the macro operation retained in the schedule is a compatibility view
for unmigrated consumers.

# Gemma package and execution

The retained Gemma graphs produced 48 primitive regions in prefill and 48 in
decode. Model Program validation reads those counts back from the packaged
schedules after serialization. The exact 37-token probe produced:

| Runtime/path | Policy argmax | Policy logit |
|---|---:|---:|
| llama.cpp reference | 100 | 28.4734364 |
| LLMOpt legacy-backed | 236888 | 20.234375 |
| LLMOpt assignment-backed | 236888 | 20.234375 |

Every recorded raw and policy top-10 token/logit pair in the assignment-backed
run is identical to the prior LLMOpt diagnostic. This proves that the 48 Gemma
gated-linear regions execute through the new packaged assignment path without
changing the observed LLMOpt result. It does not establish llama.cpp parity:
the cross-runtime top-10 sets remain disjoint.

# Remaining boundary

RMSNorm/RoPE, ShortConv/SiLU, GQA layout, and gated-delta discovery remain in
their legacy passes. They were not renamed or reported retired. The explicit
execution order requires llama.cpp comparison before deletion, and the current
comparison is divergent, so those modules remain active while their primitive
models and backend tactics are implemented.

Evidence: package readback preserved 96 primitive regions and the model run
reproduced every prior LLMOpt top-10 value; assumption: identifying the first
cross-runtime arithmetic divergence still requires intermediate-tensor
comparison.
