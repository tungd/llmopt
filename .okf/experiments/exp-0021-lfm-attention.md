---
type: Experiment
title: 'Typed LFM masked attention and Metal kernel'
description: 'Lower the captured float16 GQA prefill boundary into typed OCaml IR, a versioned binary command, CPU softmax reference, and correctness-first fused MSL.'
tags: [experiment, lfm25, attention, gqa, sdpa, ocaml, metal, schedule]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-23T19:20:32Z' }
sources:
  - id: shape
    resource: /lib/tensor_shape.ml
    title: Attention shape and mask validation
  - id: planner
    resource: /lib/fx_plan.ml
    title: Captured SDPA lowering
  - id: cpu
    resource: /lib/cpu.ml
    title: Masked softmax CPU reference
  - id: metal
    resource: /lib/metal.ml
    title: Generated fused attention MSL
  - id: replan
    resource: /_artifacts/lfm25-350m-q8-v5-attention-serving-replan-2026-08-24/package.llmopt
    title: Offline serving-package replan
---

# Targeted form

The saved no-cache graph contains six identical attention calls after GQA key
and value expansion:

```text
query/key/value [1,16,6,64] f16
mask            [1,1,6,6] bool
dropout         0
scale           0.125
is_causal       false
output          [1,16,6,64] f16
```

The planner accepts this masked inference contract. Non-zero dropout,
non-boolean masks, non-float16 tensors, incompatible shapes, or the captured
mask combined with causal mode retain the opaque fallback.

# Implementation

The shape domain validates rank-four query/key/value agreement and broadcast
mask dimensions. `Ir.Attention.t` owns a finite scale and causal flag. Schedule
version 5 encodes those fields, reads versions 1 through 5, and validates all
input/output shapes and dtypes on decode.

The CPU effect handler computes scaled query/key products, applies both the
boolean mask and causal condition, uses max-subtracted softmax, and combines
values. Its two-token fixture proves masking and both softmax weights.

The generated MSL fuses score, masked softmax, and value accumulation into one
kernel with float32 accumulation and float16 output. One thread owns one
batch/head/query row. It deliberately recomputes scores while producing each
head dimension; this is a correctness-first boundary, not the intended tiled
long-context schedule.

# Evidence

```sh
ninja -f ninja.build test attention-smoke
```

passes the CPU, FX, schedule, and kernel-ABI checks and compiles the generated
source with Xcode Metal. Replanning the saved manifest against its single
safetensors archive, without loading the model, produced:

```text
planned 1115 FX nodes into 835 IR nodes
valid serving package: 5 kernels, 835 commands, 12 opaque, tensor-store=241
```

All six SDPA nodes moved out of opaque fallback. The remaining 12 are 5
arange, 2 advanced getitem, and one each of embedding, diff, cumsum, new_ones,
and a logging side effect.

# Exact boundary

This slice compiles but does not dispatch the attention kernel. Decode
attention, KV reads/writes, a tiled/simdgroup schedule, and native command
interpretation remain separate work. The package also still omits generated
kernels for much of its already typed schedule, including Q8 linear commands.
No model/device process, parity measurement, generation, cache request, needle
request, or ERS run was performed.
