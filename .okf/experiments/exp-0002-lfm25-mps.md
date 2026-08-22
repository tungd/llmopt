---
type: Experiment
title: 'LFM2.5-350M naive MPS execution'
description: 'The complete LFM2.5 forward graph runs through the OCaml-planned FX backend and matches eager PyTorch MPS exactly.'
tags: [experiment, lfm2.5, pytorch, mps, end-to-end]
status: stable
generated: { by: codex/gpt-5, at: '2026-08-20T13:15:00Z' }
sources:
  - id: harness
    resource: /bench/lfm25_mps.py
    title: LFM2.5 MPS benchmark harness
  - id: result
    resource: /bench/results/lfm25-mps-2026-08-20.json
    title: recorded MPS benchmark result
  - id: executor
    resource: /python/llmopt_backend/__init__.py
    title: naive MPS FX executor
---

# Question

Can the current Dynamo/FX-to-OCaml planning boundary execute the complete
LFM2.5-350M forward graph on PyTorch MPS without an optimization pass?

# Procedure

```sh
python3.13 bench/lfm25_mps.py \
  --model LiquidAI/LFM2.5-350M \
  --iterations 3 --warmup 1 \
  --artifact-dir _artifacts/lfm25-benchmark/graphs \
  --output _artifacts/lfm25-benchmark/result.json
```

The harness loads `LiquidAI/LFM2.5-350M` in float16, runs eager MPS, compiles
with `torch.compile(backend=llmopt)`, and runs the resulting naive FX
interpreter on the same input.

# Observation

On the recorded Mac16,8 host, the 5-token prompt produced exact matching logits
with shape `[1, 5, 128000]`. The OCaml compiler planned 1,725 FX nodes into
1,725 IR nodes. This run measured eager MPS at 0.0411435977 seconds per
forward and the naive planned path at 0.0312050277 seconds; the first compiled
call including planning and execution took 1.9117972920 seconds. These are
single-run comparison measurements, not a performance claim.

# Limits

This measures prefill-style forward execution with `use_cache=False`. It does
not claim custom Metal kernels, KV-cache decode throughput, quantized weights,
or optimized scheduling.
