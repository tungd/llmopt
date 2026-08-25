---
type: Experiment
title: 'Fused Short-Convolution block optimization'
description: 'Passes.fuse_short_conv collapses 13-command prefill and 16-command decode conv subgraphs across 10 conv layers into single SIMD Metal kernels, dropping decode commands to 606 (specialized 546) and boosting short-trace ERS to 0.510917.'
tags: [experiment, compiler, ocaml, metal, q8, conv1d, short-conv, benchmark, ers, needle, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-26T01:45:00Z' }
sources:
  - id: implementation
    resource: /bench/results/lfm25-350m-q8-short-conv-compiler-2026-08-26.txt
    title: Fused Short-Convolution compiler boundary
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-short-conv-measurement-2026-08-26.txt
    title: Bounded short and long-context measurement
  - id: prior
    resource: /bench/results/lfm25-350m-q8-rms-rope-measurement-2026-08-26.txt
    title: Previous fused RMSNorm-RoPE observation
---

# Compiler optimization

The `Passes.fuse_short_conv` optimizer pass targets the Short-Convolution block bodies across all 10 conv layers in LFM2.5-350M:
1. **Decode conv step**: Folds the 16-command chain (`transpose(2,1)` -> 3x `index` -> `mul` -> `roll` -> `update-index` -> `copy` -> `cast(f16)` -> `index(conv_weight)` -> `mul` -> `sum(axis=2)` -> `unsqueeze(2)` -> `mul` -> `transpose(2,1)` -> `contiguous`) into a single `Short_conv_step` IR operation dispatched as the `llmopt_short_conv_step_f16` SIMD Metal kernel (1024 threads across channels with in-place ring-buffer update and input/output gating).
2. **Prefill conv step**: Folds the 13-command chain (`transpose(2,1)` -> 3x `index` -> `mul` -> state slice `index` -> `fill(0)` -> `copy` -> `short-conv` -> `index` -> `mul` -> `transpose(2,1)` -> `contiguous`) into a single `Short_conv_prefill` IR operation dispatched as the `llmopt_short_conv_prefill_f16` SIMD Metal kernel.

Command counts drop from:
- Prefill: 702 -> 592 commands (-110 commands)
- Decode: 756 -> 606 commands (-150 commands)
- Specialized decode: 696 -> 546 commands (-150 commands)
- Memory allocations in decode: 390 -> 262 allocations (-128 allocations)

# Measurement on Apple M4 Pro

Against `_artifacts/lfm25-350m-q8-short-conv-replan-2026-08-26`:
- **Short trace**: 4/4 requests complete with 100% exact token sequence parity (`[1098, 5706, 803, 4481]`, `[41677, 7, 2, 1]`, `[1098, 5410, 4100, 856]`, `[31466, 7, 2, 1]`). ERS surges to `0.510917` (+0.098657), median TTFT reaches `56.511 ms` (-17.195 ms), and median TPOT reaches `5.554 ms` (-1.506 ms).
- **Long-context needle**: 6/6 requests complete with 6/6 retrieval and 100% exact eager-Q8 12-token sequence parity (`[8832, 563, 2880, 522, 31429, 526, 7, 2, 1, 553, 849, 18149]`). 2K TTFT improves to `1183.136 ms` and 4K TTFT improves to `2828.978 ms` (-207 ms).
