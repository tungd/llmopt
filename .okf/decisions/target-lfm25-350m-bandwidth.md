---
type: Decision
title: 'Target LFM2.5-350M for on-device Apple Silicon ESR viability'
description: 'Standardize on Liquid AI LFM2.5-350M as the primary research and benchmark model, establishing physical memory bandwidth feasibility for the Racebench ERS metric on Apple Silicon.'
tags: [decision, lfm2.5, 350m, memory-bandwidth, esr, racebench, apple-silicon]
status: stable
generated: { by: codex/gpt-5, at: '2026-08-22T13:30:00Z' }
sources:
  - id: score-formula
    resource: /bench/racebench/score.py
    title: Racebench ERS scoring functions
  - id: target-config
    resource: /lib/lfm25.ml
    title: LFM2.5-350M configuration descriptor
  - id: baseline-350m
    resource: /bench/results/lfm25-350m-q8-racebench-baseline.json
    title: 350M Q8 MPS baseline record
  - id: target-concept
    resource: /.okf/target-lfm25.md
    title: LFM2.5-350M target concept
---

# Decision

Standardize on `LiquidAI/LFM2.5-350M` as the primary on-device model target for the `llmopt` compiler and benchmark pipeline, replacing `LFM2.5-2.6B`.

# Context & Problem

The Viettel AI Race benchmark (`racebench`) evaluates serving engines using the Effective Request Score (ERS), defined by strict latency deadlines:

$$\text{tpot\_score} = \left( \operatorname{clamp}\left(\frac{\text{TPOT}_{\text{ceiling}} - \text{TPOT}}{\text{TPOT}_{\text{ceiling}} - \text{TPOT}_{\text{floor}}}\right) \right)^\gamma \qquad (\text{floor} = 1.0\text{ ms}, \text{ceiling} = 10.0\text{ ms}, \gamma = 2.0)$$

$$\text{ttft\_score} = \left( \operatorname{clamp}\left(\frac{\text{TTFT}_{\text{ceiling}} - \text{TTFT}}{\text{TTFT}_{\text{ceiling}} - \text{TTFT}_{\text{floor}}}\right) \right)^\gamma \qquad (\text{floor} = 10.0\text{ ms}, \text{ceiling} = 400.0\text{ ms}, \gamma = 2.0)$$

$$\text{Request Score} = 0.5 \cdot \text{ttft\_score} + 0.5 \cdot \text{tpot\_score}$$

$$\text{Final Score} = 100 \cdot \text{ERS} \cdot \text{accuracy\_factor}$$

Under this formula:
1. Any generation step with $\text{TPOT} \ge 10.0\text{ ms}$ ($\le 100\text{ tokens/sec}$) receives a hard score of `0.0`.
2. Any prefill with $\text{TTFT} \ge 400.0\text{ ms}$ receives a hard score of `0.0`.
3. To clear the benchmark floor and achieve a meaningful Final Score $> 50$, the system requires $\text{ERS} \ge 0.50$, implying $\text{TPOT} \le 3.64\text{ ms}$ ($\ge 275\text{ tokens/sec}$) and $\text{TTFT} \le 124.0\text{ ms}$.

# Memory Bandwidth Physics

In single-stream autoregressive decoding, every generated token requires streaming the full model weights through memory exactly once.

| Model | Q8 Weight Footprint | Apple Silicon Memory Bandwidth | Theoretical Min TPOT | Max Decode Throughput | ERS $\text{tpot\_score}$ Ceiling |
|---|---|---|---|---|---|
| **LFM2.5-2.6B** | $\approx 2.6\text{ GB}$ | $90\text{ GB/s}$ (Base M-series sustained)<br>$200\text{ GB/s}$ (M4 Pro sustained) | $\mathbf{28.8\text{ ms}}$<br>$\mathbf{13.0\text{ ms}}$ | $34.7\text{ tok/s}$<br>$76.9\text{ tok/s}$ | **`0.0`** (Physically impossible)<br>**`0.0`** (Physically impossible) |
| **LFM2.5-350M** | $\approx 0.35\text{ GB}$ | $90\text{ GB/s}$ (Base M-series sustained)<br>$200\text{ GB/s}$ (M4 Pro sustained) | $\mathbf{3.88\text{ ms}}$<br>$\mathbf{1.75\text{ ms}}$ | $257.7\text{ tok/s}$<br>$571.4\text{ tok/s}$ | $\mathbf{0.46}$ ($\text{Final Score} \approx 46$)<br>$\mathbf{0.84}$ ($\text{Final Score} \approx 84$) |

### Architectural Implication
- For **2.6B**: Regardless of compiler sophistication, kernel fusion, or runtime zero-copy bridges, the hardware unified-memory bandwidth cannot mathematically clear the $10.0\text{ ms}$ TPOT ceiling on laptop-class Apple Silicon. The metric would remain permanently pinned near zero.
- For **350M**: The physical hardware bandwidth is capable of clearing the benchmark floor and achieving an ERS score above 50.

# Root Cause of Current 350M Latency ($\approx 39.6\text{ ms}$)

While the 350M hardware limit is $\sim 1.75 - 3.88\text{ ms}$, baseline runs currently observe $\sim 39.6\text{ ms}$ TPOT. This gap is entirely software runtime overhead:
1. **Python `transformers` Per-Token Forward Loop**: Calling Python-side forward iterations incurs Python interpreter overhead, PyTorch dispatch tables, and autograd overhead ($\sim 15 - 25\text{ ms}$ CPU overhead per token).
2. **Layer-by-Layer MPS Kernel Launches**: Dispatches 92 individual Q8 Metal linear kernels sequentially per token, incurring driver submission and barrier synchronization overhead.
3. **Prefill Inefficiencies**: Full multi-turn prompt ingestion without chunked GEMM or continuous KV-caching pushes TTFT over the $400\text{ ms}$ ceiling.

# Roadmap Enabled by this Decision

By standardizing on `LFM2.5-350M`, compiler optimization efforts have a measurable headroom to advance ERS toward $> 50$:
1. **Persistent C++/Metal Decode Engine**: Replace Python `generate()` loops with a fused C++/Metal command buffer dispatch that executes all 16 layers in a single hardware submission.
2. **Epilogue & Layer Fusion**: Fuse Q8 linear projections with activation functions (SiLU/GELU) and residual adds to eliminate intermediate VRAM roundtrips.
3. **Static KV-Cache & Chunked Prefill**: Lower TTFT below $100\text{ ms}$ for multi-turn conversational contexts.
