---
type: Experiment
title: 'Hardware-derived prefill roofline and dynamic chunking cost model'
description: 'Derive prefill roofline arithmetic intensity knee, core saturation limits, and SLA-bounded chunk budgets dynamically from hardware discovery parameters, integrating them into the serving queue scheduler.'
tags: [experiment, cost-model, prefill, roofline, chunking, scheduler, serving]
status: complete
generated: { by: antigravity, at: '2026-08-28T16:10:00Z' }
sources:
  - id: target-hardware
    resource: /lib/target_hardware.ml
    title: hardware discovery and prefill cost model
  - id: server
    resource: /bin/lfm_serve.ml
    title: dynamic chunk budget integration in serving loop
  - id: test
    resource: /test/test.ml
    title: prefill cost model unit tests
---

# Hardware-Derived Prefill Roofline and Dynamic Chunking Cost Model

## Context

Prefill batch size directly dictates arithmetic intensity and GPU core occupancy. Small prompt batches ($M < 16$) are heavily memory-bandwidth bound ($T(M) \approx W / B$), leading to low arithmetic intensity ($< 4\text{ FLOP/B}$). In contrast, excessively large chunks can exceed SLA bounds ($> 25\text{ ms}$), causing decode jitter for concurrent streams.

## Implementation

We implemented `Target_hardware.Prefill_cost_model`:

1. **Roofline Knee ($M_{\text{knee}}$)**:
   $$M_{\text{knee}} = \left\lceil \frac{W}{B} \times \frac{P}{2 \times \text{Params}} \right\rceil$$
   Evaluated using probed bandwidth $B$ (273 GB/s on M4 Pro) and peak FP16 compute $P$ (18.7 TFLOPS).
2. **Compute Saturation ($M_{\text{sat}}$)**:
   Calculates minimum batch size ($M_{\text{sat}} \ge 32$) to saturate all 16 GPU compute units and 32 SIMD lanes.
3. **SLA-Bounded Chunk Budget ($M^*$)**:
   Slices large prompts into optimal 64-token-aligned buckets ($M^* \in [64, 512]$) such that single-chunk prefill latency stays within interactive streaming budgets ($< 25\text{ ms}$).
4. **Serving Scheduler Integration**:
   `bin/lfm_serve.ml` dynamically queries `Target_hardware.Prefill_cost_model.analyze` to set `prefill_chunk_budget`, replacing static constants with hardware-adaptive scheduling.

## Verification

- Unit tests in `test/test.ml` verify roofline knee calculation ($M \in [8, 64]$), core saturation constraints ($M \ge 32$), 64-token boundary alignment, and JSON serialization round-trips.
- Verified on Apple M4 Pro: knee = 18 tokens, saturation = 64 tokens, optimal chunk budget = 256 tokens, estimated chunk latency = 19.18 ms.
