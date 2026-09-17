---
type: Experiment
title: QK-only device parity, Gemma-shape microbenchmark, and decode kernel breakdown
status: stable
description: 'On-device QK-only QKV parity across Q4_0 and Q5_0 at Gemma decode shapes, a Gemma-shape microbenchmark, and a serial decode breakdown showing kernels far below the memory-bandwidth roofline (~84 of ~273 GB/s).'
tags: [experiment, qk-only, qkv, q4_0, q5_0, metal, parity, microbenchmark, kernel-breakdown, m4-pro]
generated: { by: 'autoagent/union-alpha', at: '2026-09-18' }
sources:
  - resource: /test/test_schedule.ml
    title: QK-only schedule and Metal device tests
  - resource: /bench/results/qk-profile-2026-09-18/qk-tests.txt
    title: QK test receipt
  - resource: /bench/results/qk-profile-2026-09-18/microbench.txt
    title: Gemma-shape QK microbenchmark receipt
  - resource: /bench/results/qk-profile-2026-09-18/baseline.txt
    title: Memory-stopped first full-model attempt receipt
  - resource: /bench/results/qk-profile-2026-09-18/normal-baseline.txt
    title: Normal (non-profiled) 16-token baseline receipt
  - resource: /bench/results/qk-profile-2026-09-18/current-run.txt
    title: Serial decode kernel breakdown receipt
  - resource: /_artifacts/gemma4-12b-qat-primitive-schedule-2026-09-03/source/target.weights.llmopt
    title: 6.70 GB Gemma weight archive (bandwidth denominator)
---

# QK-only fusion

The working slice recognizes two compatible Q4_0 or Q5_0 projections feeding
rotary math, with Q wider than K. It represents them using `Quant_qkv_linear`
with `n_v = 0`, three inputs and one extra output. Schedule validation,
serialization and sequence specialization support this form. Runtime preserves
the fixed QKV ABI by aliasing unused V buffer slots to K; zero V columns mean
no V work is dispatched.

## Device evidence

`test_quant_qk_device` in `test/test_schedule.ml` compiles emitted Metal, loads
a compiled-graph package, and executes a serialized/deserialized schedule.
Both Q and K match separate Linear execution byte-for-byte for Q4_0 and Q5_0
at `m = 1, 2, 4, 16, 33`, `k = 64`, Q width 19 and K width 11. The odd widths
and m=33 exercise partial column and row tiles. Fixtures use signed dyadic
activations and varied packed weights, with exactly representable arithmetic.
Each fused execution records one QKV kernel.

This is fixture parity, not arbitrary-input bit-exactness or full-model parity.
The device fixture constructs the fused op directly; the existing topology
fixture separately proves pass recognition and schedule roundtrip.

The initial Q5-only device fixture failed MSL compilation with
`unknown type name 'block_q8_0'`: the Q5 QKV catalog included mixed Q5/Q8
kernels without a Q8 declaration. Splitting homogeneous and mixed
source/entry lists and admitting mixed kernels only when Q8 source is present
fixed that failure. The combined catalog exports remain available.

## Gemma-shape microbenchmark (single pass, diagnostic)

A standalone probe at the layer-0 Gemma QK shape (m=1, k=576, Q=576, K=192)
timed 100 synced executions per graph after 10 warmups, using the production
memory plan and workspace. Both graphs remained bit-exact against separate
projection execution. Q4_0: separate median 154.0 us vs fused 132.8 us
(-14%). Q5_0: 152.1 vs 159.0 us (no measurable win; p90 spread 231-431 us
shows run-to-run noise at this scale). Single-run wall timing of one
`encode_schedule` includes host submission; kernels were not timed in
isolation. One run only.

## Verification

- `ninja -f ninja.build ocaml-test`: passed, including all ten new device cases.
- `ninja -f ninja.build test`: passed, 79 Python tests.
- `git diff --check`: clean.
- `ninja -f ninja.build all test`: blocked by unrelated Webs interface mismatch
  between the built `webs_iomux.cmi` and installed `webs.cmi`. Not repaired here.
- `ninja -f ninja.build _build/bin/llmopt-bench-tokens`: passed.

# Decode kernel breakdown of the exp-0173 package (diagnostic)

Two four-token attempts on the saved exp-0173 Gemma package were stopped by
the 25% free-memory guard during loading (47%/63% free falling to 24%); free
memory recovered to 63-65% and a later attempt completed. The successful run
set `LLMOPT_PROFILE_BREAKDOWN`, which submits each of the 714 decode
dispatches serially and waits, so its wall time (2.38 s/token mean) is not
throughput evidence. The saved package was not recompiled with the QK change:
these numbers diagnose the existing package/runtime, not QK speedup.
Receipt: bench/results/qk-profile-2026-09-18/current-run.txt.

Per-kernel GPU time at past_tokens=37 totals 79.8 ms — equal to the
72.4-79.8 ms normal median TPOT (16-token normal baseline this session:
72.4 ms median, receipt normal-baseline.txt). Since every kernel was
submitted serially and still matched normal batched latency, host
submission is not the limiter: decode spends ~all its time inside the
dispatched kernels. But "compute-bound" is the wrong frame: for each
sub-kernel the work is dominated by streaming weights from memory, and the
measured 79.8 ms over a 6.70 GB weight archive is ~84 GB/s — roughly 31% of
this M4 Pro's ~273 GB/s. Decode kernels are nowhere near the memory
bandwidth roofline; that is why halving dispatch count didn't help and why
full bandwidth is not yet utilized.

| Kernel | Calls | Total ms | Avg us | Pct |
|---|---|---|---|---|
| q4_0_gated_linear_f16_x8 | 48 | 23.8 | 495 | 29.8% |
| q4_0_linear_f16_x8 | 96 | 20.2 | 210 | 25.2% |
| q4_0_linear_f16_x16 | 17 | 6.6 | 388 | 8.3% |
| q4_0_qkv_linear_f16_x8 | 40 | 5.8 | 144 | 7.2% |
| paged attention h256 + h512 | 48 | 3.3 | — | 4.2% |
| rms_norm / rms_rope family | ~330 | ~11 | — | ~14% |

Findings:

1. Quantized linear projections (gated + plain + QKV) are ~70% of decode GPU
   time. The 40 fused QKV calls already avoid the V-stream reads that fusion
   targets; the remaining 96 x8 + 17 x16 plain Linear calls are the next
   fusion surface (multi-export / contraction-group widening).
2. Paged attention is only ~4-5% of decode GPU time in this breakdown. The
   h256/h512 kernels selected here match commit `4088329`'s specialization;
   that commit's own 4.5x attention receipt was not re-verified this session.
3. The gated x8 kernel (29.8%, avg 495 us) is the single largest target; its
   weight streaming across 48 FFN layers dominates.

Next: raise projection-kernel memory throughput toward the ~273 GB/s
roofline — larger column tiles for m=1 decode and fewer redundant weight
reads across the 96 plain x8 calls (multi-export / contraction-group
widening). Dispatch count is the wrong metric. The saved package was not
recompiled with the QK change; these numbers diagnose the existing package.

The 4 generated tokens `100, 45518, 107, 14947` match the exp-0173 golden
prefix. See [prior full-model result](exp-0173-gemma4-uoc-g4p12-2026-09-14.md).
