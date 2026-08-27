---
type: Experiment
title: 'W4A16 RMSNorm and SwiGLU cast absorption'
description: 'Remove single-use f16-to-f32 widening casts before native FP16-accepting RMSNorm and W4A16 SwiGLU regions, then replan the preserved serving pair.'
tags: [experiment, compiler, fusion, w4a16, kvq8, rmsnorm, swiglu, cast, metal]
status: draft
generated: { by: codex/gpt-5.6, at: '2026-08-27T04:40:00Z' }
sources:
  - id: rms-pass
    resource: /lib/pass_fuse_rms_norm.ml
    title: RMSNorm fusion and single-use widening-cast absorption
  - id: swiglu-pass
    resource: /lib/pass_fuse_swiglu_ffn.ml
    title: declarative W4A16 SwiGLU rule with optional cast input
  - id: query-engine
    resource: /lib/fusion_query.ml
    title: typed cast operation and input-level alternation
  - id: rope-pass
    resource: /lib/pass_fuse_rms_rope.ml
    title: direct f16 RMSNorm compatibility for RMS-RoPE fusion
  - id: receipt
    resource: /bench/results/lfm25-350m-w4a16-kvq8-rms-cast-absorption-2026-08-27.txt
    title: offline package and dispatch-count audit
  - id: native-receipt
    resource: /bench/results/lfm25-350m-w4a16-kvq8-rms-cast-absorption-vs-restored-2026-08-27.json
    title: fresh native cast-absorption versus restored trace
  - id: candidate-prefill
    resource: /_artifacts/w4-engine-2026-08-27-rms-cast/prefill/plan.txt
    title: candidate prefill plan
  - id: candidate-decode
    resource: /_artifacts/w4-engine-2026-08-27-rms-cast/decode/plan.txt
    title: candidate decode plan
---

# Cast absorption

The preserved FX capture emits an f16-to-f32 cast before every traced
RMSNorm-shaped boundary. The generated Metal RMSNorm and W4A16 SwiGLU kernels
already read FP16 inputs and accumulate the reduction in FP32 registers, so a
cast is removable when its f32 result has exactly one consumer: the RMSNorm
node being replaced or the matched SwiGLU region. The rewrite keeps all other
casts, including f32/i64 position and mask conversions.

The rule engine now exposes `cast` as an operation and supports an input-level
`or` pattern. The SwiGLU query can therefore capture either a direct f16
activation or the original f16 activation behind a validated cast; the cast is
included in the region member set and disappears with the rewrite. The RMSNorm
pass applies the same single-consumer check after its existing decomposed-chain
replacement. RMS-RoPE accepts both direct f16 and casted RMSNorm inputs so
absorbing a cast does not disable that fusion.

# Static replan

The candidate engine is built from the existing W4A16/KVQ8 graphs and shared
322,667,136-byte archive. Compared with the restored baseline package:

| Stage | Commands | Kernel entries | f16->f32 casts | W4A16 SwiGLU operations | Opaque |
|---|---:|---:|---:|---:|---:|
| Prefill baseline | 752 | 58 | 33 | 16 | 0 |
| Prefill candidate | 692 | 58 | 0 | 16 | 0 |
| Decode baseline | 771 | 55 | 33 | 16 | 0 |
| Decode candidate | 708 | 55 | 0 | 16 | 0 |

The specialized decode schedule is 479 commands versus 512 from the baseline
pair. All 33 widening casts are gone in each candidate plan; the remaining
five f32 casts are position/mask conversions from f32 or i64 inputs. The
command delta is larger than 33 because the co-scheduled plans no longer need
the barriers around those removed dispatches.

`ninja -f ninja.build all test`, both candidate package checks, and
`llmopt-lfm-serving-check` pass.

# Native comparison

One fresh supervised HTTP trace then ran the restored engine and the
cast-absorption candidate sequentially on the same host, with four warmup and
four scored requests per engine. Both completed 4/4 scored requests with zero
output-token mismatches and 80 cached prompt tokens. The full receipt is
`bench/results/lfm25-350m-w4a16-kvq8-rms-cast-absorption-vs-restored-2026-08-27.json`.

| Engine | ERS | median TTFT | median TPOT |
|---|---:|---:|---:|
| Restored baseline | 0.6029898047 | 65.8663335 ms | 3.8495070 ms |
| Cast absorption | 0.5858433138 | 61.8422705 ms | 4.1920208 ms |
| Candidate minus baseline | -0.0171464909 | -4.0240630 ms | +0.3425138 ms |

This is one sequential sample per engine rather than repeated or
counterbalanced sampling; it records a mixed latency result, with lower TTFT
but higher TPOT and lower aggregate ERS for the candidate.
