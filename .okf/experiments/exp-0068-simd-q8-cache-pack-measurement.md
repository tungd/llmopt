---
type: Experiment
title: 'SIMD-group Q8 cache-pack model measurement'
description: 'Measure SIMD-group attention and recurrent Q8 cache packing on the bounded LFM2.5-350M trace.'
tags: [experiment, compiler, ocaml, metal, q8, kv-cache, benchmark, ers, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T10:18:13Z' }
sources:
  - id: measurement
    resource: /bench/results/lfm25-350m-q8-simd-cache-pack-measurement-2026-08-25.txt
    title: SIMD Q8 cache-pack native observation
  - id: compiler
    resource: /bench/results/lfm25-350m-q8-simd-cache-pack-compiler-2026-08-25.txt
    title: SIMD Q8 cache-pack compiler evidence
  - id: previous-q8
    resource: /bench/results/lfm25-350m-q8-paired-simd-measurement-2026-08-25.txt
    title: Previous paired Q8-cache observation
  - id: previous-fp16
    resource: /bench/results/lfm25-350m-q8-paired-simd-fp16-kv-measurement-2026-08-25.txt
    title: Previous paired FP16-cache observation
---

# Bounded trace

The SIMD-cache-pack package completes 4/4 warmup and 4/4 scored requests,
preserves every established full-Q8 eager token sequence, and reports the same
80/194 radix reuse. ERS is `0.4021550914067862`, median TTFT is
`73.13212499138899 ms`, and median TPOT is `7.307645835680887 ms`.

Against the preceding paired Q8-cache observation, ERS changes by
`-0.004860666959368376`, median TTFT by `-2.092812501359731 ms`, and median
TPOT by `+0.37099299758362303 ms`. Mean TTFT/TPOT changes by
`-0.7675520319025964/+0.14051741648775806 ms`. Request-level deltas are mixed.

Against the separate FP16-cache observation, current Q8 ERS is
`0.02754812366853393` lower and median TTFT/TPOT is
`3.9688959950581193/0.6094374984968454 ms` higher. Q8-group-64 remains the
default serving policy.

# Supervision

The one 240-second attempt started at 52% free memory with no resident model
or native server. Memory was 45% free after warmup and after exit; sampled
server RSS was 112,256 KiB. The intentionally terminated server exited 143,
the supervising attempt exited 0, port 18094 was released, and no server
remained resident.

# Evidence boundary

The current and comparison reports are separate, non-interleaved single
observations. The attempt did not load eager PyTorch or run the long-context
needle matrix; exact parity uses the established full-Q8 eager IDs.
