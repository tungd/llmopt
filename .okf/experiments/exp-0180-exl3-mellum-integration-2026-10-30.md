---
type: Experiment
title: Mellum GGUF to EXL3 conversion and native integration
description: Explicit EXL3 storage binding, bounded approximate Metal encoding, and serialized Mellum execution evidence.
tags: [exl3, mellum, metal, benchmark]
status: draft
generated: { by: codex/gpt-6, at: '2026-10-30T00:00:00Z' }
sources:
  - id: canonical
    resource: /decisions/exl3-canonical-runtime.md
    title: User-approved EXL3 direction
  - id: source-model
    resource: https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF
    title: Original Mellum GGUF checkpoint
  - id: local-receipts
    resource: _artifacts/exl3-mellum
    title: Serialized conversion, build, native integration, and model benchmark receipts
---

# Scope

Start with `Mellum2.1-12B-A2.5B-Thinking-Q4_K_M.gguf`, acquired with authenticated
`hf download --max-workers 1`. Convert with `bench/convert_gguf_exl3.py`, capture
with the existing Mellum FX builder, and link through `llmopt-pipeline`.
No prequantized EXL3 checkpoint is substituted.

# Representation and conversion

Archive dtype tag 17, `EXL3_4`, stores a logical `[N,K]` or `[E,N,K]` projection.
The packed payload occupies half a byte per weight; expert payloads are
contiguous. Each expert uses K-major 16x16 trellis tiles. The existing `.exl3`
metadata records the common geometry; `.suh` and `.svh` contain F16 signed
scales, with an expert axis for rank-three weights. Archive loading checks
metadata, physical length, and companion shapes.

All aligned attention, output-head, and expert projections use EXL3_4.
Embeddings, norms, and small router matrices remain dense. The converter
streams at most 1024 output rows from one expert, with a hard worker bound of
8,388,608 elements. It retains resumable tensor files and provenance, then
writes an atomic archive. Extra RoPE constants come from the existing archive;
its quantized weights are not reused.

This encoder uses deterministic random signs, per-output RMS scales, normalized
128-wide Hadamard transforms, and a two-step greedy lookahead with three
circular sweeps. It is an approximate mul1 encoder, not ExLlamaV3's calibrated
optimizer. Normalized rotated-weight relative RMSE is a conversion metric;
it does not measure generated text quality or compiler arithmetic error.
The first real `blk.0.attn_k.weight` conversion measured 0.189985 relative RMSE.

# Runtime

One projection interpreter handles ordinary and expert-indexed weights.
It emits scale/input transform, packed contraction, and output transform.
Ordinary rows select vector or matrix schedules; routed rows select the same
vector schedule with an expert index. Top-k routing, SwiGLU, and weighted expert
reduction compose these projections. Scratch buffers belong to the execution
batch and are reused between projections. This initial routed schedule does
not sort experts or reuse expert weights across routed tokens.

The legacy affine path remains for existing archives. EXL3 bypasses affine
contraction fusion and uses explicit stages; this experiment does not claim
that legacy affine code has been removed.

# Evidence

The dynamic-geometry generated stages passed both retained upstream-oracle
fixtures, including exact codebook and packed decoding and complete linear
outputs. Full-model conversion completed in 515.505 seconds including process startup:
197 quantized tensors among 339 original tensors, 6,465,384,192 archive bytes,
aggregate normalized relative RMSE 0.190921. Native archive binding, schedule
serialization, ordinary projections, and 3-expert/top-2 MoE passed at 1, 3, and
17 rows. Largest absolute error was 1.52588e-5 (ordinary) and 7.62939e-6 (MoE).
The full-model bundle and throughput comparison follow these implementation checks.

The full `ninja -f ninja.build -j1 all test` suite passed: 97 Python tests and
native checks. Retained implementation receipt:
`bench/results/exl3-mellum-integration-2026-10-30.json`.
