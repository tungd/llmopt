---
type: Experiment
title: Mellum GGUF to EXL3 conversion and native integration
description: Explicit EXL3 storage binding, bounded approximate Metal encoding, and serialized Mellum execution evidence.
tags: [exl3, mellum, metal, benchmark]
status: stable
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
The compiled model and throughput comparison completed as recorded below.

The full `ninja -f ninja.build -j1 all test` suite passed: 97 Python tests and
native checks. Retained implementation receipt:
`bench/results/exl3-mellum-integration-2026-10-30.json`.

# Compiled bundle and measured comparison

Implementation commit `7482643` is on `main` and was used for the measurement.
The user selected the existing compiled bundle, not one executable embedding
all weights. The self-contained bundle is at
`_artifacts/mellum21-exl3/engine` in the main checkout; `serve` launches the
included executable with that Model Program. It contains the EXL3 archive,
3.46 MB tokenizer, compiled Metal libraries, and relative entrypoint links.
It has no dependency on the source GGUF. Relocation after measurement retained
the original worktree path as a symlink and passed Model Program/package checks.
Both packages have 108 declared kernels and zero opaque operations; prefill has
1200 schedule commands, decode 1256, and each binds a 932-entry tensor store.

All our heavy stages ran serially. The controller waited for other detected
compiler processes before launch. The comparison used a 64-token prefill and
64-token generation; llama-bench ran three repetitions. LLMOpt used warmup and
synthetic IDs 1..64 for prefill and 1,2,3,4 for decode. llama-bench uses its own
synthetic inputs, so this is a workload-level throughput comparison, not a
matched-token quality test.

| Metric | LLMOpt EXL3 | llama.cpp source Q4_K_M | Ratio |
| --- | ---: | ---: | ---: |
| Prefill, 64 tokens | 18.9487 tok/s | 435.0386 tok/s | 4.3556% |
| Steady decode, 64 generated tokens | 11.0031 tok/s | 88.2226 tok/s | 12.4719% |
| Decode including every inter-token gap | 11.0144 tok/s | 88.2226 tok/s | 12.4848% |

LLMOpt TTFT was 0.235699 s, total measured generation 5.955496 s, and end-to-end
completion throughput 10.7464 tok/s. Median inter-token latency was 88.794 ms,
maximum 178.261 ms. The steady metric excludes the first gap; both definitions
are retained. Load times were 23.3897 s for the prefill process and 10.0446 s
for the decode process, separate from the measured request. Synthetic generated
IDs repeatedly alternate 28 and 61048; these receipts do not establish semantic
quality parity. The observed ratios are below the user's 95% throughput target.

Raw outputs and exact commands are retained in
`bench/results/mellum21-exl3-vs-llamacpp-2026-10-30.json` and adjacent
`mellum21-exl3-2026-10-30-*.txt`. The receipt includes the model/Metal-library,
serving-executable, tokenizer, and archive hashes. Converter payload temporaries
were removed only after hashing the complete archive; provenance and conversion
measurements remain under `_artifacts/exl3-mellum` in the implementation worktree.

The implementation currently sends each routed token/expert pair through the
vector schedule, including prefill. Grouping routes by expert and lowering
through the existing matrix schedule is a concrete next compiler improvement;
no speedup from it has been measured here. No separate model-specific kernel
family was introduced by this integration.
