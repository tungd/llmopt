---
type: Decision
title: 'EXL3 canonical weight representation and generated Metal schedules'
description: 'Use EXL3 as the canonical runtime weight format, accept approximate import conversion, and pursue at least 95 percent of the selected llama.cpp throughput through reusable compiler schedules.'
tags: [decision, exl3, quantization, metal, compiler, performance]
status: stable
decision_status: approved
generated: { by: codex/gpt-6, at: '2026-10-29T18:39:51Z' }
sources:
  - id: user-direction
    resource: 'User conversation on 2026-10-30: EXL3 canonical format, approximate conversion, 95 percent llama.cpp performance target, and approval of first implementation slice'
    title: User-approved architectural direction and first slice
  - id: upstream-reference
    resource: https://github.com/turboderp-org/exllamav3/blob/cdc785db89e263e65e5849cb53e1ada156411414/tests/testlib/trellis.py
    title: Pinned independent EXL3 packing and linear CPU oracle
  - id: previous-format
    resource: /decisions/canonical-tile32-affine-and-hadamard-attention-rotation.md
    title: Superseded canonical weight-format choice
---

# Decision

EXL3 is the intended canonical runtime **weight** representation. Conversion
from an existing quantized checkpoint is approximate; lossless GGUF conversion
is not a runtime architecture constraint. The performance objective is at least
95% of a specified llama.cpp baseline's throughput on the target workload.
Conversion error is recorded separately from compiler arithmetic error and
performance. No new performance threshold is introduced for intermediate work.

The existing capture/compiler/link/runtime pipeline remains the integration
path. Multiple generated specializations are expected; independently maintained
model-specific kernel bodies are not the architecture. Schedule choices own
SIMD mapping, tile geometry, staging, and fusion. Packed decode is shared.

This supersedes the previous Tile32_Affine canonical **weight-format** choice.
Existing affine/GGUF execution paths remain during migration. This decision does
not change KV-cache representation. The subsequent integration adds an approximate
GGUF encoder and explicit ordinary/expert runtime binding; see the
[Mellum integration experiment](/experiments/exp-0180-exl3-mellum-integration-2026-10-30.md).

# First implementation slice

`lib/exl3.ml` owns an abstract integer-rate mul1 descriptor and validates archive
components. The existing ABI-v1 archive stores these components without a new
physical dtype or format-version change:

| Component suffix | Dtype and physical shape | Meaning |
| --- | --- | --- |
| `.exl3` | U8 `[32]` | Magic `EXL3MUL1`, then six little-endian u32 fields: version=1, bits, K, N, Hadamard width=128, reserved=0 |
| `.trellis` | U8 `[K/16,N/16,32*bits]` | Upstream packed little-endian 32-bit words, with MSB-first ring bits and tensor-fragment permutation |
| `.suh` | F16 `[K]` | Signed input scales |
| `.svh` | F16 `[N]` | Signed output scales |

Supported descriptor dimensions are positive multiples of 128; integer bits
1..8 share one decoder. The retained fixtures exercise 3 and 4 bits, including
unaligned/circular windows. Fractional rates and other codebooks are not yet
implemented. A window can be extracted independently from two packed words;
decoding need not execute a sequential trellis recurrence.

The logical weight is `diag(suh) H128 W_hat H128 diag(svh)`, with normalized
blockwise Sylvester transforms. The implementation computes input scale/transform
in F32, rounds that intermediate to F16, accumulates contractions into F32,
then applies the output transform, scale, existing algebraic epilogue, and final
F16 rounding. It uses three explicit stages so transform costs are measured.
The vector and matrix schedules share the same packed decoder.

This slice emits and runs the contraction through a dedicated probe. Full FX
storage classification, Model Program binding, grouped/MoE contractions, and
model conversion/integration are subsequent work, not completed support.

# Verification and measurements

`test/fixtures/exl3` contains reproducible archives and provenance receipts from
the pinned upstream CPU reference. Random packed bits are valid trellis inputs;
they are not quantized trained weights and say nothing about model quality.

`python3 bench/run_exl3_probe.py` builds and runs one job at a time, with process
preflight, subprocess timeouts, logs, and a final receipt. It checks all 65,536
mul1 codebook outputs and every fixture weight exactly, then checks the complete
linear result against a stage-aware independent reference. GPU timings separate
input transform, contraction, output transform, and their combined command.
Small repeated fixtures are cache-resident tests, not DRAM or full-model claims.

# Model integration

The canonical model archive uses dtype tag 17 (`EXL3_4`) for logical `[N,K]`
and `[E,N,K]` weights with the same `.exl3`, `.suh`, and `.svh` companions.
The original component-only fixture representation remains supported. Runtime
archive checks reject mismatched descriptors, payload lengths, and scale shapes.
One projection interpreter serves ordinary and indexed expert contractions.

`bench/convert_gguf_exl3.py` converts aligned projections from GGUF with bounded
Metal requests. Its lookahead encoder is approximate and uncalibrated; it does
not reproduce the upstream optimizer. Dense embeddings/norms/routers remain
dense. The user selected the existing self-contained compiled bundle, rather
than embedding weights into one executable. Full-model throughput is recorded
separately from conversion error and upstream-reference arithmetic checks.
