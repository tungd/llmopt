---
type: Experiment
title: 'Q8 linear-SiLU epilogue fusion'
description: 'Fuse the 16 LFM feed-forward Q8 linear and SiLU pairs into typed Metal GEMM/GEMV epilogues while preserving alias safety and float16 rounding semantics.'
tags: [experiment, compiler, ocaml, metal, q8, fusion, silu, prefill, decode, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T06:22:00Z' }
sources:
  - id: pass
    resource: /lib/passes.ml
    title: Alias-safe Q8 linear-SiLU graph rewrite
  - id: emitter
    resource: /lib/metal.ml
    title: Fused tiled GEMM and one-row GEMV Metal epilogues
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Typed fused-operation dispatch
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-linear-silu-fusion-2026-08-25.txt
    title: Offline model replan and bounded device observation
---

# Optimization boundary

Each of the model's sixteen feed-forward blocks contains a Q8 `w1` projection
whose sole consumer is SiLU. The previous schedule wrote the projection to the
Metal workspace and launched a separate pointwise kernel to read, activate,
and write it again.

# Typed rewrite and ABI

`Passes.fuse_q8_silu` replaces only adjacent `Q8_linear -> SiLU` pairs whose
intermediate value has no second consumer. It retains the SiLU node's output
identity and transfers the Q8 inputs to `Ir.Op.Q8_linear_silu`; a regression
case with a separately named raw Q8 output remains unfused.

Schedule ABI v9 adds the fused opcode and package ABI v9 adds its kernel-family
tag. Readers retain schedule v1-v8 and package v2-v8 compatibility. LFM
sequence specialization rewrites the fused operation's dynamic `m`, `n`, and
`k` dimensions exactly as it did for unfused Q8 linear.

# Metal and LLVM lowering

The generated Metal family contains float16 and float32 tiled GEMM+SiLU and
one-row GEMV+SiLU entries. Bias is accumulated before activation. The float16
form explicitly converts the accumulator to `half` before converting it back
to float for SiLU, preserving the original Q8 output materialization rounding
point. The LLVM inspection emitter records the fused exponential epilogue.

# Saved-model structure

Offline recompilation of the preserved binary graphs found exactly sixteen
pairs in each stage:

| Stage | Commands before | Commands after | Fused pairs | Standalone SiLU | Kernel entries |
|---|---:|---:|---:|---:|---:|
| Prefill | 872 | 856 | 16 | 0 | 52 |
| Decode | 926 | 910 | 16 | 0 | 50 |

Both packages remain zero-opaque, validate all 241 bindings against the shared
422,137,216-byte binary tensor archive, and compile/link with Xcode Metal.
The persistent artifact pair is
`_artifacts/lfm25-350m-q8-prefill-decode-binary-v1-abi9-silu-v1-2026-08-25`.
Its prefill and decode archives share inode `241927011`; both Q8-group-64 and
selectable-FP16 package-pair checks pass.

# Device and final static evidence

At 47% free memory with no resident model, Torch, or native-server process,
one 30-second-supervised Apple M4 Pro fixture executed 137 commands and 40
kernels from a 10,240-byte workspace. All 41 outputs matched their expected
bytes, including `llmopt_q8_gemv_silu` output bits `3f0c 47ff b461`.

After that one device attempt, the final emitter made the float16
materialization point explicit and the fixture added the standalone path as a
byte-for-byte reference. The current 139-command package compiles and validates
with 49 entries, zero opaque commands, a 10,496-byte workspace, and 41
allocations. That strengthened 42-output fixture was not launched.

# Measurement boundary

No model execution or ERS scoring ran for this pass. The latest valid native
ERS remains `0.11381808711306604` from
[cache-submission batching](exp-0044-cache-submission-batching.md).
