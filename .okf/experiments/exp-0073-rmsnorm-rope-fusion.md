---
type: Experiment
title: 'Fused RMSNorm and RoPE'
description: 'Compiler pass fuses twelve 10-command RMSNorm-RoPE chains into single Metal SIMD kernels, reducing plans by 108 commands.'
tags: [experiment, compiler, ocaml, metal, q8, rmsnorm, rope, fusion, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T11:29:00Z' }
sources:
  - id: implementation
    resource: /lib/passes.ml
    title: RMSNorm-RoPE pattern matching and fusion
  - id: kernel
    resource: /lib/metal.ml
    title: Generated SIMD RMSNorm-RoPE kernel
  - id: schedule
    resource: /lib/serving_schedule.ml
    title: Serving schedule serialization and shape inference
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-rms-rope-compiler-2026-08-25.txt
    title: Compiler and exact Metal evidence
---

# Boundary and motivation

The LFM2.5 attention block calculates rotary position embeddings (RoPE) immediately following
query/key RMSNorm and head transposition. In un-fused graphs, each of the twelve attention
query and key paths consists of a 10-command sequence: float32 cast, RMSNorm, transpose
(`axis0=1, axis1=2`), direct cosine multiplication, low/high half slices, negation,
concatenation, sine multiplication, and addition.

The `Passes.fuse_rms_rope` optimizer pass traces these subgraphs and folds each entire 10-command
sequence into a single typed `Rms_rope` IR operation.

# Lowering and Metal kernel

The generated Metal kernel `llmopt_rms_rope_f16_simd_h64` assigns one 32-lane SIMD group per
query/key token row. It computes the root-mean-square norm, applies weights, performs the
half-dimension split and sign inversion on rotated components, multiplies cosine and sine tables,
and stores the final transposed FP16 values directly into attention layout.

Package ABI v12 and serving schedule version 12 serialize `Rms_rope` parameters (epsilon and
half dimension) while preserving backward compatibility with versions 1 through 11.

# Structural result

Across the 6 attention layers of LFM2.5-350M (12 query/key paths per stage), the pass turns
twelve 10-command chains into twelve kernels:

- Prefill plan: 810 → 702 commands (74 package kernels, 0 opaque operations)
- Decode plan: 864 → 756 commands (72 package kernels, 0 opaque operations)
- Direct-paged decode specialization: 756 → 696 commands (13 runtime inputs, 6 paged-Q8 attention operations)

All 16 FFNs continue executing their body as three Q8 kernels:
1. `w1` + SiLU
2. standalone `w3`
3. `(w1_output × w3_output)` + `w2` + residual

Workspaces:
- Prefill lengths 13, 128, and 4,096: 676,864, 5,525,504, and 184,680,448 bytes
- Decode past lengths 1, 127, and 4,095: 171,008, 172,800, and 199,424 bytes

# Exact execution evidence

One preflighted Apple M4 Pro device probe dispatched 178 commands and 59 kernels from a
12,800-byte workspace, verifying 49 exact fixture outputs including `rms-rope-reference: exact`
and `paged-attention: exact`. Full Ninja test targets, Python tests, and OCaml tests pass.

# Evidence boundary

This experiment contains compiler, package, and exact synthetic-device evidence. The complete
350M ERS and needle comparison has not yet been rerun after RMS–RoPE, so there is no measured
latency or ERS delta for this latest fusion.
