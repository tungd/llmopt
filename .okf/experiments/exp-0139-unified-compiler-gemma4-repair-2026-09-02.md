---
type: Experiment
title: 'Unified compiler Gemma 4 repair and queue-collapse diagnosis'
description: 'Records constructive Q4_0 GELU lowering, unambiguous legacy accounting, AOT queue deltas, the Metal ICB collapse isolation and direct-command repair, widened native fixtures, and the corrected Gemma generation-policy boundary.'
tags: [experiment, gemma-4, unified-compiler, constructive-lowering, q4-0, gelu, aot-command-queue, metal, icb, diagnosis]
status: stable
generated: { by: codex/gpt-5.6, at: '2026-09-02T12:53:59+07:00' }
sources:
  - id: repair-receipt
    resource: /bench/results/uoc-gemma4-repair-2026-09-02.json
    title: Compiler, device-fixture, runtime, and reference repair receipt
  - id: transition
    resource: /experiments/exp-0138-unified-compiler-gemma4-transition-2026-09-02.md
    title: Original one-shot transition run
  - id: decision
    resource: /decisions/constructive-primitive-loop-lowering-and-gpu-queue-compilation.md
    title: Constructive assignment lowering and AOT GPU command-queue decision
  - id: f32-constant-probe
    resource: /bench/results/gemma4-f32-reference-constant-probe-2026-09-02.json
    title: F32 embedding and rotary constant repair plus one-token diagnostic
  - id: matched-policy-correction
    resource: /experiments/exp-0140-gemma4-matched-policy-diagnostic-2026-09-03.md
    title: Matched-policy result and constructive-transition correction
---

# Scope

> Correction (2026-09-03): the zero-legacy and unified-lowering accounting
> below did not mean that `Kernel_ir` drove backend code generation. The copied
> operation-family matchers still produced macro `Ir.Op` nodes, so the legacy
> passes have been restored. The authoritative matched-policy result and the
> corrected compiler boundary are recorded in exp-0140.

This follow-up diagnoses the unfavorable observations in the immutable
transition receipt without redefining its one-shot result. It treats fusion
regions and ready antichains as separate compiler operations derived from the
same dependency model, uses the probed target profile as the tactic cost model,
and refers only to the AOT compiler queue when reporting stages, dependencies,
and barriers.

# Compiler repair

The captured Gemma feed-forward branch is Q4_0
`Linear -> GELU -> Mul`, followed by a distinct Q4_0 down projection. The
constructive primitive model previously represented only SiLU and Sigmoid, so
the shared-input gate/up join could not lower through the unified path. It now
represents GELU explicitly, accepts Q4_0 through the target's SIMD32 and
threadgroup facts, and emits `llmopt_q4_0_gated_linear_f16`. The directional
candidate classifier also stops treating the later `Mul -> Linear` edge as an
unfinished gated join.

Both gate and up accumulators are rounded to FP16 before GELU and multiplication,
matching the two materialized Linear outputs. Their block decode shares one
activation traversal. The device fixture covers `m=37`, `k=3840`, and two
outputs and is byte-exact to the materialized Q4_0/GELU/multiply chain.

Receipt schema 2 distinguishes matcher invocation from graph rewrites;
`legacy-count` remains a compatibility alias for rewrite count. The repaired
Gemma receipts record zero invocations and zero rewrites for every covered
legacy matcher.

| Graph | Accepted | Cut | Unified | Legacy invocations | Legacy rewrites | Linear tactics | AOT stages | Dependencies | Barriers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prefill | 144 | 2,568 | 48 | 0 | 0 | 233 | 1,017 | 2,714 | 918 |
| Decode | 144 | 2,664 | 48 | 0 | 0 | 233 | 1,065 | 2,954 | 966 |

Relative to the transition artifact, each graph has 96 fewer AOT stages, 192
fewer dependencies, and 96 fewer barriers. These are compiler-queue counts,
not serving-request queue measurements.

# Decode collapse and command realization

The original completion first became token `0` at one-based completion position
29, when the total sequence reached 64 tokens. The same precompiled dispatch
list returned token `31436` through direct serial encoding and token `0` through
the Metal ICB. Adding ICB barriers before every indirect command did not change
that result. Reusing the length-64 specialization directly at later lengths was
also invalid because its schedule, launch dimensions, workspace bindings, and
cache parameters were specialized to that length.

The serving engine now obtains the decode schedule for the current prefix on
every step. `Metal_runtime.Prebaked` retains the precompiled records but encodes
them directly into one serial compute encoder, patches dynamic paged-attention
and checkpoint parameters, and emits explicit buffer barriers. Paged-attention
recognition uses the kernel-family prefix rather than three hard-coded names.
Two sequential 36-token runs after this repair produced the same stream, no
token `0`, and `2.80` and `2.74 tok/s` respectively.

The AOT queue remains the schedule's kernels, dependencies, stages, and
barriers. Metal ICB is a backend realization tactic, not the definition of that
queue, and the serving SRPT request queue is a separate runtime policy.

# Native semantic probes

The native fixture now additionally exercises:

- Q4_0 GELU gated Linear at the 37-token, 3,840-wide Gemma prefill shape;
- two-query causal FP16-mask GQA with 256-wide heads; and
- paired Q/K RMSNorm-RoPE with 256-wide heads.

The gated and Q/K fused outputs are byte-exact to their materialized GPU chains;
the GQA output is exact for both query positions and shared KV-head mapping. A
separate GGUF-backed Q4_0 Linear check at `m=2`, `n=4096`, `k=3840` matched
8,156 of 8,192 FP16 elements exactly, had maximum absolute delta
`0.000244140625`, and preserved both row argmaxes against GGUF Python
dequantization.

# Reference boundary

For the exact 37 prompt IDs, the same-GGUF llama.cpp generated stream begins
`100,45518,107,14947,135383`. The transition LLMOpt raw stream begins
`258882,86559,43208,146894,141156`, and the repaired fused artifact begins
`505,98677,236888,207451,236779`. This comparison originally treated token 1
as a full-forward mismatch, but the GGUF declares
`tokenizer.ggml.suppress_tokens = [258883, 258882]` and llama.cpp applies that
model policy while LLMOpt's compiled greedy path does not. The first transition
token is therefore not a matched-policy comparison.

The Gemma builder also rounded `sqrt(hidden_size)`, full-attention inverse
frequencies, and sliding-attention inverse frequencies to FP16. Their archive
bindings now preserve the source constants as FP32, and the captured rotary
paths consume the inverse frequencies as FP32. The embedding graph still casts
its scalar to FP16 before multiplying the FP16 lookup result, so this change
does not establish llama.cpp embedding-arithmetic parity. A fresh build produced
valid zero-opaque prefill/decode packages and one controlled 37-token probe
returned raw LLMOpt token `258882` in `8.429582 s`, versus generated llama.cpp
token `100`; `258882` is one of the two suppressed GGUF tokens. The F32
correction therefore did not establish generated-token parity, and the next
unsuppressed argmax was not captured.

For a single BOS token, both prior runs selected `236749`. That observation
does not exclude embedding, normalization, RoPE, attention, or feed-forward
differences on the 37-token path; single-token attention has no competing
historical keys. The configured attention scale is already `1.0`, and the GGUF
full/sliding RoPE bases are `1,000,000` and `10,000`, matching the captured
Gemma configuration.

Evidence: the builder had an FP16 source-constant mismatch and the observed raw
token `258882` is suppressed by llama.cpp; assumption: attributing the remaining
generated-token difference to arithmetic drift is unproven until a
matched-policy top-k or logit comparison isolates it.

This experiment makes no full-model token-parity or matched-throughput claim.
