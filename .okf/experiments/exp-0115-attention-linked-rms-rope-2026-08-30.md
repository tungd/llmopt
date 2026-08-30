---
type: Experiment
title: 'Attention-linked token-major RMSNorm-RoPE fusion'
description: 'Fuse token-major captured RMSNorm and rotary topology, pairing Q/K branches through their shared downstream Attention.'
tags: [experiment, compiler, fusion, metal, rmsnorm, rope, attention, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T12:18:06+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-13-2026-08-30.json
    title: Slice 13 benchmark receipt
  - id: fusion
    resource: /lib/pass_fuse_rms_rope.ml
    title: Attention-linked RMSNorm-RoPE topology fusion
  - id: metal
    resource: /lib/metal.ml
    title: Float16 and Float32-weight SIMD RMSNorm-RoPE kernels
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Shape and dtype selected RMSNorm-RoPE dispatch
---

# Structural change

The existing RMSNorm-RoPE pass now recognizes both head-major and token-major
captured layouts. The token-major form follows producers from the final
token/head transpose through the rotary add, cosine/sine products,
split/negate/concat rotation, and RMSNorm. It accepts Float16 or Float32 norm
weights and either singleton-axis placement for contiguous trigonometric
tables.

Q/K branches are paired only when their captured outputs reach the same
semantic Attention through movement aliases. The pairing derives batch,
token, head, and width dimensions from the graph and compares alias roots for
the trigonometric inputs. It contains no default head counts, model/tensor
names, GGUF architecture, or llama.cpp architecture ID.

# Full-model result

| Probe | Slice 12 LLMOpt | Slice 13 LLMOpt | Dispatch change | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `12.558937 ms` | `12.533069 ms` | `1128 -> 1128` | `7.938250 ms` | `1.578820x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `28.671980 ms` | `26.480079 ms` | `1635 -> 1220` | `17.208500 ms` | `1.538779x` |

Gemma's 50 rotary branches become 15 paired Q/K dispatches plus 20 single
dispatches, eliminating 415 dispatches. Its slice-12 output changes by mean
absolute `0.0061832881` and maximum `0.0795898438`, while row argmax IDs remain
`84904,148465`. Qwen matches no new site, remains byte exact, and its timing
change is an observation rather than an attributed effect.

# Validation

The complete OCaml and Python suites, Xcode Metal compilation, native Metal
fixture, token-major alias/Attention regression, schedule round trip, both
package checks, zero-opaque audits, full-model output comparisons, and fresh
`llama-bench -p 2 -n 0 -r 10` runs pass.

Evidence is the semantic package inventory and retained timing/output records;
the latency attribution is an inference from Gemma's selected sites and Qwen's
zero-site control.
