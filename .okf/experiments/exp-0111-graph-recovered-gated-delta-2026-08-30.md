---
type: Experiment
title: 'Graph-recovered zero-state gated-delta execution'
description: 'Replace a captured padded recurrence expansion with a semantic five-input operator and direct SIMD Metal execution.'
tags: [experiment, compiler, graph, recurrence, gated-delta, metal, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T10:56:48+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-9-2026-08-30.json
    title: Slice 9 benchmark receipt
  - id: fusion
    resource: /lib/pass_fuse_gated_delta.ml
    title: Captured-topology gated-delta recovery
  - id: metal
    resource: /lib/metal.ml
    title: Width-specialized gated-delta kernel
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Semantic gated-delta dispatch
---

# Structural deficiency

Qwen's two-token graph expanded each gated-delta layer into a padded 64-token
chunk algorithm. Earlier triangular recurrence fusion removed the 63-step
carried-state chain but retained its surrounding matrix, movement, indexing,
and pointwise graph. Eighteen layers therefore still accounted for a large
share of Qwen's command and dispatch overhead.

The captured graph already contains the semantic boundary: three Float32
`[batch,heads,tokens,width]` inputs, two Float32
`[batch,heads,tokens]` inputs, five axis-2 pads, and a final transposed
Float16 `[batch,tokens,heads,width]` result. The compiler now recovers this
boundary by producer/user topology and typed metadata. It reads no model name,
tensor name, GGUF architecture, or llama.cpp architecture ID.

# Compiler and kernel change

The pass first follows named graph outputs to discard dead expansion branches,
then identifies query, key, value, gate, and beta from their structural uses.
It replaces the live expansion with `Ir.Primitive.Gated_delta`. Package ABI 19
and schedule ABI 21 carry the operation through serialization and kernel
selection.

Metal lowering specializes only on captured state width. One SIMD group owns
one state-matrix row, keeps its row in registers, and performs the gated decay,
key correction, and query projection directly for the captured token count.
Other graphs retain their captured operations when the full typed topology does
not match.

# Full-model result

| Probe | Slice 8 LLMOpt | Slice 9 LLMOpt | Change | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `39.842010 ms` | `14.506459 ms` | `-25.335551 ms` (`-63.590042%`) | `7.937812 ms` | `1.827514x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `30.259609 ms` | `30.664444 ms` | `+0.404835 ms` (`+1.337873%`) | `17.2634165 ms` | `1.776267x` |

Qwen recovers all 18 sites and falls from 4,470 commands/2,407 dispatches to
2,723/1,272. Its logits differ from slice 8 by mean absolute `0.0021786743`
and maximum `0.017578125`, while row argmax IDs remain `760,16`. Gemma matches
no gated-delta region; graph-output liveness removes two dead commands and its
output remains byte exact.

# Validation

The complete OCaml suite, all 49 Python tests, Xcode Metal compilation, native
454-dispatch fixture, both package checks, zero-opaque audits, full-model output
comparisons, and fresh `llama-bench -p 2 -n 0 -r 10` runs pass. A synthetic
captured-topology fixture verifies input-role recovery, ABI round-trip, dead
branch pruning, and captured-width Metal emission.

Evidence is the structural fixture, typed package inventories, native outputs,
and retained timing samples; attributing Qwen's measured reduction to semantic
replacement is an inference from the only executable Qwen graph change.
