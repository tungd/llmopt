---
type: Experiment
title: 'Paired-row mixed-quant Linear tactics'
description: 'Reuse decoded GGUF block weights across captured two-row Linear operations for every supported quant layout.'
tags: [experiment, compiler, metal, tactics, linear, gguf, ud-quant, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T04:03:47+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-6-2026-08-30.json
    title: Slice 6 benchmark receipt
  - id: tactics
    resource: /lib/metal.ml
    title: Registered quantized Linear tactics and Metal kernels
  - id: runtime
    resource: /lib/metal_runtime.ml
    title: Semantic quantized Linear dispatch
  - id: target
    resource: /.okf/decisions/llama-cpp-target.md
    title: User-declared llama.cpp parity target
---

# Compiler change

The registered paired-row Linear tactic now covers every executable GGUF block
layout: `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q5_0`, `Q4_0`, and `IQ4_XS`. When the
captured row count is two, one SIMD group owns an output column, decodes each
weight once, and accumulates both rows. Other row counts retain the generic
implementation.

Selection reads the semantic Linear operation, captured shape and activation
dtypes, physical block layout, and discovered SIMD/threadgroup support. It does
not read a model name, tensor name, GGUF architecture field, or llama.cpp
architecture ID. Qwen adds paired execution at 103 non-Q4 quantized Linear
sites; Gemma adds it at 67 sites. The previously registered 187 Q4_K sites
remain paired.

# Full-model result

| Probe | Slice 5 LLMOpt | Slice 6 LLMOpt | Change | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `49.862504 ms` | `49.884439 ms` | `+0.021935 ms` (`+0.043991%`) | `7.9059375 ms` | `6.309744x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `46.065092 ms` | `42.235494 ms` | `-3.829598 ms` (`-8.313449%`) | `17.9571045 ms` | `2.352021x` |

The user-declared target is `0.9x` through `1.1x`. Command and dispatch counts
are unchanged because this slice replaces implementations below the semantic
schedule. The Qwen result is flat at the recorded precision; the Gemma median
falls by 8.313449%.

# Numerical scope

Both probes preserve the two row argmax IDs. Sharing decoded weights changes
rounded Float16 intermediates: Qwen differs from the preceding package by mean
absolute `0.0045750914` and maximum `0.02734375`; Gemma differs by mean absolute
`0.0092911124` and maximum `0.0703125`. This is not a byte-exact result.

# Validation

The tactic tests cover all seven quant layouts and retain generic selection for
other row counts and incompatible SIMD targets. The OCaml suite, all 49 Python
tests, native 453-dispatch Metal fixture, Xcode Metal compilation, and both
full-model package checks pass. Fresh packages contain zero opaque commands;
fresh llama.cpp runs use `llama-bench -p 2 -n 0 -r 10` and the receipt retains
all samples.

Evidence is the registered selector, generated package kernel inventories,
full-model execution, output comparison, and fresh timing receipt; attributing
Gemma's measured reduction to shared quant decode is an inference from the only
executable change in the slice.
