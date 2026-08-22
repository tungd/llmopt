---
type: Experiment
title: 'Q8 weight-only linear lowering'
description: 'Make per-output-channel int8 weight-only linear execution an explicit default compiler target for LFM2.5 while retaining FP16 as the runtime fallback.'
tags: [experiment, optimizer, quantization, q8, metal, llvm, ocaml]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-20T16:30:00Z' }
sources:
  - id: ir
    resource: /lib/ir.ml
    title: typed Q8 operation and quantization domain
  - id: tile
    resource: /lib/tile.ml
    title: typed Q8 linear construction surface
  - id: pass-pipeline
    resource: /lib/passes.ml
    title: pure optimizer pipeline entry point
  - id: metal
    resource: /lib/metal.ml
    title: Q8 Metal emitter
  - id: llvm
    resource: /lib/llvm_ir.ml
    title: Q8 LLVM inspection emitter
  - id: python-q8
    resource: /python/llmopt_backend/quantization.py
    title: Python Q8 model rewrite and callable boundary
  - id: q8-tests
    resource: /python/tests/test_quantization.py
    title: Python Q8 runtime and FX boundary tests
  - id: q8-benchmark
    resource: /bench/results/lfm25-350m-q8-racebench-baseline.json
    title: bounded 350M Q8 MPS comparison
  - id: ninja
    resource: /ninja.build
    title: Q8 artifact validation targets
---

# Decision

The model-shaped LFM2.5 compiler target defaults to `Q8_weight_only`: signed
int8 weights with one float16 scale per output channel, float16 activations,
and float16 output. FP16 remains an explicit configuration value for fallback
and parity work.

# Implementation

The first optimizer slice adds a dedicated `q8-linear` IR operation rather
than reinterpreting an FP16 weight. Its inputs are activation, `[N,K]` int8
weight, `[1,N]` scale, and an optional `[1,N]` bias. The CPU handler computes
the dequantized reference result, while the Metal emitter uses `char` weight
storage and the LLVM emitter uses signed `i8` loads followed by scale multiply.

`Passes.optimize` is now the common pure pass entry point used by the demo and
FX compiler executable. The existing linear/bias fusion remains in that
pipeline; the Q8 operation is already fused at the linear boundary.

# Verification

```sh
ninja -f ninja.build test
ninja -f ninja.build q8-smoke
```

The tests cover Q8 CPU numerical results, captured IR, default LFM2.5 Q8
configuration, and both source emitters. `q8-smoke` compiles the generated
Q8 Metal source with `xcrun metal` and Q8 LLVM text with `clang`. The Python
suite covers per-channel quantization, model rewriting, and preservation of the
Q8 operator boundary in an FX manifest. The historical 350M FP16 baseline
remains recorded; the new Q8 result is recorded at
`bench/results/lfm25-350m-q8-racebench-baseline.json`.

# Boundary

The model-level runners default to the Python Q8 loader and set
`LLMOPT_QUANTIZATION=q8`; `--quantization fp16` is the explicit fallback. At
the time of this experiment the callable dequantized in the PyTorch CPU/MPS
implementation, so the generated Metal Q8 kernel was not yet the runtime
execution path. CPU and FX boundary tests pass, and a correctly shaped MPS
callable probe returned exact reference output. The bounded 350M Q8 run completed with `engine_pass: true`,
15/15 successful warmup and scored requests per candidate, exact fixed-forward
digest and token-ID parity, and `0/6` needle retrieval for each candidate. Its
single-run relative latency comparison remains invalid under the bench
contract; the result is evidence for the Q8 execution boundary, not a claim
that the current dequantizing callable is faster than FP16.
