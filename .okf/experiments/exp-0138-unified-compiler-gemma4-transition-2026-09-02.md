---
type: Experiment
title: 'Unified compiler Gemma 4 12B transition run'
description: 'Records the one-shot measured-profile Gemma 4 12B AOT build, constructive-lowering and command-queue receipts, package checks, raw-token generation, and comparison with the pinned llama.cpp sequential result.'
tags: [experiment, gemma-4, qat, unified-compiler, constructive-lowering, cost-model, aot-command-queue, metal]
status: stable
generated: { by: codex/gpt-5.6, at: '2026-09-02T02:50:48+07:00' }
sources:
  - id: hardware-profile
    resource: /bench/results/uoc-gemma4-target-hardware-2026-09-02.json
    title: Measured Apple M4 Pro target profile
  - id: build-receipt
    resource: /bench/results/uoc-gemma4-build-2026-09-02.meta.txt
    title: One-shot engine build receipt
  - id: prefill-optimization
    resource: /bench/results/uoc-gemma4-prefill-optimization-2026-09-02.json
    title: Prefill optimization and AOT queue receipt
  - id: decode-optimization
    resource: /bench/results/uoc-gemma4-decode-optimization-2026-09-02.json
    title: Decode optimization and AOT queue receipt
  - id: generation-receipt
    resource: /bench/results/uoc-gemma4-benchmark-2026-09-02.meta.txt
    title: One-shot raw-token generation receipt
  - id: baseline
    resource: /bench/results/llama-cpp-gemma4-12b-mtp-2026-08-31.json
    title: Pinned llama.cpp sequential comparison receipt
  - id: decision
    resource: /decisions/constructive-primitive-loop-lowering-and-gpu-queue-compilation.md
    title: Constructive assignment lowering and AOT GPU command-queue decision
---

# Protocol

The run used source commit
`73c63b5c3f29796829c227ce7bc8925d6d90338c` on Apple M4 Pro. The target
profile was measured once immediately before the build and then passed through
the Gemma build wrapper to `llmopt-pipeline --target-profile`. It recorded
`228.772094 GB/s` sustained effective bandwidth at the 256 MiB stream size and
`5.375012 us` empty-dispatch overhead. The probe wrapper subsequently hit
zsh's read-only `status` variable while writing metadata; the child probe had
already completed, emitted its normal stdout, and written the complete JSON.
It was not rerun.

The target was the dereferenced 6,716,356,800-byte
`gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`, SHA-256
`90fd44e29e0d7cffeb0fd00dc73cfdab9ed0b0e95306ecf7821ea634c940c370`.
The output directory was fresh. `/usr/local/bin/python3.13` supplied PyTorch
2.13.0, Transformers 5.13.0, and GGUF support. No MTP assistant was used.

The prompt was:

> Write a comprehensive summary of general relativity, explaining spacetime
> curvature, gravitational time dilation, and the equivalence principle.

The target GGUF chat template was rendered with a user message and generation
prompt, then tokenized without adding a second BOS. This produced the same
37-token prompt count recorded by the llama.cpp baseline:

```text
2,105,9731,107,98,107,106,107,105,2364,107,6974,496,13611,12323,529,2870,74413,236764,26804,65272,32404,236764,39524,990,58692,236764,532,506,38789,11441,236761,106,107,105,4368,107
```

The LLMOpt benchmark accepted these IDs directly. It therefore did not exercise
the Model Program's packaged tokenizer or its reported generic `chatml`
metadata.

# Build and validation observation

The one authorized full build ran from `2026-09-01T19:32:05Z` to
`2026-09-01T19:32:41Z` and exited `0`. It captured 5,895 prefill FX nodes into
3,492 IR commands and 6,087 decode FX nodes into 3,732 IR commands. The build's
Transformers warning that `gemma4_unified` was instantiated as `gemma4` is
retained in stderr.

The prefill package check, decode package check, and Model Program check each
ran once and exited `0`. Prefill contains 109 kernels, zero opaque commands,
and a 3,064,064-byte planned workspace. Decode contains 108 kernels, zero
opaque commands, and a 2,234,112-byte planned workspace. The Model Program
reports ABI v6, 48 attention layers, no recurrent layers, and paged-Q8
attention in all 48 layers.

# Compiler and AOT queue receipt

Fusion and concurrency are separate products in these receipts. Accepted and
cut counts describe dependency-edge fusion analysis; stages, dependencies, and
barriers describe the AOT compiler command queue, not the serving request queue.

| Graph | Accepted | Cut | Unified lowerings | Legacy gated fallback | Costed Linear tactics | Kernels | AOT stages | Dependencies | Barriers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prefill | 48 | 2,664 | 0 | 1 | 329 | 109 | 1,113 | 2,906 | 1,014 |
| Decode | 48 | 2,760 | 0 | 1 | 329 | 108 | 1,161 | 3,146 | 1,062 |

All 329 prefill Linear selections chose the target-aware
`llmopt_q4_0_linear_f16_m2` tactic. All 329 decode selections chose
`llmopt_q4_0_linear_f16`. Their receipts retain the Q4_0 physical schema,
Metal SIMD32 primitives, target legality, resource use, alternatives, and full
cost breakdown.

Constructive dependency analysis therefore ran and target-aware quant tactics
were cost-selected, but no accepted region produced a unified lowering in this
Gemma build. The executable graph still invoked
`fuse_quant_gated_linear` once in both prefill and decode. This run does not
show that the covered Gemma path has replaced ad hoc matching.

The retained prefill and decode optimization receipts have SHA-256 values
`7b2eb4536a9144fb91c47c1a7269f9da2ad96d41b00653a0c052ddeef5237538`
and
`1ce153935a5dfd182f2131bb4df891fe59870bdab0adc9cc75f5bbae7c93a5d3`.

# Generation observation

The sole generation command requested 128 greedy tokens. It ran from
`2026-09-01T19:34:11Z` to `2026-09-01T19:34:40Z` and exited `0`.

| Measure | LLMOpt | Pinned llama.cpp sequential | Delta | Ratio |
|---|---:|---:|---:|---:|
| Throughput | `4.474057 tok/s` | `27.94 tok/s` | `-23.465943 tok/s` (`-83.986911%`) | `0.160130888x` |
| Per-token time | `166.168352 ms` mean inter-token | `35.79 ms/token` | `+130.378352 ms` (`+364.287097%`) | `4.642870970x` |

The throughput row preserves the requested arithmetic comparison but its
denominators differ: LLMOpt divides 128 tokens by TTFT plus all inter-token
intervals, while the llama.cpp field divides generated tokens by decode
evaluation time and reports prompt time separately. It is not a like-for-like
decode-throughput ratio. The inter-token/TPOT row is the closer decode-time
comparison.

LLMOpt additionally measured `7.506000 s` TTFT, `28.609381 s` total elapsed
time, and `133.504152 ms` median inter-token time.

The completion contains a repeated `176695,31436,180542,236761` pattern and
then token ID `0` from one-based completion position 29 through position 128.
The process exit and requested token count show that native execution
completed; they do not establish output correctness. No matched LLMOpt
reference token stream exists for this engine, and no token-parity claim is
made.

# Result boundary

This one-shot transition produced usable build, package, Model Program,
cost-selection, dependency-analysis, and AOT-queue receipts. Its measured
compiler coverage, output tokens, and timings are the result: zero unified
lowerings for the target graphs, one retained legacy gated matcher per graph,
a degenerate completion, and the deltas above. No new performance or success
threshold is attached to those observations.

## Follow-up clarification

The original schema-1 `legacy-count: 1` recorded matcher invocation, not a
proved graph rewrite. Receipt schema 2 now reports invocations and rewrites
separately. The transition artifact and its counts remain unchanged; the
constructive-coverage and decode-collapse repairs are recorded in
[exp-0139](exp-0139-unified-compiler-gemma4-repair-2026-09-02.md).
