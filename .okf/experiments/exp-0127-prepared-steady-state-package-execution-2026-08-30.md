---
type: Experiment
title: 'Prepared steady-state package execution'
description: 'Reuse generic schedule memory preparation and measure synchronized execution without validation readback.'
tags: [experiment, runtime, benchmark, metal, qwen, gemma, llama.cpp]
status: stable
generated: { by: 'process:codex', at: '2026-08-30T21:51:34+07:00' }
sources:
  - id: receipt
    resource: /bench/results/compiler-generalization-slice-25-2026-08-30.json
    title: Slice 25 benchmark receipt
  - id: runner
    resource: /bin/package_run.ml
    title: Native package benchmark runner
---

# Structural change

The package runner now builds the typed liveness memory plan and allocates its
Metal workspace once, before warmup, then reuses both for every synchronized
schedule execution. The timed interval ends after command-buffer completion;
the final output readback and artifact write happen once after all repetitions.

This uses the existing generic `execute_schedule` API and captured schedule. It
does not inspect a model name, tensor name, architecture ID, command family, or
quantization layout, and it changes no package, command, dispatch, or kernel.

The boundary matches llama.cpp's benchmark structure: its timer begins after
context reset, `test_prompt` calls `llama_decode` and `llama_synchronize`, and
result printing occurs after the timer. The retained llama.cpp source is commit
`f20395d`.

# Full-model result

| Probe | Previous harness | Prepared execution | Fresh llama.cpp | Ratio |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B UD-Q4_K_XL | `10.149598 ms` | `8.777499 ms` | `7.591500 ms` | `1.156227x` |
| Gemma-4-E2B-it UD-Q4_K_XL | `20.252466 ms` | `18.818855 ms` | `17.220854 ms` | `1.092795x` |

The generic preparation and timing correction removes `1.372099 ms` from the
Qwen measurement and `1.433611 ms` from Gemma. Gemma is inside the declared
`[0.9x, 1.1x]` range. Qwen is `0.426849 ms` above its `1.1x` latency.

Both final output artifacts are byte exact with slice 24. Their hashes remain
`3db055cc...` and `896aa2a6...`, and all four row argmax IDs are unchanged.

# Validation

The OCaml suite, 49-test Ninja Python suite, 52-pass/1-skip Pytest suite,
native 101-kernel primitive fixture, output byte comparisons, and fresh
`llama-bench -p 2 -n 0 -r 10` runs pass.

Evidence is the unchanged full-model packages, source-level timing boundary,
exact output hashes, tests, and retained timing receipt. This slice changes the
steady-state benchmark/runtime preparation boundary, not compiler or kernel
performance.
