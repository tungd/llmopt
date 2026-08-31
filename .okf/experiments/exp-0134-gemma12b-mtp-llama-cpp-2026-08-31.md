---
type: Experiment
title: 'Gemma 4 12B QAT llama.cpp MTP baseline'
description: 'Records three serial sustained-generation campaigns for sequential and MTP decoding with the same Gemma 4 12B QAT target, prompt, and greedy sampling configuration.'
tags: [experiment, gemma-12b, qat, mtp, speculative-decoding, llama-cpp, apple-silicon]
status: stable
generated: { by: 'process:codex', at: '2026-08-31T12:24:43+07:00' }
sources:
  - id: receipt
    resource: /bench/results/llama-cpp-gemma4-12b-mtp-2026-08-31.json
    title: Structured three-campaign benchmark receipt
  - id: runner
    resource: /bench/llama_cpp_speculative_bench.py
    title: Bounded llama.cpp sequential and MTP benchmark runner
  - id: regression
    resource: /python/tests/test_llama_cpp_speculative_bench.py
    title: Runner lifecycle and output-parser regressions
  - id: decision
    resource: /decisions/gemma-12b-qat-mtp-speculative-benchmark.md
    title: Gemma 4 12B QAT and MTP measurement protocol
---

# Protocol

The Apple M4 Pro host had 24 GB unified memory and reported 72% system-wide
free memory immediately before the final run. No other `llama-cli` process was
active. llama.cpp was version `0.1.2-dev`, build 10531, commit `f20395d`.

The target was `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`; the drafter was
`mtp-gemma-4-12B-it.gguf`. Both modes used the same prompt, temperature `0`,
128 requested generation tokens, eight CPU threads, Metal offload, no warmup,
single-turn execution, and three serial campaigns. MTP used
`--spec-type draft-mtp --spec-draft-n-max 4`.

# Result

| Mode | Campaign TPOT (ms) | Campaign throughput (tok/s) | Median TPOT | Median throughput | Reported acceptance |
|---|---|---|---:|---:|---:|
| Sequential | `32.30`, `32.38`, `32.36` | `30.96`, `30.89`, `30.90` | `32.36 ms` | `30.90 tok/s` | n/a |
| MTP, K=4 | `53.82`, `54.39`, `53.87` | `18.58`, `18.39`, `18.56` | `53.87 ms` | `18.56 tok/s` | `67.2%` in each campaign |

The runner reported a `0.60x` MTP-to-sequential throughput ratio. These are the
observed values; no pass/fail threshold is attached to them.

# Harness correction and boundary

The original wrapper allowed chat-template `llama-cli` to retain a TTY and
wait for another turn. It also expected only legacy timing lines. The corrected
runner uses single-turn/simple I/O, owns a new process group, bounds each child,
terminates the group on timeout or shutdown, rejects overlapping benchmark or
`llama-cli` processes, accepts the current compact throughput footer, enables
verbose speculative statistics, and rejects nonzero or incomplete results.

Seven focused regressions and the repository's 56-test Python Ninja target
pass. This receipt is an external llama.cpp baseline only; it does not show
that LLMOpt can execute Gemma 4 12B or its MTP drafter.
