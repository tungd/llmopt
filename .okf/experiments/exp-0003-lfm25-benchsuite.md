---
type: Experiment
title: 'First LFM2.5 MPS ERS benchsuite'
description: 'Historical short smoke observation; it established generation routing and the ERS harness but is not an optimization comparison.'
tags: [experiment, benchmark, ERS, needle, lfm2.5, mps]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-20T12:20:00Z' }
sources:
  - id: suite
    resource: /bench/lfm25_benchsuite.py
    title: benchsuite implementation
  - id: result
    resource: /bench/results/lfm25-benchsuite-2026-08-20.json
    title: compact recorded result
  - id: trace
    resource: /bench/traces/lfm25-mps-smoke.json
    title: scored local trace
---

# Procedure

```sh
ninja -f ninja.build bench-suite
```

The run used the 2-conversation x 2-turn local trace, pinned four generated
tokens per request, a distinct shape-matched warmup trace, and a 128/256-token
needle matrix at 10/50/90 percent placement.

# Observation

The corrected eager PyTorch MPS baseline completed all four scored requests
with ERS `0.4836256290`. The corrected llmopt direct-FX path completed all four
scored requests with ERS `0.5`; its scored outputs and completion-token counts
matched eager. The fixed-input direct-forward comparison was
`[1, 15, 128000]` with max and mean absolute error `0.0`.

The ERS value is saturated for llmopt under the adopted reference constants:
its TTFT median was `7.9492 ms` versus eager's `18.1430 ms`, and its TPOT
median was `50.9673 ms` versus eager's `56.2095 ms`. Because TPOT remains above
the 10 ms ceiling, the equal-weight score contributes zero TPOT points; the
sub-10 ms TTFT contributes the full TTFT half, producing `0.5`.

This remains an ordered single-process smoke observation, not an optimization
comparison: eager ran first, and llmopt ran second after shared MPS state was
warm. The ERS value is saturated under the adopted constants. The eager value
is retained as historical baseline evidence; the long-context isolated profile
is the active comparison workload.

The needle probe returned `0/6` for eager and `0/6` for llmopt. Both responses
continued an explanation rather than emitting `RAVEN-4271` within the pinned
12-token response budget. Needle correctness was recorded separately and was
not required for the engine exit status; the benchsuite exited with code `0`.

The earlier invalid llmopt observation is retained as provenance in the result
record. The corrected run emitted eight OCaml graph artifacts during the
generation path before writing the llmopt report.

# Limits

This is an in-process serial MPS adapter, not the remote OpenAI-compatible
server runner. The recorded trace is a local smoke workload. The active
comparison profile is defined in `bench/racebench/profiles.py` and the full
natural needle defaults remain available through the CLI.
