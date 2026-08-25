---
type: Experiment
title: 'Full-Q8 LM-head capture and native measurement'
description: 'Recapture LFM2.5-350M with a Q8 vocabulary projection, then measure exact tokens, radix reuse, ERS, and long-context needle retrieval.'
tags: [experiment, compiler, pytorch, ocaml, metal, q8, lm-head, benchmark, ers, needle, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T09:36:28Z' }
sources:
  - id: capture
    resource: /bench/results/lfm25-350m-q8-lm-head-capture-2026-08-25.txt
    title: Memory-bounded full-Q8 graph capture
  - id: measurement
    resource: /bench/results/lfm25-350m-q8-lm-head-measurement-2026-08-25.txt
    title: Native, eager, and needle observations
  - id: prior
    resource: /bench/results/lfm25-350m-q8-last-token-projection-measurement-2026-08-25.txt
    title: Previous final-row FP16-head native observation
---

# Capture

One supervised `LiquidAI/LFM2.5-350M` capture converted 93 linear modules,
including `lm_head`, and skipped none. The prefill/decode graphs contain
1,157/1,197 FX nodes and optimize to 810/864 commands, 60/58 kernel entries,
zero opaque commands, and 243 shared tensor bindings. The single binary tensor
archive is 489,377,152 bytes. Its 67,239,936-byte growth is exactly one
65,536-by-1,024 int8 matrix plus 65,536 FP16 scales; the original FP16 matrix
remains the token embedding.

The real plans end in `q8-linear[6x65536x1024]` and
`q8-linear[1x65536x1024]`. Runtime specialization exposes one logits row at
prefill lengths 13/128/4,096. Q8-group-64 and selectable FP16 package-pair
validation pass. Capture-time eager and `torch.compile` prefill/decode logits
are exact, and both choose token `19130`.

# Native and eager observations

The native 4+4 serial trace completes with exact token parity against a
separate full-Q8 eager run and preserves 80/194 radix reuse. Native ERS is
`0.3908962321067631`, median TTFT is `71.4766460005194 ms`, and median TPOT is
`7.31056933485282 ms`. Against the previous final-row FP16-head native report,
the observed changes are `+0.032049180526578736`, `-7.680833485210314 ms`, and
`-0.989271007711066 ms`.

The separate full-Q8 eager report has ERS `0.3663754874502978`, median TTFT
`65.51054149167612 ms`, and median TPOT `49.24876399066609 ms`. Native minus
eager is `+0.02452074465646531` ERS, `+5.966104508843273 ms` TTFT, and
`-41.93819465581327 ms` TPOT in these non-interleaved observations.

# Long-context correctness

The Q8-head native package completes all six fixed-output 2,048/4,096-token
needle prompts. Retrieval and complete 12-token parity are 6/6; exact-only text
is 0/6 because the pinned continuation remains `RAVEN-4271Lottery`. Median
TTFT/TPOT is `1164.398/33.007 ms` at 2,048 tokens and `2706.019/60.866 ms` at
4,096 tokens. The preceding long matrix predates both final-row projection and
Q8 head capture, so its timing deltas do not isolate this one change.

# Safety and attribution

Capture, native ERS, eager, and needle workloads each used one preflighted,
supervised attempt. Every process exited, ports were released, and memory
recovered after each run. These are separate single observations rather than
interleaved repetitions. The current long matrix compares with the established
eager-Q8 token sequence; it did not load another eager long-context process.
