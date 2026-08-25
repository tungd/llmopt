---
type: Experiment
title: 'Fused RMSNorm-RoPE model measurement'
description: 'One bounded LFM2.5-350M trace and one needle matrix preserve exact tokens while observing improved short-trace ERS (0.412260) and lower TPOT.'
tags: [experiment, compiler, ocaml, metal, q8, rmsnorm, rope, benchmark, ers, needle, lfm25]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-25T18:18:00Z' }
sources:
  - id: implementation
    resource: /bench/results/lfm25-350m-q8-rms-rope-compiler-2026-08-25.txt
    title: Fused RMSNorm-RoPE compiler boundary
  - id: evidence
    resource: /bench/results/lfm25-350m-q8-rms-rope-measurement-2026-08-26.txt
    title: Bounded short and long-context measurement
  - id: prior
    resource: /bench/results/lfm25-350m-q8-paged-attention-measurement-2026-08-25.txt
    title: Previous direct paged-attention observation
---

# Bounded 350M execution

One 240-second supervised LFM2.5-350M attempt started at 74% system-wide free
memory with no resident model or native server. It ran four serial warmup and
four serial scored requests through the fused RMSNorm-RoPE schedule. All
requests completed, all four scored token sequences match the established
full-Q8 eager IDs (`[1098, 5706, 803, 4481]`, `[41677, 7, 2, 1]`, `[1098, 5410, 4100, 856]`,
`[31466, 7, 2, 1]`), and radix reuse remains 80/194 prompt tokens. Memory was 74%
free after warmup and 75% after process exit; sampled server RSS was 77,664 KiB
and port 18097 was released.

Scored ERS is `0.4122601696838274`, median TTFT is `73.70556250680238 ms`, and
median TPOT is `7.060125004500151 ms`. Against the previous paged-attention
observation, those values change by `+0.02899227286491235`, `+1.15575001109391451 ms`,
and `-0.97455549985170364 ms`. Mean TTFT improved by `-3.25130175042431802 ms`
and mean TPOT by `-0.83281950113208314 ms`.

# Long-context execution

A separate 900-second supervised attempt started at 77% free memory and ran
the six fixed-output 2,048/4,096-token needle requests. Retrieval is 6/6 and
all six 12-token sequences match the eager-Q8 reference exactly (`[8832, 563, 2880, 522, 31429, 526, 7, 2, 1, 553, 849, 18149]`).
Exact-only text remains 0/6 because the pinned output is `RAVEN-4271Lottery`.

At 2,048 tokens, median TTFT/TPOT/latency is
`1,241.0910410108045 / 25.416818182830784 / 1,520.6933750305325 ms`; deltas
against paged attention are `+1.57516601029783487 / +0.66904927371069789 /
+10.75679203495383263 ms`. At 4,096 tokens, medians are
`3,036.1555000417866 / 39.67043563765897 / 3,492.8893750184216 ms`, with
median TPOT decreasing by `-1.81341290854933845 ms`.

# Evidence boundary

The short and long reports each contain one non-interleaved observation. Exact
tokens and radix accounting are unchanged. Short-trace ERS improves from 0.383268
to 0.412260 and median TPOT drops below 7.1 ms, reflecting reduced dispatch overhead
from folding twelve 10-command chains into single SIMD kernels. No eager process ran
in either attempt. The model target was `LiquidAI/LFM2.5-350M`; 2.6B was not loaded.
