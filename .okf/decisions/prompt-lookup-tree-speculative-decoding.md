---
type: Decision
title: 'Weightless Prompt-Lookup (N-Gram) Tree Speculative Decoding'
description: 'Records the linked weightless n-gram proposal and target-verification path with exact token parity and comparative measurements.'
tags: [decision, prompt-lookup, n-gram, speculative-decoding, tree-attention, zero-weights, metal]
status: stable
generated: { by: 'codex/gpt-5', at: '2026-09-01T17:25:00+07:00' }
sources:
  - id: serving-engine
    resource: /lib/serving_engine.ml
    title: Serving engine prompt-lookup verification
  - id: tree-attention
    resource: /lib/metal.ml
    title: Metal speculative tree attention kernel
  - id: slice31-sustained
    resource: /bench/results/prompt-lookup-sustained-1024-slice31-2026-09-01.json
    title: Slice31 mixed-GGUF sustained serving receipt
  - id: slice31-gqa3-probe
    resource: /bench/results/smollm-slice31-gqa3-paged-attention-probe-2026-09-01.json
    title: Slice31 GQA3 paged-attention probe
  - id: split8-prebaked-probe
    resource: /bench/results/smollm-slice31-split8-prebaked-probe-2026-09-01.json
    title: Slice31 split8 plus prebaked decode probe
  - id: prebaked-isolation-probe
    resource: /bench/results/smollm-slice31-prebaked-isolation-probe-2026-09-01.json
    title: Slice31 prebaked decode isolation probe
  - id: prebaked-maskfix-probe
    resource: /bench/results/smollm-slice31-prebaked-maskfix-probe-2026-09-01.json
    title: Slice31 corrected mask-independent prebaked decode probe
  - id: m1n4-linear-probe
    resource: /bench/results/smollm-slice31-m1n4-linear-probe-2026-09-01.json
    title: Slice31 four-column single-token linear probe
  - id: q5-gated-linear-probe
    resource: /bench/results/smollm-slice31-q5-gated-linear-probe-2026-09-01.json
    title: Slice31 Q5_0 gated-linear probe
  - id: q5-gated-exact-probe
    resource: /bench/results/smollm-slice31-q5-gated-exact-probe-2026-09-01.json
    title: Slice31 instruction-order-preserving Q5_0 gated-linear probe
  - id: split8-contiguous-probe
    resource: /bench/results/smollm-slice31-split8-contiguous-probe-2026-09-01.json
    title: Slice31 contiguous split8 attention probe
  - id: q8-r2-sg4-probe
    resource: /bench/results/smollm-slice31-q8-r2-sg4-probe-2026-09-01.json
    title: Slice31 four-SIMDgroup Q8_0 vocabulary projection probe
  - id: q5-r4-sg2-probe
    resource: /bench/results/smollm-slice31-q5-r4-sg2-probe-2026-09-01.json
    title: Slice31 four-row Q5_0 projection probe
  - id: prebaked-profile-coverage
    resource: /bench/results/smollm-slice31-prebaked-profile-coverage-2026-09-01.json
    title: Slice31 prebaked profiling coverage receipt
  - id: icb-zero-barrier-probe
    resource: /bench/results/smollm-slice31-icb-zero-barrier-probe-2026-09-01.json
    title: Slice31 zero-barrier ICB observation
  - id: icb-stage-barrier-probe
    resource: /bench/results/smollm-slice31-icb-stage-barrier-probe-2026-09-01.json
    title: Slice31 stage-preserving ICB observation
  - id: direct-argmax-probe
    resource: /bench/results/smollm-slice31-direct-argmax-probe-2026-09-01.json
    title: Slice31 direct shared-buffer argmax observation
  - id: hostpath-regression-probe
    resource: /bench/results/smollm-slice31-hostpath-regression-probe-2026-09-01.json
    title: Slice31 host-path regression observation
  - id: lazy-schedule-regression-probe
    resource: /bench/results/smollm-slice31-lazy-schedule-regression-probe-2026-09-01.json
    title: Slice31 lazy schedule regression observation
  - id: same-binary-lazy-schedule-probe
    resource: /bench/results/smollm-slice31-same-binary-lazy-schedule-probe-2026-09-01.json
    title: Slice31 same-binary lazy schedule observation
  - id: prebaked-kernel-inventory
    resource: /bench/results/smollm-slice31-prebaked-kernel-inventory-2026-09-01.json
    title: Slice31 prebaked kernel inventory
  - id: mixed-qkv-fusion-static
    resource: /bench/results/smollm-slice31-mixed-qkv-fusion-static-2026-09-01.json
    title: Slice31 mixed-quantized QKV fusion static evidence
  - id: mixed-qkv-fusion-probe
    resource: /bench/results/smollm-slice31-mixed-qkv-fusion-probe-2026-09-01.json
    title: Slice31 mixed-quantized QKV fusion device evidence
  - id: paired-cache-pack-static
    resource: /bench/results/smollm-slice31-paired-cache-pack-static-2026-09-01.json
    title: Slice31 paired K/V cache-pack static evidence
  - id: paired-cache-pack-probe
    resource: /bench/results/smollm-slice31-paired-cache-pack-probe-2026-09-01.json
    title: Slice31 paired K/V cache-pack device evidence
  - id: llama-style-q5-mv-static
    resource: /bench/results/smollm-slice31-llama-style-q5-mv-static-2026-09-01.json
    title: Slice31 llama.cpp-style Q5_0 matrix-vector static evidence
  - id: llama-style-q5-mv-probe
    resource: /bench/results/smollm-slice31-llama-style-q5-mv-probe-2026-09-01.json
    title: Slice31 llama.cpp-style Q5_0 matrix-vector device evidence
  - id: llama-style-kquant-mv-static
    resource: /bench/results/smollm-slice31-llama-style-kquant-mv-static-2026-09-01.json
    title: Slice31 llama.cpp-style Q4_K and Q6_K matrix-vector static evidence
  - id: llama-style-kquant-mv-probe
    resource: /bench/results/smollm-slice31-llama-style-kquant-mv-probe-2026-09-01.json
    title: Slice31 llama.cpp-style Q4_K and Q6_K matrix-vector device evidence
  - id: llama-style-q8-mv-static
    resource: /bench/results/smollm-slice31-llama-style-q8-mv-static-2026-09-01.json
    title: Slice31 llama.cpp-style Q8_0 vocabulary matrix-vector static evidence
  - id: llama-style-q8-mv-probe
    resource: /bench/results/smollm-slice31-llama-style-q8-mv-probe-2026-09-01.json
    title: Slice31 llama.cpp-style Q8_0 vocabulary matrix-vector device evidence
  - id: radix-offset-walk-static
    resource: /bench/results/smollm-slice31-radix-offset-walk-static-2026-09-01.json
    title: Slice31 allocation-reduced radix walk static evidence
  - id: radix-offset-walk-probe
    resource: /bench/results/smollm-slice31-radix-offset-walk-probe-2026-09-01.json
    title: Slice31 allocation-reduced radix walk device evidence
  - id: radix-offset-insert-static
    resource: /bench/results/smollm-slice31-radix-offset-insert-static-2026-09-01.json
    title: Slice31 allocation-reduced radix insertion static evidence
  - id: radix-offset-insert-probe
    resource: /bench/results/smollm-slice31-radix-offset-insert-probe-2026-09-01.json
    title: Slice31 allocation-reduced radix insertion device evidence
  - id: radix-direct-values-static
    resource: /bench/results/smollm-slice31-radix-direct-values-static-2026-09-01.json
    title: Slice31 direct radix path-value assembly static evidence
  - id: radix-direct-values-probe
    resource: /bench/results/smollm-slice31-radix-direct-values-probe-2026-09-01.json
    title: Slice31 direct radix path-value assembly device evidence
  - id: radix-borrowed-values-static
    resource: /bench/results/smollm-slice31-radix-borrowed-values-static-2026-09-01.json
    title: Slice31 borrowed radix match values static evidence
  - id: radix-borrowed-values-probe
    resource: /bench/results/smollm-slice31-radix-borrowed-values-probe-2026-09-01.json
    title: Slice31 borrowed radix match values device evidence
  - id: llama-style-qkv-mv-static
    resource: /bench/results/smollm-slice31-llama-style-qkv-mv-static-2026-09-01.json
    title: Slice31 llama.cpp-style fused QKV matrix-vector static evidence
  - id: llama-style-qkv-mv-probe
    resource: /bench/results/smollm-slice31-llama-style-qkv-mv-probe-2026-09-01.json
    title: Slice31 invalid stale-executable QKV attempt
  - id: llama-style-qkv-mv-valid-probe
    resource: /bench/results/smollm-slice31-llama-style-qkv-mv-valid-probe-2026-09-01.json
    title: Slice31 valid all-Q5 llama.cpp-style QKV measurement
  - id: llama-style-mixed-qkv-mv-static
    resource: /bench/results/smollm-slice31-llama-style-mixed-qkv-mv-static-2026-09-01.json
    title: Slice31 mixed Q5/Q5/Q8 llama.cpp-style QKV static evidence
  - id: llama-style-mixed-qkv-mv-probe
    resource: /bench/results/smollm-slice31-llama-style-mixed-qkv-mv-probe-2026-09-01.json
    title: Slice31 mixed Q5/Q5/Q8 llama.cpp-style QKV device evidence
  - id: rms-wide-sg4-static
    resource: /bench/results/smollm-slice31-rms-wide-sg4-static-2026-09-01.json
    title: Slice31 four-SIMDgroup wide RMSNorm static evidence
  - id: rms-wide-sg4-probe
    resource: /bench/results/smollm-slice31-rms-wide-sg4-probe-2026-09-01.json
    title: Slice31 rejected four-SIMDgroup wide RMSNorm device evidence
  - id: sequence-cursor-static
    resource: /bench/results/smollm-slice31-sequence-cursor-static-2026-09-01.json
    title: Slice31 lease-backed sequential cursor static evidence
  - id: sequence-cursor-probe
    resource: /bench/results/smollm-slice31-sequence-cursor-probe-2026-09-01.json
    title: Slice31 lease-backed sequential cursor device evidence
  - id: sequence-slot-append-static
    resource: /bench/results/smollm-slice31-sequence-slot-append-static-2026-09-01.json
    title: Slice31 incremental sequence slot-index upload static evidence
  - id: sequence-slot-append-probe
    resource: /bench/results/smollm-slice31-sequence-slot-append-probe-2026-09-01.json
    title: Slice31 rejected incremental sequence slot-index upload device evidence
  - id: maskless-decode-static
    resource: /bench/results/smollm-slice31-maskless-decode-static-2026-09-01.json
    title: Slice31 maskless one-token decode static evidence
  - id: maskless-decode-probe
    resource: /bench/results/smollm-slice31-maskless-decode-probe-2026-09-01.json
    title: Slice31 maskless one-token decode device evidence
  - id: gqa3-split8-static
    resource: /bench/results/smollm-slice31-gqa3-split8-static-2026-09-01.json
    title: Slice31 combined GQA3 split8 attention static evidence
  - id: gqa3-split8-probe
    resource: /bench/results/smollm-slice31-gqa3-split8-probe-2026-09-01.json
    title: Slice31 combined GQA3 split8 changed-output device evidence
  - id: gqa3-split8-strictfp-static
    resource: /bench/results/smollm-slice31-gqa3-split8-strictfp-static-2026-09-01.json
    title: Slice31 strict-FP GQA3 split8 parity-repair static evidence
  - id: gqa3-split8-strictfp-probe
    resource: /bench/results/smollm-slice31-gqa3-split8-strictfp-probe-2026-09-01.json
    title: Slice31 rejected strict-FP GQA3 split8 device evidence
  - id: gqa3-split8-vshared-probe
    resource: /bench/results/smollm-slice31-gqa3-split8-vshared-probe-2026-09-01.json
    title: Slice31 rejected serial-score shared-V GQA3 split8 device evidence
  - id: smollm-matched-sustained
    resource: /bench/results/smollm-slice31-llamacpp-matched-trace-probe-2026-09-01.json
    title: Slice31 matched-token sustained LLMOpt and llama.cpp comparison
  - id: smollm-matched-q8kv
    resource: /bench/results/smollm-slice31-llamacpp-q8kv-matched-trace-probe-2026-09-01.json
    title: Slice31 matched-token Q8-KV LLMOpt and llama.cpp comparison
---

# 1. Context

Prompt lookup proposes tokens by searching the already generated token history,
so it adds no draft-model weights. Proposal cost and total throughput are measured
rather than inferred from a synthetic CPU matcher.

# 2. Mathematical Formulation

Given the token history $T = [t_0, t_1, \dots, t_{N-1}]$:

1. **Suffix Matching**: For a search window $n \in [\text{max\_ngram}, \text{min\_ngram}]$ (typically $n \in [3, 1]$), extract the current query suffix:
   $$W = [t_{N-n}, \dots, t_{N-1}]$$
2. **History Scan**: Find the latest matching index $j < N - n$ in $T$:
   $$T[j - n + 1 \dots j] = W$$
3. **Draft Proposal**: Propose the subsequent $K$ tokens following the match:
   $$\text{Draft} = [t_{j+1}, \dots, t_{j+K}]$$
4. **Verification**:
   - Execute the linked ABI-v5 verification entrypoint over the previous target
     cache plus the current token and proposed continuation.
   - Apply the greedy `K+1` acceptance rule and commit accepted cache slots.

# 3. Implemented boundary

- `Model_program.Prompt_lookup` declares the verification package and captured
  query/past dimensions.
- `bin/pipeline.ml` links that package into ABI v5 while retaining ABI-v4 MTP
  compatibility.
- `Generation.generate_prompt_lookup` performs real n-gram proposal, target
  verification, acceptance, cache commit/rollback, and token emission.
- `bench/bench_prompt_lookup_speculative.py` invokes the real LLMOpt and
  llama.cpp executables; it does not report synthetic matcher latency.

# 4. Measurement on 2026-09-01

The first end-to-end probe exposed a layout defect: multi-token paged attention
wrote head-major `[head, token, dim]` bytes into a token-major
`[token, head, dim]` IR output. The corrected Metal index is
`((batch * q_len + q_pos) * query_heads + head) * head_dim`; a source regression
covers the multi-token layout.

After that correction, three 128-token LLMOpt SmolLM2 campaigns record sequential
median `91.601159 tok/s` and prompt-lookup median `305.218219 tok/s`
(`3.332034466943808x`). Every prompt-lookup campaign performs 33 target forward
passes and 25 verifier passes, proposes 99 tokens, accepts 96, and produces the
same completion-token hash as all sequential campaigns.

The same fresh receipt records llama.cpp `ngram-simple` on Gemma 4 12B UD-Q4 at
`18.4 tok/s` sequential and `13.77 tok/s` n-gram median
(`0.7483695652173913x`), with identical generated-text hashes and exact `13/53`
accepted proposals in every campaign. All six llama.cpp timing records are
consistent with measured wall time.

## 4.1 Slice31 mixed-GGUF sustained observation

The stateful prefill, one-token decode, and five-token verification captures were
rebound to the exact enriched 273-tensor SmolLM2 Slice31 GGUF. The linked
packages have zero opaque commands; decode has 1,399 commands and retains 60
quantized `linear-add` residual fusions.

Three 1,024-token campaigns measured LLMOpt sequential at `29.637509 tok/s`
median and prompt lookup at `71.161972 tok/s` median. llama.cpp measured
`381.39 tok/s` sequential and `376.19 tok/s` `ngram-simple` on the publisher
GGUF. The LLMOpt prompt-lookup completion was not equal to sequential: the first
campaign diverged at completion index 2 (`1517` sequential versus `4165`
verification), and prompt-lookup produced two hashes across three campaigns.
This receipt therefore records the measured throughput and the observed
multi-token verification mismatch separately; the earlier 128-token uniform
W4A16 parity result is not evidence for the mixed-GGUF path.

## 4.2 Paged-attention GQA3 probe

A geometry-selected Metal kernel shared each Q8 K/V cache scan across the three
query heads belonging to one KV head, reducing paged-attention rows per layer
from nine to three. The native Metal fixture was exact and the 128/512-token
completion streams matched the baseline.

The 128-token observation changed from `78.785680` to `80.458064 tok/s`; the
512-token observation changed from `46.159371` to `45.528602 tok/s`, while
median inter-token latency changed from `21.409035` to `21.912098 ms`. This
isolated result does not explain the sustained gap to the recorded llama.cpp
`381.39 tok/s` sequential result.

## 4.3 Correctly packaged split8 and prebaked decode

The first engine named for split8 was linked by a stale pipeline executable and
did not declare that kernel; its measurement was retracted. A freshly linked
engine declares the kernel and executes long-context decode from one indirect
command buffer whose paged-attention past length is updated per token.

One 128/512-token observation measured `127.801943` and `129.855524 tok/s`,
respectively, with 512-token median inter-token latency of `7.409811 ms`. The
completion stream differs from the sequential baseline, so split8 is disabled
by default and retained behind `LLMOPT_EXPERIMENTAL_SPLIT8=1`; the next
measurement isolates prebaking with the prior GQA3 arithmetic.

## 4.4 Prebaked decode isolation

With split8 disabled, prebaked GQA3 decode measured `107.023210 tok/s` at 128
tokens and `72.564393 tok/s` at 512 tokens. The completion stream still changed,
showing that the fixed indirect plan omitted a dynamic dependency rather than
the divergence being specific to split8 arithmetic.

The paged-attention past-length parameter is updated for every indirect launch,
but the captured one-token causal mask extent was fixed at plan creation. Since
one-token autoregressive decode attends to every prior token and the current
token, that mask is redundant. Prebaking is now explicit-only through
`LLMOPT_EXPERIMENTAL_PREBAKED=1`, and one-token paged attention ignores the mask
before the next isolated observation.

## 4.5 Mask-independent prebaked decode

After one-token paged attention stopped reading the redundant captured causal
mask, prebaked GQA3 decode reproduced the original completion stream exactly at
both 128 and 512 generated tokens. The observations measured `92.231757 tok/s`
and `54.288495 tok/s`, respectively: `1.170666509x` and `1.176109939x` the
original mixed-GGUF Slice31 observations.

The corrected result isolates command-plan reuse from changed attention
arithmetic. Prebaked decode is therefore the default for attention-only model
programs from 64 past tokens; recurrent models retain dynamic decode. At 512
tokens, the recorded llama.cpp sequential reference remains `7.025245404x` the
corrected LLMOpt observation, so the next implementation target is the
single-token mixed-quant linear path.

## 4.6 Four-column single-token linear probe

A full-SIMD Q5_0/Q8_0 tactic kept each output column's original accumulation
and reduction order while sharing one activation load across four adjacent
columns. The linked Metal package compiled and the first 128 completion tokens
matched the original sequential receipt exactly.

The single 128/512 campaign measured `87.446463` and `52.359673 tok/s`, which
were `5.1883%` and `3.5529%` below the corrected prebaked observations. The
tactic was removed from selection after the measurement; the corrected
prebaked GQA3 result remains the active implementation.

## 4.7 Q5_0 gated-linear dispatch fusion

The mixed-GGUF decode graph contained 30 FFN sequences with separate gate and
up Q5_0 linears, SiLU, and multiply dispatches. A typed graph rewrite reduced
each sequence to one gated-linear operation, changing decode from 1,399 to
1,249 IR commands and eliminating 90 GPU dispatches per token.

The first coupled-accumulator implementation measured `97.867794 tok/s` at 128
tokens and `56.037000 tok/s` at 512 tokens, `6.1107%` and `3.2208%` above the
corrected prebaked observations. Both completion streams differed from the
original sequential receipt. The selected kernel now assigns independent
SIMDgroups to gate and up projections, retains the original per-projection Q5_0
instruction order, rounds both projection outputs to FP16 before activation,
and synchronizes them inside one threadgroup. That replacement is compiled and
linked but not part of this observation; its follow-up is recorded below.

## 4.8 Instruction-order-preserving gated-linear follow-up

The two-SIMDgroup gated kernel measured `94.946972 tok/s` at 128 tokens and
`55.652426 tok/s` at 512 tokens, `2.9439%` and `2.5124%` above corrected
prebaked GQA3. Its 128-token completion matched the original sequential receipt
exactly; the 512-token completion did not.

The long-context median remained `17.747164 ms`, compared with `9.413958 ms`
at 128 tokens. The next attention package keeps split8's eight-way context
parallelism but assigns contiguous temporal chunks and merges those chunks in
chronological online-softmax order instead of interleaving positions across
splits. It is compiled and linked for the next isolated observation.

## 4.9 Contiguous split-context attention

The contiguous-chunk split8 package measured `125.904092 tok/s` at 128 tokens
and `130.969856 tok/s` at 512 tokens. Median inter-token latency changed from
`5.984068 ms` to `7.339001 ms` across those lengths, flattening the context
growth observed with the unsplit attention kernel. The completion streams did
not match the original sequential receipt.

## 4.10 Four-SIMDgroup Q8_0 vocabulary projection

The single-token `49152x576` Q8_0 vocabulary projection now assigns two output
rows to one 128-thread threadgroup and splits K across its four SIMDgroups. With
contiguous split8 attention unchanged, one chronological 128/512 campaign
measured `135.133528 tok/s` and `137.560438 tok/s`, ratios of `1.073305290` and
`1.050321366` over the preceding package. The 512-token result is `36.0682%` of
the `381.39 tok/s` llama.cpp sequential reference; no completion-stream
comparator was run for this observation.

## 4.11 Four-row Q5_0 projection observation

A llama.cpp-shaped Q5_0 tactic assigned four output rows to each SIMDgroup and
two SIMDgroups to each threadgroup. With the Q8_0 head and contiguous split8
unchanged, it measured `137.111571 tok/s` at 128 tokens and `135.997683 tok/s`
at 512 tokens. Those are ratios of `1.014637692` and `0.988639503` against the
Q8-head control. The tactic was removed after the measurement; no
completion-stream comparator was run.

## 4.12 Prebaked dependency and profiling correction

One 512-token profiled campaign measured `139.438162 tok/s`, but the existing
batch profiler reported only 37 command buffers and did not cover any of the
long-context ICB decode steps. Those 37 buffers averaged `5.857061 ms` of GPU
time and `7.254234 ms` of commit-and-wait time, so they cannot decompose the
reported per-token latency.

The prebaked native path now reports its own GPU and wait duration. Its ICB
also carries the schedule's `barrier-wait` boundaries into each recorded
dispatch instead of setting a barrier before every dispatch indiscriminately.
The corrected path compiles and is not part of the profiling observation above.

## 4.13 Zero-barrier ICB observation

The first profiled 128/512 campaign after adding prebaked timing reported 372
recorded dispatches and zero barriers. It measured `171.668334 tok/s` and
`215.092082 tok/s`; the 512-token prebaked path averaged `1.586993 ms` of GPU
time and `2.178189 ms` through command completion. This is `56.3969%` of the
`381.39 tok/s` llama.cpp reference.

The zero count showed that schedule specialization had pruned every barrier
before precompilation. The fully concurrent ICB could race workspace aliases
whose lifetimes were planned for sequential execution, so this observation is
recorded but not retained as the dependency policy. Specialization now
preserves barrier commands, allowing workspace planning and ICB execution to
use the same stage boundaries. That corrected path compiles and is unmeasured.

## 4.14 Stage-preserving ICB observation

With specialization retaining schedule barriers, the ICB recorded 372
dispatches and 245 barriers. One 128/512-token campaign measured
`148.038654/145.079279 tok/s`; the 512-token result is `1.054658455x` the Q8-head
control and `38.0396%` of the `381.39 tok/s` llama.cpp sequential reference.

The 128/512 completion hashes exactly equal the contiguous split8 control. At
512 tokens, prebaked GPU time averaged `3.793287 ms`, command completion
averaged `4.410893 ms`, and whole inter-token time averaged `6.793853 ms`.

## 4.15 Direct shared-buffer argmax observation

Greedy decode now scans the final FP16 logits row in the shared Metal buffer
instead of allocating and copying 98,304 bytes into OCaml each token. One
128/512-token campaign measured `147.725691/146.959392 tok/s`; both completion
hashes exactly equal the stage-preserving control.

At 512 tokens this is `1.012959211x` the stage-preserving control and `38.5326%`
of the `381.39 tok/s` llama.cpp reference. Mean inter-token time changed by
`-0.084024 ms`; prebaked GPU/wait means were `3.780689/4.380801 ms`. Static
inspection then found that page-size-one radix lookup allocated every remaining
token suffix at each tree depth. It also found that the already-created
long-context ICB still rebuilt and cached a 1,249-command specialized schedule
for every new past length even though execution ignored those schedules.

## 4.16 Host-path candidate regression

The combined offset-indexed radix walk and lazy schedule-specialization campaign
measured `191.729340/229.125803 tok/s` at 128/512 tokens, but both streams
diverged from the direct-argmax control at completion index 2 (`1517` control,
`4165` candidate). These timings are recorded as a changed-output observation,
not as the active throughput result.

The radix rewrite was removed after the observation. The available pure-cache
tests preserved slot order under both implementations and therefore did not
exercise the model-level divergence. Lazy specialization was then measured in
isolation as recorded below.

## 4.17 Lazy schedule-specialization regression

With the original radix walk restored, lazy post-ICB schedule specialization
measured `194.809814/218.221446 tok/s` at 128/512 tokens. It reproduced the
combined candidate's exact changed hashes and the same first divergence at
completion index 2 (`1517` control, `4165` candidate), isolating the changed
output to the schedule-lifetime edit rather than the radix rewrite.

The timing is recorded as changed-output evidence and the lazy specialization
edit was removed. The direct-argmax observation remains the active measured
result at `146.959392 tok/s` for 512 tokens.

## 4.18 Schedule-lifetime correction

The prebaked plan now retains the schedule that created its execution and ICB.
One executable compared eager and lazy schedule modes at 128/512 tokens; both
modes reproduced the established hashes at each length. Lazy mode measured
`205.514830/217.933883 tok/s`, versus `148.080915/147.043224 tok/s` control.

At 512 tokens this is `1.482107622x` control and `57.1420%` of the recorded
`381.39 tok/s` llama.cpp result. Mean inter-token time was `4.487488 ms`, with
`3.664962 ms` mean GPU time and `3.844650 ms` through command completion. The
corrected path is retained behind `LLMOPT_EXPERIMENTAL_LAZY_SCHEDULE=1`.

## 4.19 Prebaked kernel inventory

An exact 128/512-token repeat with plan profiling measured
`198.283242/217.977505 tok/s` and reproduced the established completion hashes.
The 512-token run recorded `3.660957 ms` mean GPU time and `3.842525 ms` through
command completion.

The retained decode plan contains 372 dispatches. Its measured inventory includes
76 Q5_0 linear, 14 Q8_0 linear, 61 RMSNorm, 60 cache-pack, 30 rotary, 30 paged
attention, 30 Q5_0 linear-add, and 30 Q5_0 gated-linear dispatches. From the graph
relationships and these counts, 90 Q/K/V projections plus 30 attention-input
RMSNorm operations form the next 120-dispatch fusion target; that grouping is an
inference from the measured inventory, not a measured fusion result.

## 4.20 Mixed-quantized QKV fusion

The compiler now recognizes one shared activation feeding the topology-linked Q,
K, and V projections and emits one `Quant_qkv_linear` operation. Its Metal
lowering preserves the existing per-output Q5_0 and Q8_0 accumulation order and
supports the two Slice31 layouts, `Q5_0/Q5_0/Q5_0` and
`Q5_0/Q5_0/Q8_0`.

Fresh source capture against the exact 273-tensor GGUF found 30 fused operations
in each prefill, decode, and verification graph. All three packages compile and
validate with zero opaque commands; the decode package contains 1,189 IR
commands, representing 30 QKV dispatches instead of 90 separate projections.

Relinking those captured graphs with `--no-greedy` produced an ABI-v5 engine
with 30 attention layers and zero opaque commands. One exact profiled 128/512
campaign measured `179.214981/218.577515 tok/s`; both completion hashes match
the established baseline. At 512 tokens the fused plan records 312 dispatches
instead of 372, mean GPU/wait time changes by `-0.027321/-0.030780 ms`, and
throughput changes by `+0.275262%`. The recorded `381.39 tok/s` llama.cpp
reference remains `1.744873x` faster.

## 4.21 Paired K/V cache packing

The next measured cluster contains 60 cache-pack dispatches, two per attention
layer. The cache runtime now exposes one paired SIMD kernel that independently
computes the existing FP16-stored Q8 scale and quantized values for K and V in
one dispatch. New packages select the paired kernel; older packages fall back
to the two existing dispatches.

Relinking the same captured graphs produced 98-entry, zero-opaque prefill,
decode, and verification packages. One exact profiled 128/512 campaign measured
`174.324529/219.697086 tok/s`; both completion hashes match the established
baseline. At 512 tokens the paired plan records 282 dispatches instead of 312,
mean GPU/wait time changes by `-0.016562/-0.016679 ms`, and throughput changes
by `+0.512208%` relative to QKV-only fusion. The recorded `381.39 tok/s`
llama.cpp reference remains `1.735981x` faster.

## 4.22 llama.cpp-style Q5_0 matrix-vector candidate

Current llama.cpp commit `0eadefebd3f8f92a86d634a0e5b8fffc9dc792c0`
computes four Q5_0 output rows per SIMDgroup with two SIMDgroups per
threadgroup. LLMOpt now has an opt-in `LLMOPT_EXPERIMENTAL_Q5_LLAMA_MV=1`
implementation of that structure for the 30 gated FFN projections and 30
down-projection residual adds in single-token decode.

The Slice31 candidate compiles 100-entry, zero-opaque prefill, decode, and
verification packages containing both new kernels. One profiled 128/512
campaign measured `185.383602/238.003417 tok/s`. At 512 tokens, mean GPU/wait
time changes by `-0.322576/-0.323965 ms` and throughput changes by `+8.332532%`
relative to paired K/V; the llama.cpp reference is `1.602456x` faster. The
128-token stream is exact, while the 512-token stream first differs at
zero-based completion index 459 (`94` versus `2692`).

## 4.23 llama.cpp-style Q4_K and Q6_K matrix-vector candidate

The remaining down projections contain 16 Q4_K and 14 Q6_K dispatches. LLMOpt
now implements current llama.cpp's two-output-row, two-SIMDgroup matrix-vector
structure for both formats behind `LLMOPT_EXPERIMENTAL_KQUANT_LLAMA_MV=1`.

The combined Slice31 engine compiles 102-entry, zero-opaque prefill, decode, and
verification packages containing the Q5_0, Q4_K, and Q6_K candidates. One
profiled 128/512 campaign measured `204.910372/243.728265 tok/s`. At 512 tokens,
mean GPU/wait time changes by `-0.102621/-0.100407 ms` and throughput changes by
`+2.405364%` relative to Q5-only. Both completion streams match Q5-only; the
128-token stream remains exact against the original, while the 512-token stream
retains the same first divergence at index 459. The llama.cpp reference remains
`1.564816x` faster.

## 4.24 llama.cpp-style Q8_0 vocabulary matrix-vector candidate

The remaining plan contains one `49152x576` Q8_0 vocabulary projection. The
existing LLMOpt tactic assigns one quantized value to each lane and splits K
across four SIMDgroups. Current llama.cpp instead has each lane dot eight
contiguous quantized values while four SIMDgroups cover blocks and jointly
reduce two output rows.

LLMOpt now implements that structure behind
`LLMOPT_EXPERIMENTAL_Q8_LLAMA_MV=1`. The combined Slice31 engine compiles
103-entry, zero-opaque prefill, decode, and verification packages containing
all four llama-style format paths.

## 4.25 llama.cpp-style Q8_0 vocabulary measurement

One profiled 128/512 campaign measured `199.051336/248.837338 tok/s`. The
128-token result changes by `-2.859316%` versus the Q5/Q4_K/Q6_K control; the
512-token result changes by `+2.096217%`, with mean GPU and command-completion
time changing by `-0.047170/-0.053177 ms`. Both completion hashes exactly match
that control. The 128-token stream remains exact against the original, while
the 512-token stream retains the first divergence at completion index 459.

The 512-token result is `65.2449%` of the recorded `381.39 tok/s` llama.cpp
sequential result. Its `3.938270 ms` whole inter-token mean exceeds the
`3.317517 ms` mean command-completion time by `0.620753 ms`; this difference
includes host work around execution and is not attributed to one operation.

## 4.26 Allocation-reduced radix match candidate

Page-size-one prefix matching previously allocated a complete remaining-token
suffix at each traversed radix edge. The opt-in
`LLMOPT_EXPERIMENTAL_RADIX_OFFSET_WALK=1` path indexes child keys and compares
compressed edges directly against the original token array and offset, leaving
lease acquisition, resumable-node selection, splitting, and returned slot
assembly unchanged.

The full Python and OCaml suites pass, including a transformer-only cache test
that obtains the same three-token match without a recurrent checkpoint through
the new branch.

## 4.27 Allocation-reduced radix match measurement

One same-executable control/candidate campaign measured
`235.462676/245.045496 tok/s` at 128 tokens and
`255.105371/261.745024 tok/s` at 512 tokens. Both completion hashes are exact
between modes and equal the preceding Q8 candidate hashes.

At 512 tokens, the candidate changes throughput by `+2.602710%` and whole
inter-token mean by `-0.102124 ms`. Mean GPU and command-completion time change
by only `-0.006197/-0.014298 ms`; the remaining difference between whole
inter-token and command-completion means changes from `0.612980` to
`0.525155 ms`. At 128 tokens, whole inter-token mean changes by only
`-0.001287 ms`, so its `+4.069783%` total-throughput observation is primarily a
startup-time difference rather than steady decode.

The measured match candidate remains opt-in. Its 512-token result is
`68.6292%` of the recorded `381.39 tok/s` llama.cpp sequential result.

## 4.28 Allocation-reduced radix insertion candidate

Static inspection found the same suffix-allocation pattern in
`Radix_cache.insert`: inserting each generated prefix copied the complete
remaining token suffix at every existing page-size-one edge before allocating
the final new leaf. The separate opt-in
`LLMOPT_EXPERIMENTAL_RADIX_OFFSET_INSERT=1` branch now indexes and compares
existing edges against the original token array and offset, allocating a token
suffix only when it creates the final leaf.

The full Python and OCaml suites pass. The transformer-only cache regression
inserts three prefixes through the candidate, matches them through both radix
walk implementations, verifies canonical slot order, and validates cache/KV
ownership.

## 4.29 Allocation-reduced radix insertion measurement

One counterbalanced same-executable campaign ran control then candidate at 128
tokens and candidate then control at 512 tokens. It measured
`221.787638/233.089729 tok/s` at 128 tokens and
`252.693474/260.882516 tok/s` at 512 tokens. Both candidate hashes exactly
equal their controls and the established Q8 candidate streams.

At 512 tokens, insertion changes throughput by `+3.240702%` and whole
inter-token mean by `-0.123897 ms`. Mean GPU and command-completion time change
by only `-0.002787/-0.009308 ms`; the difference between whole inter-token and
command-completion means changes from `0.536185` to `0.421596 ms`. At 128
tokens, whole inter-token mean changes by `-0.128654 ms` and throughput by
`+5.095907%`.

The insertion candidate remains opt-in alongside the measured match candidate.
Its 512-token observation is `68.4031%` of the recorded `381.39 tok/s`
llama.cpp sequential result.

## 4.30 Direct radix path-value assembly candidate

After traversal, the retained radix match assembled slots by first allocating
one list cell per page, then a boxed option array, then a second result array;
`Serving_cache.Match.slots` subsequently copied that result once more. The
opt-in `LLMOPT_EXPERIMENTAL_RADIX_DIRECT_VALUES=1` branch now computes the
path length and fills one ordered result array backwards with `Array.blit`,
without changing the public copied-slot boundary.

The full Python and OCaml suites pass. The transformer cache regression matches
the same three canonical slot identifiers through the direct assembly branch
and releases the lease normally.

## 4.31 Direct radix path-value assembly measurement

One counterbalanced same-executable campaign measured
`220.235127/235.581186 tok/s` at 128 tokens and
`261.640521/261.189949 tok/s` at 512 tokens. Both candidate hashes exactly
equal their controls and the established Q8 candidate streams.

At 512 tokens, direct assembly changes throughput by `-0.172210%` and whole
inter-token mean by `+0.003770 ms`. Its host residual changes by
`-0.013203 ms`, but mean GPU and command-completion time change by
`+0.014493/+0.016973 ms`, offsetting that observation. At 128 tokens, whole
inter-token mean changes by `-0.113936 ms`; this short result is reported
separately from the neutral sustained result.

The direct-value branch remains opt-in as allocation evidence rather than an
end-to-end sustained speed result. Its 512-token observation is `68.4837%` of
the recorded `381.39 tok/s` llama.cpp sequential result.

## 4.32 Borrowed radix match-value candidate

`Radix_cache.match_prefix` owns an isolated matched-slot array inside its
lease. The existing `matched_values` API copied that lease-local array before
sequential decode only read it and appended the one reserved slot. New
`matched_values_view` and `Serving_cache.Match.slots_view` APIs expose a
documented read-only borrow valid until lease release; sequential decode selects
it behind `LLMOPT_EXPERIMENTAL_RADIX_SHARED_VALUES=1`. Other cache callers keep
the copied API.

The full Python and OCaml suites pass, including exact canonical slot-order
coverage for both copied and borrowed match values.

## 4.33 Borrowed radix match-value measurement

One counterbalanced same-executable campaign measured the copied control at
`221.268438/261.909394 tok/s` and the borrowed candidate at
`235.149153/262.093098 tok/s` for 128/512 tokens. Both completion hashes are
exact. At 512 tokens, throughput changes by `+0.183704 tok/s` (`+0.070140%`),
whole inter-token mean by `-0.002305 ms`, GPU mean by `-0.002903 ms`, and
command-completion mean by `-0.002320 ms`. The 128-token result changes by
`+6.273247%`, but the sustained observation is neutral. The 512-token candidate
is `68.7205%` of the recorded `381.39 tok/s` llama.cpp sequential result.

## 4.34 llama.cpp-style all-Q5 fused QKV candidate

The fused QKV kernel still assigned one SIMDgroup to each output row after the
ordinary Q5_0 Linear path adopted llama.cpp-style block decomposition. The new
opt-in `LLMOPT_EXPERIMENTAL_QKV_LLAMA_MV=1` kernel assigns four SIMDgroups to a
128-thread threadgroup and computes sixteen combined Q/K/V output rows per
threadgroup using the existing llama-style Q5_0 dot product. Selection is
restricted to single-token fused projections where Q, K, and V weights are all
Q5_0; the fourteen mixed Q5_0/Q5_0/Q8_0 projections remain unchanged.

The exact Slice31 v9 engine contains sixteen matching all-Q5 projections in
each graph. Prefill/decode/verification validate with 104 kernels, zero opaque
commands, and unchanged `1069/1189/1219` IR command counts. All 72 Python tests
and the OCaml suite pass. This candidate is statically compiled and not yet
device-measured.

## 4.35 Rejected stale-executable QKV attempt

The first intended control/candidate campaign did not execute the candidate.
Both labeled modes recorded the identical 282-dispatch inventory, including
sixteen `llmopt_q5_0_qkv_linear_f16` dispatches and zero
`llmopt_q5_0_qkv_linear_f16_llama_mv` dispatches. The executable hash was the
pre-selector `813dcc...21f`; therefore its `261.885343/261.689908 tok/s`
control/candidate-labeled numbers are invalid as QKV evidence and the receipt
sets `measurement` to null.

The Ninja-built generator was relinked to hash `e774d6...74d`. A pure selector
seam now covers enabled all-Q5 single-token selection, disabled control,
mixed-Q8 value weights, and multi-token fallback; the executable contains both
the environment flag and candidate kernel name. No second device attempt was
performed.

## 4.36 Valid all-Q5 llama.cpp-style QKV measurement

The corrected counterbalanced campaign asserted the runtime inventories before
accepting timing: control selected sixteen original all-Q5 QKV dispatches and
zero llama-style dispatches, while candidate selected sixteen llama-style
dispatches and zero originals. Both 128- and 512-token completion hashes are
exact.

At 512 tokens, the candidate measures `262.832556 tok/s` versus
`262.277636 tok/s` control, a change of `+0.554920 tok/s` (`+0.211577%`). Mean
inter-token, GPU, and command-completion times change by
`-0.008136/-0.014371/-0.012009 ms`. The separately reported 128-token endpoint
changes by `-1.519578%` despite GPU and wait means falling by
`0.045139/0.049256 ms`. The sustained candidate is `68.9144%` of the recorded
`381.39 tok/s` llama.cpp sequential result.

## 4.37 llama.cpp-style mixed Q5/Q5/Q8 fused QKV candidate

The same opt-in tactic now covers the fourteen remaining single-token QKV
dispatches whose Q/K weights are Q5_0 and V weight is Q8_0. The new 128-thread,
four-SIMDgroup kernel computes sixteen combined output rows per threadgroup,
using the existing llama-style Q5 dot for Q/K and eight-element signed Q8
chunks for V. Together with the measured all-Q5 kernel, the flag now selects
all thirty fused QKV dispatches per Slice31 graph.

The exact Slice31 v10 prefill/decode/verification packages compile with 105
kernels, zero opaque commands, and unchanged `1069/1189/1219` IR command
counts. Five selector cases, 72 Python tests, and the OCaml suite pass. This
mixed extension has a separate flag so its effect can be isolated from the
already measured all-Q5 tactic.

## 4.38 llama.cpp-style mixed Q5/Q5/Q8 fused QKV measurement

One counterbalanced campaign asserted all four relevant dispatch inventories.
Control retained sixteen llama-style all-Q5 dispatches plus fourteen original
mixed dispatches; candidate retained those sixteen and replaced all fourteen
mixed dispatches with the new kernel. Both 128- and 512-token completion hashes
are exact between modes.

At 512 tokens, candidate measures `265.441442 tok/s` versus `262.299452 tok/s`
control, a change of `+3.141990 tok/s` (`+1.197864%`). Mean inter-token, GPU,
and command-completion times change by `-0.045457/-0.059236/-0.069441 ms`.
The separately reported 128-token endpoint changes by `+2.608502%`, while its
mean inter-token time changes by `+0.081104 ms`. The sustained candidate is
`69.5984%` of the recorded `381.39 tok/s` llama.cpp sequential result.

## 4.39 Rejected four-SIMDgroup wide RMSNorm candidate

The tested opt-in tactic targeted the 61 single-row, width-576, F16-input and
F32-weight wide RMSNorm dispatches in each Slice31 decode graph. The
`LLMOPT_EXPERIMENTAL_RMS_WIDE_SG4=1` kernel uses 128 threads and four
SIMDgroups instead of the retained 256-thread, eight-SIMDgroup wide kernel.

The exact Slice31 v11 prefill/decode/verification packages compile with 106
kernels, zero opaque commands, and unchanged `1069/1189/1219` IR command
counts. All 72 Python tests and the OCaml suite pass.

One counterbalanced campaign asserted 61 original dispatches in control and 61
SG4 dispatches in candidate. At 512 tokens, candidate measures
`260.579821 tok/s` versus `264.882324 tok/s` control, a change of
`-4.302503 tok/s` (`-1.624307%`). Mean inter-token, GPU, and command-completion
times increase by `0.057752/0.057840/0.064566 ms`. Both 128- and 512-token
candidate streams first differ from control at completion index 92 (`9212`
instead of `198`), consistent with the changed RMS reduction grouping. The
candidate implementation and selector were removed; the eight-SIMDgroup
kernel remains active.

## 4.40 Lease-backed sequential decode cursor

The retained sequential driver previously reconstructed and searched the full
generated prefix on every token: `Generation_core` copied the prior step's
tokens, `Serving_engine.decode` walked the radix tree from its root, rebuilt the
complete slot array twice, then walked the root again to insert the one-token
extension. The 512-token control reports 511 radix hits and a `0.421009 ms`
difference between mean inter-token and command-completion time.

The opt-in `LLMOPT_EXPERIMENTAL_SEQUENCE_CURSOR=1` path acquires the canonical
page-size-one prefix once and carries that lease through the generation step.
Each decode extends directly from the current radix node, retains the next
canonical lease before releasing the previous one, and avoids both root walks
and the second full slot-array reconstruction. Generation completion, initial sampling
failure, decode failure, and explicit session close release the active lease.

The exact v10 packages remain unchanged and validate with 105 kernels, zero
opaque commands, and `1069/1189/1219` IR commands. Radix ownership/order and
generation release tests, all 72 Python tests, and the OCaml suite pass.

One counterbalanced campaign asserted 127/511 control radix hits and one
candidate hit at 128/512 tokens while retaining all 16 all-Q5 and 14 mixed-QKV
llama-style dispatches. At 512 tokens, candidate measures `269.675491 tok/s`
versus `266.400688 tok/s` control, a change of `+3.274803 tok/s`
(`+1.229277%`). Mean inter-token, GPU, command-completion, and derived host
residual times change by `-0.056436/-0.008668/-0.018738/-0.037698 ms`. Both
completion hashes are exact. This result is `70.7086%` of the recorded
`381.39 tok/s` llama.cpp sequential result, and the cursor remains opt-in.

## 4.41 Incremental sequence slot-index upload

The measured cursor still passed its complete radix slot array to
`Metal_runtime.Cache.q8_attention_inputs` on every token. That path mapped the
typed slots into a fresh OCaml integer array and rewrote the complete stable
Metal slot-index buffer, even though a sequential cursor adds exactly one
canonical slot per step.

The measured candidate required the sequence cursor, uploaded the initial
prefix once, then wrote only the appended four-byte slot index after each
successful canonical extension. Decode bound the same stable Metal buffer
without rebuilding or rewriting the prefix.

One counterbalanced campaign selected the candidate marker only in candidate
runs, retained one radix hit and miss plus all 16 all-Q5 and 14 mixed-QKV
llama-style dispatches in every run, and produced exact 128/512 completion
hashes. At 512 tokens, candidate measured `268.302177 tok/s` versus
`268.338720 tok/s` control, a change of `-0.036543 tok/s` (`-0.013618%`). Mean
inter-token/GPU/command-completion/host-residual time changed by
`+0.004310/+0.004712/+0.006910/-0.002600 ms`. The candidate was removed; the
retained cursor continues to use the complete slot upload. The post-removal
full Ninja build, OCaml suite, and all 72 Python tests pass.

## 4.42 Maskless one-token paged decode

The retained 512-token cursor control selected eight mask-construction GPU
dispatches per decode: two arange, two index, two integer-add, one compare, and
one boolean-expand dispatch. The selected one-token paged-Q8 attention kernels
do not read mask values because their mask branch is guarded by query length
greater than one.

`LLMOPT_EXPERIMENTAL_MASKLESS_DECODE=1` makes decode specialization omit the
mask semantic input, allowing ordinary schedule pruning to remove its producer
subgraph. Runtime binding supplies the slots buffer again at the existing
kernel's unused mask ABI position, so packages and attention arithmetic remain
unchanged. Suffix prefill and prompt lookup retain their real masks.

A focused schedule test confirms that a one-token decode loses its mask runtime
input and carries five semantic paged-attention inputs. The exact v10 packages
remain at 105 kernels, zero opaque commands, and `1069/1189/1219` IR commands;
the full Ninja build, OCaml suite, and all 72 Python tests pass. The eight
dispatches are the target counted in the prior control inventory.

One counterbalanced campaign confirmed the candidate reduces the prebaked plan
from `282` dispatches and `245` barriers to `274` dispatches and `244` barriers,
while preserving the 16 all-Q5 and 14 mixed-QKV llama-style selections and
exact 128/512 completion hashes. At 512 tokens it measures `268.599738 tok/s`
versus `266.310432 tok/s` control, a change of `+2.289306 tok/s`
(`+0.859638%`). Mean inter-token/GPU/command-completion/host-residual time
changes by `-0.008918/-0.007271/-0.009269/+0.000351 ms`. The candidate remains
opt-in and measures `70.4265%` of the recorded `381.39 tok/s` llama.cpp result.

## 4.43 Combined GQA3 split8 attention

The retained long-context split8 kernel launches one 256-thread threadgroup per
query head, so each of the three query heads mapped to a KV head independently
dequantizes the same paged Q8 keys and values. The shorter-context GQA3 kernel
already shares those loads, but does not split the temporal reduction.

`LLMOPT_EXPERIMENTAL_GQA3_SPLIT8=1` combines both structures for one-token
long-context decode. Each threadgroup owns one KV head, eight SIMDgroups retain
the chronological temporal chunks and merge order, and the group computes all
three query-head score and value accumulators while sharing each K/V load. For
Slice31 this changes each of the 30 attention dispatches from nine to three
threadgroups without changing dispatch count.

The selector has four shape/flag cases and the generated source-presence test
passes. Apple Metal compilation succeeds for all three fresh v12 packages; the
ABI-v5 Model Program and packages validate with 106 kernels, zero opaque
commands, and unchanged `1069/1189/1219` IR commands. The full Ninja build,
OCaml suite, and all 72 Python tests pass. This candidate has no model/device
measurement.

One counterbalanced v12 campaign selected 30 control split8 kernels versus 30
combined GQA3-split8 kernels with the same `274` dispatches and `244` barriers.
At 512 tokens, candidate measured `375.581810 tok/s` versus `268.621140 tok/s`
control (`+39.818411%`), and mean GPU/command-completion time changed by
`-1.141392/-1.140426 ms`. This is `98.4771%` of the recorded `381.39 tok/s`
llama.cpp result. Both candidate streams first diverged from control at
zero-based completion index 41 (`76` versus `198`), so v12 is changed-output
path evidence rather than parity evidence.

Metal defaults to unsafe floating-point reassociation. The v13 repair disables
reassociation across the paged-Q8 value reconstruction and attention kernels so
the control and grouped paths retain the written per-head operation order while
still sharing K/V work. All three regenerated sources contain the directive,
compile through Apple Metal, and produce distinct source/metallib hashes from
v12; packages remain at 106 kernels, zero opaque commands, and unchanged
`1069/1189/1219` IR commands. The ABI-v5 Model Program, OCaml suite, and all 72
Python tests pass.

The single counterbalanced v13 campaign still first diverged at zero-based
completion index 41 (`76` versus `198`) at both 128 and 512 tokens. At 512
tokens it measured `375.310260 tok/s` candidate versus `268.987283 tok/s`
control (`+39.527139%`), with mean GPU/command-completion time changing by
`-1.144372/-1.143576 ms`; the candidate is `98.4059%` of the recorded
`381.39 tok/s` llama.cpp result. Disabling reassociation therefore did not
repair parity or materially alter the GQA3 path timing, and the directive was
removed from source after the measurement.

## 4.44 Serial-score, shared-V GQA3 split8

The v14 diagnostic candidate kept the three-head/eight-temporal-SIMDgroup
geometry but reconstructed keys and performed the online-softmax update once
per query head in serial source blocks. Only paged value reconstruction remained
shared, isolating interleaved K scoring as the changed variable.

One counterbalanced campaign selected the intended 30 candidate kernels with
the same `274` dispatches and `244` barriers. Its 128- and 512-token hashes
exactly equal the original v12 grouped candidate hashes and still first diverge
from control at completion index 41 (`76` versus `198`). At 512 tokens it
measured `348.737495 tok/s` versus `268.170972 tok/s` control
(`+30.042969%`), with mean GPU/command-completion time changing by
`-0.909130/-0.910659 ms`. Serial scoring therefore did not affect the output
divergence and reduced throughput by `26.844315 tok/s` versus v12; the source
change was removed.

## 4.45 Matched-token sustained llama.cpp comparison

The earlier `381.39 tok/s` llama.cpp reference used a 48-token raw prompt while
LLMOpt used a 28-token ChatML prompt. The corrected campaign decodes LLMOpt's
ChatML IDs with special tokens and independently retokenizes them through the
same GGUF in `llama-tokenize`; both runtimes receive the exact sequence of 28
IDs. The run order is llama.cpp then LLMOpt at 512 tokens, followed by LLMOpt
then llama.cpp at 1,024 tokens.

LLMOpt selects all 30 combined GQA3 split8 kernels, 16 all-Q5 and 14 mixed-QKV
llama-style kernels, and profiles `274` dispatches with `244` barriers per
decode step. It measures `376.065890 tok/s` versus llama.cpp `389.8 tok/s` at
512 tokens, a ratio of `0.964766x`, and `413.437096 tok/s` versus
`387.1 tok/s` at 1,024 tokens, a ratio of `1.068037x`. Both ratios are inside
the repository's existing user-declared `0.90x` to `1.10x` performance parity
envelope.

The result does not establish output parity. Retokenized streams differ at the
first generated token (`198` LLMOpt versus `13433` llama.cpp), and the runtime
state formats differ: paged Q8 KV in LLMOpt versus default F16 KV in llama.cpp.
Performance parity and cross-runtime output parity remain separate claims.

## 4.46 Matched Q8-KV llama.cpp comparison

The same 28-token trace and counterbalanced order were repeated with llama.cpp
`-ctk q8_0 -ctv q8_0`; LLMOpt retained its paged group-64 Q8 KV path. At
512 tokens LLMOpt measured `301.159736 tok/s` versus llama.cpp
`278.4 tok/s` (`1.081752x`), and at 1,024 tokens LLMOpt measured
`321.751415 tok/s` versus llama.cpp `286.3 tok/s` (`1.123826x`). llama.cpp's
Q8-KV output hashes exactly equal its prior F16-KV hashes at both lengths, and
LLMOpt also reproduced its prior hashes.

This campaign ran slower overall: llama.cpp changed by `-28.578758%` and
`-26.039783%` relative to the prior F16 campaign, while contemporaneous LLMOpt
changed by `-19.918359%` and `-22.176452%`. macOS reported no thermal or
performance warning and zero throttled VM pages. The same-campaign LLMOpt versus
llama.cpp ratios are evidence; the separate-campaign Q8-versus-F16 differences
are confounded by the shared slowdown.
