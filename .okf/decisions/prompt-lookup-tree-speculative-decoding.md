---
type: Decision
title: 'Weightless Prompt-Lookup (N-Gram) Tree Speculative Decoding'
description: 'Records the linked weightless n-gram proposal and target-verification path with exact token parity and comparative measurements.'
tags: [decision, prompt-lookup, n-gram, speculative-decoding, tree-attention, zero-weights, metal]
status: stable
generated: { by: 'process:antigravity', at: '2026-09-01T00:22:00+07:00' }
sources:
  - id: serving-engine
    resource: /lib/serving_engine.ml
    title: Serving engine prompt-lookup verification
  - id: tree-attention
    resource: /lib/metal.ml
    title: Metal speculative tree attention kernel
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
