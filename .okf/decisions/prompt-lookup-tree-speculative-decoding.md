---
type: Decision
title: 'Weightless Prompt-Lookup (N-Gram) Tree Speculative Decoding'
description: 'Establishes the weightless n-gram pattern matching algorithm and tree verification contract for zero-overhead speculative decoding on Apple Silicon GPU.'
tags: [decision, prompt-lookup, n-gram, speculative-decoding, tree-attention, zero-weights, metal]
status: stable
generated: { by: 'process:antigravity', at: '2026-09-01T00:22:00+07:00' }
sources:
  - id: serving-engine
    resource: /lib/serving_engine.ml
    title: Serving engine and Medusa tree verification
  - id: tree-attention
    resource: /lib/metal.ml
    title: Metal speculative tree attention kernel
---

# 1. Context & Motivation

Neural speculative drafters (e.g. MTP, small assistant models) incur a non-trivial drafting latency $T_{\text{draft}}$ per token. On unified memory systems like Apple Silicon where memory bandwidth is shared between CPU and GPU, running a 0.25–1.0 GB assistant model sequentially $K$ times adds $10\text{--}20\text{ ms}$ of latency, degrading net throughput whenever the empirical acceptance rate $\alpha < \frac{T_{\text{draft}}}{T_{\text{target}} \cdot (1 - \frac{T_{\text{draft}}}{T_{\text{target}}})}$.

**Weightless Prompt-Lookup Decoding** eliminates neural drafting overhead entirely:
1. **Zero Parameters**: No additional draft weights or separate neural networks are loaded into memory.
2. **Zero DRAM Latency ($T_{\text{draft}} \approx 0$)**: Candidate token sequences are extracted directly from the existing context / prompt history using fast sliding n-gram matching in CPU L1 cache ($< 10\,\mu\text{s}$).
3. **Single-Pass Tree Verification**: Proffered candidates are verified concurrently in a single target model forward pass using the Metal tree-attention kernel (`llmopt_attention_tree_speculative_f16`) with causal bitmasks.

# 2. Mathematical Formulation

Given the token history $T = [t_0, t_1, \dots, t_{N-1}]$:

1. **Suffix Matching**: For a search window $n \in [\text{max\_ngram}, \text{min\_ngram}]$ (typically $n \in [3, 1]$), extract the current query suffix:
   $$W = [t_{N-n}, \dots, t_{N-1}]$$
2. **History Scan**: Find the latest matching index $j < N - n$ in $T$:
   $$T[j - n + 1 \dots j] = W$$
3. **Draft Proposal**: Propose the subsequent $K$ tokens following the match:
   $$\text{Draft} = [t_{j+1}, \dots, t_{j+K}]$$
4. **Tree Formatting & Verification**:
   - Construct a `Medusa.tree` and causal bitmasks (`tree_masks`).
   - Execute the target Model Program with tree attention in 1 pass.
   - Accept the longest matching prefix and emit the bonus target token via `Medusa.accept_tree`.

# 3. Expected Speedup Model

Because $T_{\text{draft}} \approx 0\,\mu\text{s}$, the speculative speedup simplifies to:

$$\text{Speedup} = \frac{1 + \alpha \cdot K}{1 + \epsilon_{\text{tree}}} \approx 1 + \alpha \cdot K$$

where $\epsilon_{\text{tree}}$ is the negligible overhead of computing attention over $K$ extra query positions in SRAM.
