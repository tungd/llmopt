---
type: Decision
title: 'Macro-Operator Fusions: Dual-Linear, QKV, ShortConv Step, Residual-Norm, and On-GPU Argmax'
description: 'Introduce five high-impact compiler fusion passes to eliminate intermediate DRAM traffic and launch overheads across SwiGLU FFNs, QKV attention projections, ShortConv recurrent steps, residual normalization, and vocabulary argmax sampling.'
tags: [decision, compiler, fusion, passes, metal, swiglu, qkv, shortconv, argmax]
status: draft
generated: { by: codex/gpt-5, at: '2026-08-26T02:34:00Z' }
sources:
  - id: local-passes
    resource: /lib/passes.ml
    title: Compiler optimization passes and graph rewrite rules
  - id: local-metal
    resource: /lib/metal.ml
    title: Metal Shading Language emitter and fused kernel templates
  - id: local-schedule
    resource: /lib/serving_schedule.ml
    title: Serving schedule serializer and shape inference
  - id: local-sampling
    resource: /lib/sampling.ml
    title: Greedy float16 argmax sampling
---

# Problem Statement & Motivation

While `llmopt` already implements pointwise and residual fusions (`Q8_linear_silu`, `Q8_linear_add`, `Q8_linear_mul_add`, `Rms_rope`), the captured model graphs still incur unnecessary intermediate buffer round-trips to Metal device memory:

1. **Dual-Read FFN Projections:** SwiGLU $w_1$ (Gate) and $w_3$ (Up) linear projections read the identical input activation tensor from DRAM twice across all 16 FFN blocks.
2. **Triplicate Attention Projections:** $Q$, $K$, and $V$ projections read input activations three separate times across all 6 attention layers.
3. **Fragmented Recurrent Conv Step:** ShortConv decode executes separate slice roll, depthwise 1D conv, pointwise activation, and state copy dispatches.
4. **Intermediate Residual Allocations:** Attention and FFN out-projections write un-normalized residuals to DRAM immediately before post-RMSNorm reads them back.
5. **Vocabulary Logit Writeback:** `lm_head` writes $65,536 \times 2$ bytes (131 KB) of float16 logits to memory on every decode token step solely for the CPU to compute `argmax`.

---

# The Five Macro-Operator Fusions

```
1. SwiGLU Dual-Linear:
   x ──► [ Fused Q8 Dual Linear: [W1; W3]x ] ──► (w1_out, w3_out)  (Halves input DRAM reads)

2. 3-in-1 QKV Linear:
   x ──► [ Fused Q8 QKV Linear: [Wq; Wk; Wv]x ] ──► (Q, K, V)       (3x input read reduction)

3. Fused ShortConv Step:
   x, conv_state ──► [ Fused ShortConv SIMD Step ] ──► y, new_conv_state (1 dispatch)

4. Fused Out-Proj + Residual + Post-Norm:
   x, residual ──► [ Fused Linear-Add-RMSNorm ] ──► normalized_y   (Zero residual DRAM round-trip)

5. Fused LM_Head + GPU Argmax:
   hidden_state ──► [ Fused RMSNorm + LM_Head + Tree Argmax ] ──► token_id (4-byte output, 131 KB saved)
```

---

# Technical Specification for Each Fusion

### 1. Fused Dual-Linear Q8 GEMV (`Passes.fuse_dual_linear_swiglu`)
* **IR Op:** `Ir.Op.Q8_dual_linear { m; n1; n2; k; bias }`
* **Metal Kernel:** `llmopt_q8_dual_linear_f16`
* Loads input activation $x[0 \dots k-1]$ into threadgroup memory once and computes both $w_1$ and $w_3$ projections in parallel SIMD groups.

### 2. Fused 3-in-1 QKV Linear (`Passes.fuse_qkv_linear`)
* **IR Op:** `Ir.Op.Q8_qkv_linear { m; n_q; n_kv; k; bias }`
* **Metal Kernel:** `llmopt_q8_qkv_linear_f16`
* Produces query tensor $Q$ and key/value tensors $K, V$ in a single threadgroup sweep over $x$.

### 3. Fused ShortConv Step (`Passes.fuse_short_conv_step`)
* **IR Op:** `Ir.Op.Short_conv_step_fused`
* **Metal Kernel:** `llmopt_short_conv_step_fused_f16`
* Shifts the 3-element historical conv state in registers, computes depthwise dot product, evaluates SiLU activation, and writes back the updated state buffer.

### 4. Fused Linear-Add-RMSNorm (`Passes.fuse_linear_residual_norm`)
* **IR Op:** `Ir.Op.Q8_linear_add_norm { m; n; k; epsilon }`
* **Metal Kernel:** `llmopt_q8_linear_add_norm_f16`
* Accumulates linear output with residual $y = x_{\text{in}} + W_o x$, computes the root-mean-square reduction across SIMD lanes, and stores normalized FP16 activations.

### 5. Fused LM_Head + On-GPU Tree-Reduction Argmax (`Passes.fuse_lm_head_argmax`)
* **IR Op:** `Ir.Op.Q8_lm_head_argmax { m; n; k; epsilon }`
* **Metal Kernel:** `llmopt_q8_lm_head_argmax`
* Computes final RMSNorm, projects to vocabulary channels, executes threadgroup tree reduction over $65,536$ logit scores, and outputs a 4-byte `uint32` token ID to host memory.

---

# Verification & Success Gates

1. **Bitwise / Numerical Parity:**
   * Max absolute difference $< 10^{-4}$ against CPU reference in `lib/cpu.ml`.
   * Argmax token output matches greedy reference decoding with 100% exact token parity.
2. **Command Count Reduction:**
   * Total prefill and decode commands reduced by an additional $\ge 120$ commands across the full 16-layer model.
3. **DRAM Bandwidth & Latency:**
   * Measure single-token decode latency ($\text{TPOT}$) reduction $\ge 20\%$ on Apple Silicon GPU.
