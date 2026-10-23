---
type: Prior Art
title: 'EXL3, QuaRot, and SpinQuant: Hadamard incoherence rotation and unified low-bit quantization'
description: 'Analysis of ExLlamaV3 (EXL3) trellis quantization and QuaRot/SpinQuant orthogonal Walsh-Hadamard rotations for eliminating activation/KV outliers and unifying runtime quantization formats.'
tags: [prior-art, quantization, exl3, quarot, spinquant, hadamard, fwht, kv-cache, trellis]
status: stable
stale_after: '2027-10-23'
generated: { by: antigravity/2.0, at: '2026-10-23T16:52:00+07:00' }
verified: { by: human:tung, at: '2026-10-23T16:47:51+07:00' }
sources:
  - id: exllamav3
    resource: https://github.com/turboderp-org/exllamav3
    title: ExLlamaV3 and the EXL3 trellis quantization format
  - id: quarot
    resource: https://arxiv.org/abs/2404.00456
    title: 'QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs'
  - id: spinquant
    resource: https://arxiv.org/abs/2405.16406
    title: 'SpinQuant: LLM Quantization with Learned Rotations'
  - id: local-canonical-tile32
    resource: /decisions/canonical-tile32-affine-and-hadamard-attention-rotation.md
    title: Canonical Tile32_Affine and Hadamard attention rotation decision
---

# Overview

Modern low-bit quantization systems (**ExLlamaV3 / EXL3**, **QuaRot**, and
**SpinQuant**) address two problems that plague legacy GGUF K-quants:

1. **Structural Heterogeneity**: In GGUF (`UD-Q4_K_XL`), mixed precision requires switching between incompatible C structs (`Q4_0`, `Q5_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_XS`), forcing compilers into a combinatorial kernel zoo.
2. **Channel Outliers**: Transformer hidden states and KV-cache tensors exhibit extreme per-channel magnitude spikes (up to $100\times$ median values), degrading low-bit uniform quantization unless outliers are dispersed beforehand.[^quarot][^spinquant]

# Core Mechanisms

## 1. EXL3 (QTIP-Based Trellis Quantization)
EXL3 represents every quantized linear layer—from 2 to 8 bits per weight (`bpw`)—using a single $16 \times 16$ tile structure ($32 \times K$ bytes per 256 weights) preceded and followed by Walsh-Hadamard rotations ($W' = S_v H W H^T S_u$).[^exllamav3]
- **What it unifies**: Mixed precision across layers changes only the bit-stride parameter $K \in \{2..8\}$, not the scale layout or dequantization algebra.
- **Where EXL3 still fragments**:
  1. Multiple procedural codebooks (`mul1`, legacy `mcg`, experimental `3inst`).
  2. Sequential shift-register state dependencies (`state = (state << K) | bits`) and unaligned bit-streams for $K \in \{3, 5, 6, 7\}$.
  3. **Lossy GGUF Conversion**: Converting an already-quantized GGUF (`Q4_K`/`Q5_K`) to EXL3 requires dequantizing, rotating by $H$, and re-running Viterbi trellis quantization, compounding quantization error.

## 2. QuaRot and SpinQuant (Orthogonal Attention & KV Rotation)
QuaRot and SpinQuant exploit the orthogonal invariance of the normalized Walsh-Hadamard Transform ($H^T H = I$):[^quarot][^spinquant]
- **Attention Score Invariance ($R_3$)**: Rotating queries and keys after RoPE via block-diagonal Hadamard blocks $H_{32}$ preserves the dot product identically:
  $$\langle H_{32} q, H_{32} k \rangle = q^T H_{32}^T H_{32} k = \langle q, k \rangle$$
- **Value Rotation ($R_4$)**: Rotating $V \mapsto H_{32} V$ before KV-cache quantization and applying the inverse rotation $H_{32}$ to the attention output (or absorbing $H_{32}$ offline into $W_O \mapsto W_O H_{32}$) preserves the exact attention output while making both $K$ and $V$ Gaussian and outlier-free.

# Synthesis for `llmopt`

Rather than adopting EXL3's sequential trellis state machine (which cannot losslessly transcode existing GGUF checkpoints), `llmopt` combines:
1. **Lossless Offline GGUF $\to$ `Tile32_Affine` Transcoding**: Pre-evaluates all GGUF subblock scales, minimums, and `IQ4_XS` codebooks at link time into one canonical 32-element SIMD-aligned affine block format (`Tile32_Affine4` / `Tile32_Affine8`) with zero re-quantization loss.[^local-canonical-tile32]
2. **In-Register Fast Walsh-Hadamard Butterfly (`simd_fwht32`)**: Implements the 32-lane butterfly transform in 5 `simd_shuffle_xor` instructions with zero shared memory and zero extra dispatches, enabling outlier-free `Tile32_Affine` quantized KV caching and rotated attention.[^local-canonical-tile32]

[^exllamav3]: ExLlamaV3 (`turboderp-org/exllamav3`) architecture and EXL3 format notes.
[^quarot]: Ashkboos et al., *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs* (2024).
[^spinquant]: Liu et al., *SpinQuant: LLM Quantization with Learned Rotations* (2024).
[^local-canonical-tile32]: llmopt decision - Canonical Tile32_Affine and Hadamard attention rotation.
