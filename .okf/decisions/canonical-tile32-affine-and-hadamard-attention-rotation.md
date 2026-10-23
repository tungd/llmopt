---
type: Decision
title: 'Mathematical Reduction of Runtime Quantization: Canonical Tile32_Affine Storage and In-Register Hadamard Attention Rotation'
description: 'Eliminate the multi-format GGUF kernel zoo by transcoding all GGUF/UD block formats losslessly offline into one canonical 32-element SIMD-aligned affine format (Tile32_Affine4 / Tile32_Affine8), unifying contraction codegen over B in {4, 8}, and introducing a 5-step in-register Walsh-Hadamard butterfly (simd_fwht32) for outlier-free affine KV caching.'
tags: [decision, compiler, algebra, quantization, tile32-affine, hadamard, fwht, kv-cache, gguf, metal, mathematical-reduction]
status: stable
decision_status: approved
generated: { by: antigravity/2.0, at: '2026-10-23T16:52:00+07:00' }
verified: { by: human:tung, at: '2026-10-23T16:47:51+07:00' }
sources:
  - id: algebraic-ir
    resource: /decisions/applicative-ir-coordinate-category-and-epilogue-monoid.md
    title: Applicative IR, Affine Coordinate Category, and Contraction Epilogue Monoid
  - id: constructive-lowering
    resource: /decisions/constructive-primitive-loop-lowering-and-gpu-queue-compilation.md
    title: Constructive Assignment Lowering and AOT GPU Command-Queue Compilation
  - id: gguf-ud
    resource: /decisions/gguf-unsloth-dynamic-quantization.md
    title: Target GGUF and Unsloth Dynamic (UD) Quantization
  - id: exl3-quarot
    resource: /prior-art/exl3-quarot-hadamard.md
    title: EXL3, QuaRot, and SpinQuant prior-art analysis
  - id: block-schema
    resource: /lib/block_schema.ml
    title: Validated physical and logical contracts for quantized Linear storage
  - id: gguf-transcode
    resource: /lib/gguf.ml
    title: Native GGUF parser and offline transcoder
---

# Decision

`llmopt` adopts **mathematical reduction** across its quantization, contraction,
and KV-cache runtime stack:

1. **One Canonical Runtime Block Format (`Tile32_Affine` for $B \in \{4, 8\}$)**:
   All seven GGUF block/superblock formats (`Q4_0`, `Q5_0`, `Q8_0`, `Q4_K`,
   `Q5_K`, `Q6_K`, `IQ4_XS`) are transcoded **losslessly offline** into a single
   32-element SIMD-aligned affine block representation:
   $$w_i = s \cdot q_i + b \quad (i \in 0..31,\; q_i \in [0, 2^B - 1])$$
   where every 32-element block carries an identical 4-byte header
   `(half s, half b)` followed by $4B$ bytes of unsigned $B$-bit lane-aligned codes.
2. **One Parametric Contraction Kernel Generator**:
   Linear, multi-slot (gated/QKV), and multi-export contractions over quantized
   weights lower through a single parametric `Tile32_Affine` emitter
   parameterized by $B \in \{4, 8\}$, tile geometry, and the `Contraction_group`
   epilogue expression:
   $$\langle a, w \rangle = \sum_{\text{block}} \left( s \sum_{i=0}^{31} a_i \, q_i + b \sum_{i=0}^{31} a_i \right)$$
3. **In-Register Fast Walsh-Hadamard Butterfly (`simd_fwht32`) & Unified Affine KV Cache**:
   Attention queries, keys, and values support orthogonal 32-element
   Walsh-Hadamard rotation (`simd_fwht32`) executed entirely in registers via 5
   `simd_shuffle_xor` butterfly steps, dispersing channel outliers so the paged
   KV cache uses the same canonical 32-element affine block structure.

---

# Context & Motivation

While the front-half of `llmopt` achieved a compact algebraic reduction (`1,345`
lines across `Coordinate_map`, `Applicative_expr`, `Epilogue`, and
`Contraction_group`),[^algebraic-ir] the Metal backend grew to `21,913` lines.
The root cause is **GGUF structural heterogeneity**: Unsloth Dynamic checkpoints
(`UD-Q4_K_XL`) mix `Q4_0`, `Q5_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, and `IQ4_XS`
within the same model.[^gguf-ud] Each format changes the block size (`32` vs
`256`), scale bit-packing (6-bit packed scale/min tables vs signed `int8`
tables vs power-of-two exponents), signedness, and high-bit planes (`qh`).

We evaluated adopting **EXL3** (trellis quantization with Hadamard rotation) as
the unified runtime format and converting GGUF checkpoints to EXL3.[^exl3-quarot]
However:
- Converting an already-quantized GGUF to EXL3 is **lossy** (requiring
  dequantization, Hadamard rotation, and Viterbi trellis re-quantization).
- EXL3 itself carries multiple codebook variants (`mul1`, `mcg`, `3inst`) and
  sequential shift-register dependencies for odd bitwidths ($K \in \{3, 5, 6, 7\}$).

By contrast, inspecting [`lib/block_schema.ml`](file:///Users/tung/Projects/std23/llmopt/lib/block_schema.ml) reveals that **every GGUF format
is already a special case of a 32-element affine map** $w_i = s \cdot q_i + b$
once its subblock scale, minimum, high-bit planes, or 16-entry `IQ4_XS` codebook
values are unpacked.[^block-schema]

---

# Mathematical Specification

## 1. Lossless GGUF $\to$ `Tile32_Affine{4,8}` Homomorphism

For each 32-element subblock (matching one 32-lane Apple Silicon SIMDgroup):

| Source GGUF Format | Target Canonical Format | Scale $s$ (`half` @ $+0$) | Bias $b$ (`half` @ $+2$) | Unsigned Code $q_i$ ($i \in 0..31$ @ $+4$) | Lossless? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Q4_0`** (18B / 32) | **`Tile32_Affine4`** (20B / 32) | $d$ | $\text{f16}(-8 \cdot d)$ | $q_i \in [0..15]$ (nibbles preserved verbatim) | **Exact** |
| **`Q4_K`** (144B / 256) | **`Tile32_Affine4`** ($8 \times 20\text{B}$) | $\text{f16}(d \cdot sc_{sb})$ | $\text{f16}(-d_{\text{min}} \cdot m_{sb})$ | $q_i \in [0..15]$ (subblock nibbles) | **Exact** |
| **`Q8_0`** (34B / 32) | **`Tile32_Affine8`** (36B / 32) | $d$ | $\text{f16}(-128 \cdot d)$ | $\text{int8}_i + 128 \in [0..255]$ | **Exact** |
| **`Q5_0`** (22B / 32) | **`Tile32_Affine8`** (36B / 32) | $d$ | $\text{f16}(-16 \cdot d)$ | $(\text{nib}_i \mid (\text{qh}_i \ll 4)) \in [0..31]$ | **Exact** |
| **`Q5_K`** (176B / 256) | **`Tile32_Affine8`** ($8 \times 36\text{B}$) | $\text{f16}(d \cdot sc_{sb})$ | $\text{f16}(-d_{\text{min}} \cdot m_{sb})$ | $(\text{nib}_i \mid (\text{qh}_{sb,i} \ll 4)) \in [0..31]$ | **Exact** |
| **`Q6_K`** (210B / 256) | **`Tile32_Affine8`** ($8 \times 36\text{B}$) | $\text{f16}(d \cdot sc_{sb})$ | $\text{f16}(-32 \cdot s)$ | $(\text{ql}_i \mid (\text{qh}_i \ll 4)) \in [0..63]$ | **Exact** |
| **`IQ4_XS`** (136B / 256) | **`Tile32_Affine8`** ($8 \times 36\text{B}$) | $\text{f16}(d \cdot 2^{(ls_{sb}-32)/8})$ | $\text{f16}(-128 \cdot s)$ | $\text{LUT}_{\text{iq4nl}}[\text{nib}_i] + 128 \in [1..241]$ | **Exact** |

### Physical Layout Invariants
- **`Tile32_Affine4` (`20` bytes per 32 elements, alignment 2)**:
  - Bytes `0..1`: `half s`
  - Bytes `2..3`: `half b`
  - Bytes `4..19`: 16 bytes of packed nibbles where byte $l \in 0..15$ stores $q_l$ in the low nibble (`& 0xF`) and $q_{l+16}$ in the high nibble (`>> 4`), enabling coalesced 4-byte (`ushort p_lo, p_hi`) Track-C vectorized loads across 4-lane clusters.
- **`Tile32_Affine8` (`36` bytes per 32 elements, alignment 2)**:
  - Bytes `0..1`: `half s`
  - Bytes `2..3`: `half b`
  - Bytes `4..35`: 32 bytes of unsigned `uchar` codes $q_0..q_{31}$, enabling coalesced 4-byte (`uchar4` / `uint`) vectorized loads across 8-lane clusters.

## 2. In-Register Walsh-Hadamard Butterfly (`simd_fwht32`)

For any 32-element slice $x \in \mathbb{R}^{32}$ distributed across the 32 lanes
of a Metal SIMDgroup, the normalized Walsh-Hadamard transform
$\tilde{x} = \frac{1}{\sqrt{32}} H_{32} x$ is self-inverse ($H_{32}^T = H_{32}$,
$\tilde{H}_{32}^2 = I_{32}$) and orthogonal ($\langle \tilde{H}_{32} q, \tilde{H}_{32} k \rangle = \langle q, k \rangle$).

In Metal Shading Language, $\tilde{H}_{32}$ executes in 5 register-only butterfly stages:
```msl
inline float llmopt_simd_fwht32(float x, uint lane) {
  for (uint step = 1u; step < 32u; step <<= 1u) {
    const float partner = simd_shuffle_xor(x, ushort(step));
    x = ((lane & step) != 0u ? -x : x) + partner;
  }
  return x * 0.17677669529663689f; // 1.0f / sqrt(32.0f)
}
```
- **Zero shared memory, zero threadgroup barriers**: Composes directly with `Rms_rope`, KV-cache packing, andpaged attention.
- **Unified Affine KV Cache**: After optional `llmopt_simd_fwht32` rotation, per-head $K$ and $V$ slices are quantized into 32-element affine blocks (`Tile32_Affine8` or `Tile32_Affine4`), unifying weight and KV-cache storage schemas.

---

# Implementation Phases

1. **Step 1 (Lossless Transcoder & Schema Contract)**: Implement `Tile32_Affine4` and `Tile32_Affine8` schemas in [`lib/block_schema.ml`](file:///Users/tung/Projects/std23/llmopt/lib/block_schema.ml) and lossless transcoders for all 7 GGUF formats in [`lib/gguf.ml`](file:///Users/tung/Projects/std23/llmopt/lib/gguf.ml) (`Gguf.Transcode`), verified for exact element-wise equivalence in `test/test_quant.ml`.
2. **Step 2 (Unified Parametric Contraction Codegen)**: Emit single-slot, multi-slot, and multi-export `Tile32_Affine` ($B \in \{4, 8\}$) kernels from [`lib/metal_block_codegen.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal_block_codegen.ml) using the factored affine inner loop $s \sum a_i q_i + b \sum a_i$, and verify on-device Metal execution parity.
3. **Step 3 (Hadamard Attention Rotation & Affine KV Cache)**: Implement `llmopt_simd_fwht32` and `Tile32_Affine` KV cache pack/unpack and rotated paged attention kernels in [`lib/metal_cache_source.ml`](file:///Users/tung/Projects/std23/llmopt/lib/metal_cache_source.ml), verified for orthogonality, involutivity, and outlier-dispersion fidelity in `test/test_cache.ml`.

[^algebraic-ir]: llmopt decision - Applicative IR, Affine Coordinate Category, and Contraction Epilogue Monoid.
[^constructive-lowering]: llmopt decision - Constructive Assignment Lowering and AOT GPU Command-Queue Compilation.
[^gguf-ud]: llmopt decision - Target GGUF and Unsloth Dynamic (UD) Quantization.
[^exl3-quarot]: llmopt prior art - EXL3, QuaRot, and SpinQuant analysis.
[^block-schema]: `lib/block_schema.ml` - Validated physical and logical contracts for quantized Linear storage.
[^gguf-transcode]: `lib/gguf.ml` - Native GGUF parser and offline transcoder.
