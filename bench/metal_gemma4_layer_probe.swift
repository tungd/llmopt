// Benchmark Gemma 4 12B Transformer Layer Quantized Projections on Apple Silicon Metal
// Verifies Track C 4B-vectorized loop speedup on full Gemma 4 12B layer shapes:
// 1. QKV: K=3840, N_q=4096, N_k=2048, N_v=2048 (16.9 MB)
// 2. Attn Out: K=4096, N=3840 (8.4 MB)
// 3. FFN Gated (Gate + Up): K=3840, N=15360 (63.3 MB)
// 4. FFN Down: K=15360, N=3840 (31.6 MB)
// Total weights per layer: 120.2 MB. Across 48 layers: 5.77 GB.

import Foundation
import Metal

func require(_ condition: Bool, _ message: String) {
    if !condition { fatalError(message) }
}

guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    fatalError("Metal unavailable")
}

let mslSource = """
#include <metal_stdlib>
using namespace metal;

struct QuantLinearParams {
    uint m;
    uint n;
    uint k;
    uint has_bias;
};

struct QuantQKVParams {
    uint m;
    uint n_q;
    uint n_k;
    uint n_v;
    uint k;
};

inline float gelu_quick(float x) {
    return x * (1.0f / (1.0f + metal::exp(-1.702f * x)));
}

// -----------------------------------------------------------------------------
// Baseline kernels
// -----------------------------------------------------------------------------
kernel void q4_0_linear_baseline_x8(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint base_col = simd_idx << 3u;
  if (base_col >= params.n) return;
  const uint blocks = params.k / 32u;
  const uint col_stride = blocks * 18u;
  device const uchar* row_blocks0 = weight + base_col * col_stride;
  device const uchar* row_blocks1 = weight + min(base_col + 1u, params.n - 1u) * col_stride;
  device const uchar* row_blocks2 = weight + min(base_col + 2u, params.n - 1u) * col_stride;
  device const uchar* row_blocks3 = weight + min(base_col + 3u, params.n - 1u) * col_stride;
  device const uchar* row_blocks4 = weight + min(base_col + 4u, params.n - 1u) * col_stride;
  device const uchar* row_blocks5 = weight + min(base_col + 5u, params.n - 1u) * col_stride;
  device const uchar* row_blocks6 = weight + min(base_col + 6u, params.n - 1u) * col_stride;
  device const uchar* row_blocks7 = weight + min(base_col + 7u, params.n - 1u) * col_stride;
  device const half* activation = input;
  float accumulator0 = 0.0f, accumulator1 = 0.0f, accumulator2 = 0.0f, accumulator3 = 0.0f;
  float accumulator4 = 0.0f, accumulator5 = 0.0f, accumulator6 = 0.0f, accumulator7 = 0.0f;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;
  const uint half_blocks = blocks >> 1u;
  for (uint b = 0u; b < half_blocks; ++b) {
    const uint block_idx = (b << 1u) + b_offset;
    const uint blk_byte_off = block_idx * 18u;
    device const uchar* blk0 = row_blocks0 + blk_byte_off;
    device const uchar* blk1 = row_blocks1 + blk_byte_off;
    device const uchar* blk2 = row_blocks2 + blk_byte_off;
    device const uchar* blk3 = row_blocks3 + blk_byte_off;
    device const uchar* blk4 = row_blocks4 + blk_byte_off;
    device const uchar* blk5 = row_blocks5 + blk_byte_off;
    device const uchar* blk6 = row_blocks6 + blk_byte_off;
    device const uchar* blk7 = row_blocks7 + blk_byte_off;
    const half scale0 = *reinterpret_cast<device const half*>(blk0 + 0u);
    const half scale1 = *reinterpret_cast<device const half*>(blk1 + 0u);
    const half scale2 = *reinterpret_cast<device const half*>(blk2 + 0u);
    const half scale3 = *reinterpret_cast<device const half*>(blk3 + 0u);
    const half scale4 = *reinterpret_cast<device const half*>(blk4 + 0u);
    const half scale5 = *reinterpret_cast<device const half*>(blk5 + 0u);
    const half scale6 = *reinterpret_cast<device const half*>(blk6 + 0u);
    const half scale7 = *reinterpret_cast<device const half*>(blk7 + 0u);
    const uchar packed0 = blk0[2u + l];
    const uchar packed1 = blk1[2u + l];
    const uchar packed2 = blk2[2u + l];
    const uchar packed3 = blk3[2u + l];
    const uchar packed4 = blk4[2u + l];
    const uchar packed5 = blk5[2u + l];
    const uchar packed6 = blk6[2u + l];
    const uchar packed7 = blk7[2u + l];
    const uint k_idx = block_idx * 32u + l;
    const float a_low = float(activation[k_idx]);
    const float a_high = float(activation[k_idx + 16u]);
    const float q0_low = float(int(packed0 & 15u) - 8);
    const float q0_high = float(int(packed0 >> 4u) - 8);
    const float q1_low = float(int(packed1 & 15u) - 8);
    const float q1_high = float(int(packed1 >> 4u) - 8);
    const float q2_low = float(int(packed2 & 15u) - 8);
    const float q2_high = float(int(packed2 >> 4u) - 8);
    const float q3_low = float(int(packed3 & 15u) - 8);
    const float q3_high = float(int(packed3 >> 4u) - 8);
    const float q4_low = float(int(packed4 & 15u) - 8);
    const float q4_high = float(int(packed4 >> 4u) - 8);
    const float q5_low = float(int(packed5 & 15u) - 8);
    const float q5_high = float(int(packed5 >> 4u) - 8);
    const float q6_low = float(int(packed6 & 15u) - 8);
    const float q6_high = float(int(packed6 >> 4u) - 8);
    const float q7_low = float(int(packed7 & 15u) - 8);
    const float q7_high = float(int(packed7 >> 4u) - 8);
    accumulator0 += float(scale0) * (a_low * q0_low + a_high * q0_high);
    accumulator1 += float(scale1) * (a_low * q1_low + a_high * q1_high);
    accumulator2 += float(scale2) * (a_low * q2_low + a_high * q2_high);
    accumulator3 += float(scale3) * (a_low * q3_low + a_high * q3_high);
    accumulator4 += float(scale4) * (a_low * q4_low + a_high * q4_high);
    accumulator5 += float(scale5) * (a_low * q5_low + a_high * q5_high);
    accumulator6 += float(scale6) * (a_low * q6_low + a_high * q6_high);
    accumulator7 += float(scale7) * (a_low * q7_low + a_high * q7_high);
  }
  accumulator0 = simd_sum(accumulator0);
  accumulator1 = simd_sum(accumulator1);
  accumulator2 = simd_sum(accumulator2);
  accumulator3 = simd_sum(accumulator3);
  accumulator4 = simd_sum(accumulator4);
  accumulator5 = simd_sum(accumulator5);
  accumulator6 = simd_sum(accumulator6);
  accumulator7 = simd_sum(accumulator7);
  if (lane == 0u && base_col < params.n) {
    output[base_col] = half(accumulator0);
    if (base_col + 1u < params.n) output[base_col + 1u] = half(accumulator1);
    if (base_col + 2u < params.n) output[base_col + 2u] = half(accumulator2);
    if (base_col + 3u < params.n) output[base_col + 3u] = half(accumulator3);
    if (base_col + 4u < params.n) output[base_col + 4u] = half(accumulator4);
    if (base_col + 5u < params.n) output[base_col + 5u] = half(accumulator5);
    if (base_col + 6u < params.n) output[base_col + 6u] = half(accumulator6);
    if (base_col + 7u < params.n) output[base_col + 7u] = half(accumulator7);
  }
}

kernel void q4_0_gated_baseline_x8(
    device const half* input [[buffer(0)]],
    device const uchar* gate_weight [[buffer(1)]],
    device const uchar* up_weight [[buffer(2)]],
    device half* product [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint col_groups = (params.n + 7u) >> 3u;
  const uint total_elements = params.m * col_groups;
  if (simd_idx >= total_elements) return;
  const uint row = simd_idx / col_groups;
  const uint group_idx = simd_idx - row * col_groups;
  const uint base_col = group_idx << 3u;
  device const half* in_ptr = input + row * params.k;
  const uint blocks = params.k / 32u;
  const uint col_stride = blocks * 18u;
  device const uchar* gate_row0 = gate_weight + base_col * col_stride;
  device const uchar* gate_row1 = gate_weight + min(base_col + 1u, params.n - 1u) * col_stride;
  device const uchar* gate_row2 = gate_weight + min(base_col + 2u, params.n - 1u) * col_stride;
  device const uchar* gate_row3 = gate_weight + min(base_col + 3u, params.n - 1u) * col_stride;
  device const uchar* gate_row4 = gate_weight + min(base_col + 4u, params.n - 1u) * col_stride;
  device const uchar* gate_row5 = gate_weight + min(base_col + 5u, params.n - 1u) * col_stride;
  device const uchar* gate_row6 = gate_weight + min(base_col + 6u, params.n - 1u) * col_stride;
  device const uchar* gate_row7 = gate_weight + min(base_col + 7u, params.n - 1u) * col_stride;
  device const uchar* up_row0 = up_weight + base_col * col_stride;
  device const uchar* up_row1 = up_weight + min(base_col + 1u, params.n - 1u) * col_stride;
  device const uchar* up_row2 = up_weight + min(base_col + 2u, params.n - 1u) * col_stride;
  device const uchar* up_row3 = up_weight + min(base_col + 3u, params.n - 1u) * col_stride;
  device const uchar* up_row4 = up_weight + min(base_col + 4u, params.n - 1u) * col_stride;
  device const uchar* up_row5 = up_weight + min(base_col + 5u, params.n - 1u) * col_stride;
  device const uchar* up_row6 = up_weight + min(base_col + 6u, params.n - 1u) * col_stride;
  device const uchar* up_row7 = up_weight + min(base_col + 7u, params.n - 1u) * col_stride;
  float gate_acc0 = 0.0f, gate_acc1 = 0.0f, gate_acc2 = 0.0f, gate_acc3 = 0.0f;
  float gate_acc4 = 0.0f, gate_acc5 = 0.0f, gate_acc6 = 0.0f, gate_acc7 = 0.0f;
  float up_acc0 = 0.0f, up_acc1 = 0.0f, up_acc2 = 0.0f, up_acc3 = 0.0f;
  float up_acc4 = 0.0f, up_acc5 = 0.0f, up_acc6 = 0.0f, up_acc7 = 0.0f;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;
  const uint half_blocks = blocks >> 1u;
  for (uint b = 0u; b < half_blocks; ++b) {
    const uint block_idx = (b << 1u) + b_offset;
    const uint blk_byte_off = block_idx * 18u;
    const float a_low = float(in_ptr[block_idx * 32u + l]);
    const float a_high = float(in_ptr[block_idx * 32u + l + 16u]);
    #define ACC_ROW(g_row, u_row, g_acc, u_acc) \\
      { \\
        device const uchar* g_blk = g_row + blk_byte_off; \\
        device const uchar* u_blk = u_row + blk_byte_off; \\
        const half g_scale = *reinterpret_cast<device const half*>(g_blk + 0u); \\
        const half u_scale = *reinterpret_cast<device const half*>(u_blk + 0u); \\
        const uchar g_packed = g_blk[2u + l]; \\
        const uchar u_packed = u_blk[2u + l]; \\
        const float g_ql = float(int(g_packed & 15u) - 8); \\
        const float g_qh = float(int(g_packed >> 4u) - 8); \\
        const float u_ql = float(int(u_packed & 15u) - 8); \\
        const float u_qh = float(int(u_packed >> 4u) - 8); \\
        g_acc += float(g_scale) * (a_low * g_ql + a_high * g_qh); \\
        u_acc += float(u_scale) * (a_low * u_ql + a_high * u_qh); \\
      }
    ACC_ROW(gate_row0, up_row0, gate_acc0, up_acc0);
    ACC_ROW(gate_row1, up_row1, gate_acc1, up_acc1);
    ACC_ROW(gate_row2, up_row2, gate_acc2, up_acc2);
    ACC_ROW(gate_row3, up_row3, gate_acc3, up_acc3);
    ACC_ROW(gate_row4, up_row4, gate_acc4, up_acc4);
    ACC_ROW(gate_row5, up_row5, gate_acc5, up_acc5);
    ACC_ROW(gate_row6, up_row6, gate_acc6, up_acc6);
    ACC_ROW(gate_row7, up_row7, gate_acc7, up_acc7);
    #undef ACC_ROW
  }
  gate_acc0 = simd_sum(gate_acc0); gate_acc1 = simd_sum(gate_acc1);
  gate_acc2 = simd_sum(gate_acc2); gate_acc3 = simd_sum(gate_acc3);
  gate_acc4 = simd_sum(gate_acc4); gate_acc5 = simd_sum(gate_acc5);
  gate_acc6 = simd_sum(gate_acc6); gate_acc7 = simd_sum(gate_acc7);
  up_acc0 = simd_sum(up_acc0); up_acc1 = simd_sum(up_acc1);
  up_acc2 = simd_sum(up_acc2); up_acc3 = simd_sum(up_acc3);
  up_acc4 = simd_sum(up_acc4); up_acc5 = simd_sum(up_acc5);
  up_acc6 = simd_sum(up_acc6); up_acc7 = simd_sum(up_acc7);
  if (lane == 0u) {
    device half* out_row = product + row * params.n;
    out_row[base_col] = half(gelu_quick(gate_acc0) * up_acc0);
    if (base_col + 1u < params.n) out_row[base_col + 1u] = half(gelu_quick(gate_acc1) * up_acc1);
    if (base_col + 2u < params.n) out_row[base_col + 2u] = half(gelu_quick(gate_acc2) * up_acc2);
    if (base_col + 3u < params.n) out_row[base_col + 3u] = half(gelu_quick(gate_acc3) * up_acc3);
    if (base_col + 4u < params.n) out_row[base_col + 4u] = half(gelu_quick(gate_acc4) * up_acc4);
    if (base_col + 5u < params.n) out_row[base_col + 5u] = half(gelu_quick(gate_acc5) * up_acc5);
    if (base_col + 6u < params.n) out_row[base_col + 6u] = half(gelu_quick(gate_acc6) * up_acc6);
    if (base_col + 7u < params.n) out_row[base_col + 7u] = half(gelu_quick(gate_acc7) * up_acc7);
  }
}

// -----------------------------------------------------------------------------
// Vectorized 4B Kernels (Track C)
// -----------------------------------------------------------------------------
kernel void q4_0_linear_vectorized_4b_x8(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint base_col = simd_idx << 3u;
  if (base_col >= params.n) return;
  const uint blocks = params.k / 32u;
  const uint col_stride = blocks * 18u;
  device const uchar* row_blocks0 = weight + base_col * col_stride;
  device const uchar* row_blocks1 = weight + min(base_col + 1u, params.n - 1u) * col_stride;
  device const uchar* row_blocks2 = weight + min(base_col + 2u, params.n - 1u) * col_stride;
  device const uchar* row_blocks3 = weight + min(base_col + 3u, params.n - 1u) * col_stride;
  device const uchar* row_blocks4 = weight + min(base_col + 4u, params.n - 1u) * col_stride;
  device const uchar* row_blocks5 = weight + min(base_col + 5u, params.n - 1u) * col_stride;
  device const uchar* row_blocks6 = weight + min(base_col + 6u, params.n - 1u) * col_stride;
  device const uchar* row_blocks7 = weight + min(base_col + 7u, params.n - 1u) * col_stride;
  device const half* activation = input;

  float accumulator0 = 0.0f, accumulator1 = 0.0f, accumulator2 = 0.0f, accumulator3 = 0.0f;
  float accumulator4 = 0.0f, accumulator5 = 0.0f, accumulator6 = 0.0f, accumulator7 = 0.0f;

  const uint cluster_blocks = (blocks + 7u) >> 3u;
  const uint b_sub = lane >> 2u;
  const uint t_in_b = lane & 3u;
  const uint byte_offset = 2u + (t_in_b << 2u);

  for (uint c = 0u; c < cluster_blocks; ++c) {
    const uint block_idx = (c << 3u) + b_sub;
    if (block_idx < blocks) {
      const uint blk_byte_off = block_idx * 18u;
      const uint l0 = t_in_b << 2u;
      const uint k_idx = block_idx * 32u + l0;
      const float a0_lo = float(activation[k_idx]);
      const float a0_hi = float(activation[k_idx + 16u]);
      const float a1_lo = float(activation[k_idx + 1u]);
      const float a1_hi = float(activation[k_idx + 17u]);
      const float a2_lo = float(activation[k_idx + 2u]);
      const float a2_hi = float(activation[k_idx + 18u]);
      const float a3_lo = float(activation[k_idx + 3u]);
      const float a3_hi = float(activation[k_idx + 19u]);
      const float a_sum8 = 8.0f * (a0_lo + a0_hi + a1_lo + a1_hi + a2_lo + a2_hi + a3_lo + a3_hi);

      #define ACC_ROW(r_blk, acc) \\
        { \\
          device const uchar* blk = r_blk + blk_byte_off; \\
          const float scale = float(*reinterpret_cast<device const half*>(blk)); \\
          const ushort p_lo = *reinterpret_cast<device const ushort*>(blk + byte_offset); \\
          const ushort p_hi = *reinterpret_cast<device const ushort*>(blk + byte_offset + 2u); \\
          acc += scale * ( \\
              a0_lo * float(p_lo & 15u) + \\
              a0_hi * float((p_lo >> 4u) & 15u) + \\
              a1_lo * float((p_lo >> 8u) & 15u) + \\
              a1_hi * float((p_lo >> 12u) & 15u) + \\
              a2_lo * float(p_hi & 15u) + \\
              a2_hi * float((p_hi >> 4u) & 15u) + \\
              a3_lo * float((p_hi >> 8u) & 15u) + \\
              a3_hi * float((p_hi >> 12u) & 15u) - a_sum8); \\
        }
      ACC_ROW(row_blocks0, accumulator0);
      ACC_ROW(row_blocks1, accumulator1);
      ACC_ROW(row_blocks2, accumulator2);
      ACC_ROW(row_blocks3, accumulator3);
      ACC_ROW(row_blocks4, accumulator4);
      ACC_ROW(row_blocks5, accumulator5);
      ACC_ROW(row_blocks6, accumulator6);
      ACC_ROW(row_blocks7, accumulator7);
      #undef ACC_ROW
    }
  }
  accumulator0 = simd_sum(accumulator0); accumulator1 = simd_sum(accumulator1);
  accumulator2 = simd_sum(accumulator2); accumulator3 = simd_sum(accumulator3);
  accumulator4 = simd_sum(accumulator4); accumulator5 = simd_sum(accumulator5);
  accumulator6 = simd_sum(accumulator6); accumulator7 = simd_sum(accumulator7);
  if (lane == 0u && base_col < params.n) {
    output[base_col] = half(accumulator0);
    if (base_col + 1u < params.n) output[base_col + 1u] = half(accumulator1);
    if (base_col + 2u < params.n) output[base_col + 2u] = half(accumulator2);
    if (base_col + 3u < params.n) output[base_col + 3u] = half(accumulator3);
    if (base_col + 4u < params.n) output[base_col + 4u] = half(accumulator4);
    if (base_col + 5u < params.n) output[base_col + 5u] = half(accumulator5);
    if (base_col + 6u < params.n) output[base_col + 6u] = half(accumulator6);
    if (base_col + 7u < params.n) output[base_col + 7u] = half(accumulator7);
  }
}

kernel void q4_0_gated_vectorized_4b_x8(
    device const half* input [[buffer(0)]],
    device const uchar* gate_weight [[buffer(1)]],
    device const uchar* up_weight [[buffer(2)]],
    device half* product [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint col_groups = (params.n + 7u) >> 3u;
  const uint total_elements = params.m * col_groups;
  if (simd_idx >= total_elements) return;
  const uint row = simd_idx / col_groups;
  const uint group_idx = simd_idx - row * col_groups;
  const uint base_col = group_idx << 3u;
  device const half* in_ptr = input + row * params.k;
  const uint blocks = params.k / 32u;
  const uint col_stride = blocks * 18u;
  device const uchar* gate_row0 = gate_weight + base_col * col_stride;
  device const uchar* gate_row1 = gate_weight + min(base_col + 1u, params.n - 1u) * col_stride;
  device const uchar* gate_row2 = gate_weight + min(base_col + 2u, params.n - 1u) * col_stride;
  device const uchar* gate_row3 = gate_weight + min(base_col + 3u, params.n - 1u) * col_stride;
  device const uchar* gate_row4 = gate_weight + min(base_col + 4u, params.n - 1u) * col_stride;
  device const uchar* gate_row5 = gate_weight + min(base_col + 5u, params.n - 1u) * col_stride;
  device const uchar* gate_row6 = gate_weight + min(base_col + 6u, params.n - 1u) * col_stride;
  device const uchar* gate_row7 = gate_weight + min(base_col + 7u, params.n - 1u) * col_stride;
  device const uchar* up_row0 = up_weight + base_col * col_stride;
  device const uchar* up_row1 = up_weight + min(base_col + 1u, params.n - 1u) * col_stride;
  device const uchar* up_row2 = up_weight + min(base_col + 2u, params.n - 1u) * col_stride;
  device const uchar* up_row3 = up_weight + min(base_col + 3u, params.n - 1u) * col_stride;
  device const uchar* up_row4 = up_weight + min(base_col + 4u, params.n - 1u) * col_stride;
  device const uchar* up_row5 = up_weight + min(base_col + 5u, params.n - 1u) * col_stride;
  device const uchar* up_row6 = up_weight + min(base_col + 6u, params.n - 1u) * col_stride;
  device const uchar* up_row7 = up_weight + min(base_col + 7u, params.n - 1u) * col_stride;
  float gate_acc0 = 0.0f, gate_acc1 = 0.0f, gate_acc2 = 0.0f, gate_acc3 = 0.0f;
  float gate_acc4 = 0.0f, gate_acc5 = 0.0f, gate_acc6 = 0.0f, gate_acc7 = 0.0f;
  float up_acc0 = 0.0f, up_acc1 = 0.0f, up_acc2 = 0.0f, up_acc3 = 0.0f;
  float up_acc4 = 0.0f, up_acc5 = 0.0f, up_acc6 = 0.0f, up_acc7 = 0.0f;

  const uint cluster_blocks = (blocks + 7u) >> 3u;
  const uint b_sub = lane >> 2u;
  const uint t_in_b = lane & 3u;
  const uint byte_offset = 2u + (t_in_b << 2u);

  for (uint c = 0u; c < cluster_blocks; ++c) {
    const uint block_idx = (c << 3u) + b_sub;
    if (block_idx < blocks) {
      const uint blk_byte_off = block_idx * 18u;
      const uint l0 = t_in_b << 2u;
      const uint k_idx = block_idx * 32u + l0;
      const float a0_lo = float(in_ptr[k_idx]);
      const float a0_hi = float(in_ptr[k_idx + 16u]);
      const float a1_lo = float(in_ptr[k_idx + 1u]);
      const float a1_hi = float(in_ptr[k_idx + 17u]);
      const float a2_lo = float(in_ptr[k_idx + 2u]);
      const float a2_hi = float(in_ptr[k_idx + 18u]);
      const float a3_lo = float(in_ptr[k_idx + 3u]);
      const float a3_hi = float(in_ptr[k_idx + 19u]);
      const float a_sum8 = 8.0f * (a0_lo + a0_hi + a1_lo + a1_hi + a2_lo + a2_hi + a3_lo + a3_hi);

      #define ACC_ROW(g_row, u_row, g_acc, u_acc) \\
        { \\
          device const uchar* g_blk = g_row + blk_byte_off; \\
          device const uchar* u_blk = u_row + blk_byte_off; \\
          const float g_scale = float(*reinterpret_cast<device const half*>(g_blk)); \\
          const float u_scale = float(*reinterpret_cast<device const half*>(u_blk)); \\
          const ushort g_lo = *reinterpret_cast<device const ushort*>(g_blk + byte_offset); \\
          const ushort g_hi = *reinterpret_cast<device const ushort*>(g_blk + byte_offset + 2u); \\
          const ushort u_lo = *reinterpret_cast<device const ushort*>(u_blk + byte_offset); \\
          const ushort u_hi = *reinterpret_cast<device const ushort*>(u_blk + byte_offset + 2u); \\
          g_acc += g_scale * ( \\
              a0_lo * float(g_lo & 15u) + a0_hi * float((g_lo >> 4u) & 15u) + \\
              a1_lo * float((g_lo >> 8u) & 15u) + a1_hi * float((g_lo >> 12u) & 15u) + \\
              a2_lo * float(g_hi & 15u) + a2_hi * float((g_hi >> 4u) & 15u) + \\
              a3_lo * float((g_hi >> 8u) & 15u) + a3_hi * float((g_hi >> 12u) & 15u) - a_sum8); \\
          u_acc += u_scale * ( \\
              a0_lo * float(u_lo & 15u) + a0_hi * float((u_lo >> 4u) & 15u) + \\
              a1_lo * float((u_lo >> 8u) & 15u) + a1_hi * float((u_lo >> 12u) & 15u) + \\
              a2_lo * float(u_hi & 15u) + a2_hi * float((u_hi >> 4u) & 15u) + \\
              a3_lo * float((u_hi >> 8u) & 15u) + a3_hi * float((u_hi >> 12u) & 15u) - a_sum8); \\
        }
      ACC_ROW(gate_row0, up_row0, gate_acc0, up_acc0);
      ACC_ROW(gate_row1, up_row1, gate_acc1, up_acc1);
      ACC_ROW(gate_row2, up_row2, gate_acc2, up_acc2);
      ACC_ROW(gate_row3, up_row3, gate_acc3, up_acc3);
      ACC_ROW(gate_row4, up_row4, gate_acc4, up_acc4);
      ACC_ROW(gate_row5, up_row5, gate_acc5, up_acc5);
      ACC_ROW(gate_row6, up_row6, gate_acc6, up_acc6);
      ACC_ROW(gate_row7, up_row7, gate_acc7, up_acc7);
      #undef ACC_ROW
    }
  }
  gate_acc0 = simd_sum(gate_acc0); gate_acc1 = simd_sum(gate_acc1);
  gate_acc2 = simd_sum(gate_acc2); gate_acc3 = simd_sum(gate_acc3);
  gate_acc4 = simd_sum(gate_acc4); gate_acc5 = simd_sum(gate_acc5);
  gate_acc6 = simd_sum(gate_acc6); gate_acc7 = simd_sum(gate_acc7);
  up_acc0 = simd_sum(up_acc0); up_acc1 = simd_sum(up_acc1);
  up_acc2 = simd_sum(up_acc2); up_acc3 = simd_sum(up_acc3);
  up_acc4 = simd_sum(up_acc4); up_acc5 = simd_sum(up_acc5);
  up_acc6 = simd_sum(up_acc6); up_acc7 = simd_sum(up_acc7);
  if (lane == 0u) {
    device half* out_row = product + row * params.n;
    out_row[base_col] = half(gelu_quick(gate_acc0) * up_acc0);
    if (base_col + 1u < params.n) out_row[base_col + 1u] = half(gelu_quick(gate_acc1) * up_acc1);
    if (base_col + 2u < params.n) out_row[base_col + 2u] = half(gelu_quick(gate_acc2) * up_acc2);
    if (base_col + 3u < params.n) out_row[base_col + 3u] = half(gelu_quick(gate_acc3) * up_acc3);
    if (base_col + 4u < params.n) out_row[base_col + 4u] = half(gelu_quick(gate_acc4) * up_acc4);
    if (base_col + 5u < params.n) out_row[base_col + 5u] = half(gelu_quick(gate_acc5) * up_acc5);
    if (base_col + 6u < params.n) out_row[base_col + 6u] = half(gelu_quick(gate_acc6) * up_acc6);
    if (base_col + 7u < params.n) out_row[base_col + 7u] = half(gelu_quick(gate_acc7) * up_acc7);
  }
}
"""

let library = try device.makeLibrary(source: mslSource, options: nil)
let pipeLinearBaseline = try device.makeComputePipelineState(function: library.makeFunction(name: "q4_0_linear_baseline_x8")!)
let pipeGatedBaseline = try device.makeComputePipelineState(function: library.makeFunction(name: "q4_0_gated_baseline_x8")!)
let pipeLinearVec4 = try device.makeComputePipelineState(function: library.makeFunction(name: "q4_0_linear_vectorized_4b_x8")!)
let pipeGatedVec4 = try device.makeComputePipelineState(function: library.makeFunction(name: "q4_0_gated_vectorized_4b_x8")!)

func buffer(_ length: Int) -> MTLBuffer {
    device.makeBuffer(length: length, options: .storageModeShared)!
}

func finish(_ cmd: MTLCommandBuffer) {
    cmd.commit()
    cmd.waitUntilCompleted()
    require(cmd.status == .completed, "GPU command failed")
}

let eviction = buffer(256 * 1024 * 1024)
func perturbCache() {
    let cmd = queue.makeCommandBuffer()!
    let enc = cmd.makeBlitCommandEncoder()!
    enc.fill(buffer: eviction, range: 0..<eviction.length, value: 0x5a)
    enc.endEncoding()
    finish(cmd)
}

print("device=\(device.name)")
print("Benchmarking Full Gemma 4 12B Layer Projections (1 layer = 120.2 MB weights)")
print("===============================================================================")

// Define Gemma 4 12B projections:
// 1. QKV: K=3840, N=8192 (16.9 MB)
// 2. Attn Out: K=4096, N=3840 (8.4 MB)
// 3. FFN Gated: K=3840, N=15360 (63.3 MB)
// 4. FFN Down: K=15360, N=3840 (31.6 MB)

struct ProjConfig {
    let name: String
    let n: Int
    let k: Int
    let isGated: Bool
}

let projs = [
    ProjConfig(name: "QKV Projection", n: 8192, k: 3840, isGated: false),
    ProjConfig(name: "Attn Out Projection", n: 3840, k: 4096, isGated: false),
    ProjConfig(name: "FFN Gated Linear", n: 15360, k: 3840, isGated: true),
    ProjConfig(name: "FFN Down Projection", n: 3840, k: 15360, isGated: false),
]

var totalLayerWeightBytes = 0
for p in projs {
    let wBytes = (p.n * (p.k / 32) * 18) * (p.isGated ? 2 : 1)
    totalLayerWeightBytes += wBytes
    print(String(format: "  %-22@: N=%5d K=%5d | Weight: %5.1f MB", p.name as NSString, p.n, p.k, Double(wBytes) / 1024 / 1024))
}
print(String(format: "Total 1-layer weights: %.1f MB (48 layers = %.2f GB)",
             Double(totalLayerWeightBytes) / 1024 / 1024,
             Double(totalLayerWeightBytes) * 48.0 / 1024 / 1024 / 1024))
print("-------------------------------------------------------------------------------")

// Measure baseline vs vectorized
for isVec in [false, true] {
    let modeName = isVec ? "Track C (Vectorized 4B x8)" : "Baseline (Legacy x8)"
    var layerTimes: [Double] = []
    
    // Warmup 3, measure 11
    for iter in 0..<14 {
        perturbCache()
        let cmd = queue.makeCommandBuffer()!
        
        for p in projs {
            let blocks = p.k / 32
            let weightBytes = (p.n * blocks * 18) * (p.isGated ? 2 : 1)
            let wBuf = buffer(weightBytes)
            let inBuf = buffer(p.k * 2)
            let outBuf = buffer(p.n * 2)
            let biasBuf = buffer(p.n * 2)
            let groups = (p.n + 63) / 64
            
            let enc = cmd.makeComputeCommandEncoder()!
            if p.isGated {
                enc.setComputePipelineState(isVec ? pipeGatedVec4 : pipeGatedBaseline)
                enc.setBuffer(inBuf, offset: 0, index: 0)
                enc.setBuffer(wBuf, offset: 0, index: 1)
                enc.setBuffer(wBuf, offset: weightBytes / 2, index: 2)
                enc.setBuffer(outBuf, offset: 0, index: 3)
                var params: [UInt32] = [1, UInt32(p.n), UInt32(p.k), 0]
                enc.setBytes(&params, length: 16, index: 4)
            } else {
                enc.setComputePipelineState(isVec ? pipeLinearVec4 : pipeLinearBaseline)
                enc.setBuffer(inBuf, offset: 0, index: 0)
                enc.setBuffer(wBuf, offset: 0, index: 1)
                enc.setBuffer(biasBuf, offset: 0, index: 2)
                enc.setBuffer(outBuf, offset: 0, index: 3)
                var params: [UInt32] = [1, UInt32(p.n), UInt32(p.k), 0]
                enc.setBytes(&params, length: 16, index: 4)
            }
            enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }
        
        finish(cmd)
        let sec = cmd.gpuEndTime - cmd.gpuStartTime
        if iter >= 3 {
            layerTimes.append(sec)
        }
    }
    
    layerTimes.sort()
    let medLayerSec = layerTimes[layerTimes.count / 2]
    let medLayerUs = medLayerSec * 1e6
    let med48LayersMs = medLayerSec * 48.0 * 1e3
    let effectiveGBps = Double(totalLayerWeightBytes) / medLayerSec / 1e9
    let rooflinePct = (effectiveGBps / 273.0) * 100.0
    
    print(String(format: "%-30@:", modeName as NSString))
    print(String(format: "  1 Layer GPU Time:        %7.2f µs (effective %6.2f GB/s, %4.1f%% roofline)",
                 medLayerUs, effectiveGBps, rooflinePct))
    print(String(format: "  48 Layers Total Projections: %7.2f ms (throughput: %5.2f tok/s cap)",
                 med48LayersMs, 1000.0 / med48LayersMs))
    print("-------------------------------------------------------------------------------")
}
