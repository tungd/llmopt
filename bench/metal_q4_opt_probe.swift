// Benchmark optimized Q4_0 decode kernel variants against current llmopt baseline
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

// -----------------------------------------------------------------------------
// Baseline x8 (from current llmopt_q4_0_decode_codegen.ml)
// -----------------------------------------------------------------------------
kernel void q4_0_baseline_x8(
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
  device const uchar* row_blocks1 = row_blocks0 + col_stride;
  device const uchar* row_blocks2 = row_blocks1 + col_stride;
  device const uchar* row_blocks3 = row_blocks2 + col_stride;
  device const uchar* row_blocks4 = row_blocks3 + col_stride;
  device const uchar* row_blocks5 = row_blocks4 + col_stride;
  device const uchar* row_blocks6 = row_blocks5 + col_stride;
  device const uchar* row_blocks7 = row_blocks6 + col_stride;
  device const half* activation = input;
  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
  float accumulator2 = 0.0f;
  float accumulator3 = 0.0f;
  float accumulator4 = 0.0f;
  float accumulator5 = 0.0f;
  float accumulator6 = 0.0f;
  float accumulator7 = 0.0f;
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

// -----------------------------------------------------------------------------
// Variant 1: Factored offset & scale
// Precompute a_sum = 8.0f * (a_low + a_high).
// -----------------------------------------------------------------------------
kernel void q4_0_factored_x8(
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
  device const uchar* row_blocks1 = row_blocks0 + col_stride;
  device const uchar* row_blocks2 = row_blocks1 + col_stride;
  device const uchar* row_blocks3 = row_blocks2 + col_stride;
  device const uchar* row_blocks4 = row_blocks3 + col_stride;
  device const uchar* row_blocks5 = row_blocks4 + col_stride;
  device const uchar* row_blocks6 = row_blocks5 + col_stride;
  device const uchar* row_blocks7 = row_blocks6 + col_stride;
  device const half* activation = input;
  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
  float accumulator2 = 0.0f;
  float accumulator3 = 0.0f;
  float accumulator4 = 0.0f;
  float accumulator5 = 0.0f;
  float accumulator6 = 0.0f;
  float accumulator7 = 0.0f;
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
    const float scale0 = float(*reinterpret_cast<device const half*>(blk0));
    const float scale1 = float(*reinterpret_cast<device const half*>(blk1));
    const float scale2 = float(*reinterpret_cast<device const half*>(blk2));
    const float scale3 = float(*reinterpret_cast<device const half*>(blk3));
    const float scale4 = float(*reinterpret_cast<device const half*>(blk4));
    const float scale5 = float(*reinterpret_cast<device const half*>(blk5));
    const float scale6 = float(*reinterpret_cast<device const half*>(blk6));
    const float scale7 = float(*reinterpret_cast<device const half*>(blk7));
    const uchar p0 = blk0[2u + l];
    const uchar p1 = blk1[2u + l];
    const uchar p2 = blk2[2u + l];
    const uchar p3 = blk3[2u + l];
    const uchar p4 = blk4[2u + l];
    const uchar p5 = blk5[2u + l];
    const uchar p6 = blk6[2u + l];
    const uchar p7 = blk7[2u + l];
    const uint k_idx = block_idx * 32u + l;
    const float a_low = float(activation[k_idx]);
    const float a_high = float(activation[k_idx + 16u]);
    const float a_sum8 = 8.0f * (a_low + a_high);

    accumulator0 += scale0 * (a_low * float(p0 & 15u) + a_high * float(p0 >> 4u) - a_sum8);
    accumulator1 += scale1 * (a_low * float(p1 & 15u) + a_high * float(p1 >> 4u) - a_sum8);
    accumulator2 += scale2 * (a_low * float(p2 & 15u) + a_high * float(p2 >> 4u) - a_sum8);
    accumulator3 += scale3 * (a_low * float(p3 & 15u) + a_high * float(p3 >> 4u) - a_sum8);
    accumulator4 += scale4 * (a_low * float(p4 & 15u) + a_high * float(p4 >> 4u) - a_sum8);
    accumulator5 += scale5 * (a_low * float(p5 & 15u) + a_high * float(p5 >> 4u) - a_sum8);
    accumulator6 += scale6 * (a_low * float(p6 & 15u) + a_high * float(p6 >> 4u) - a_sum8);
    accumulator7 += scale7 * (a_low * float(p7 & 15u) + a_high * float(p7 >> 4u) - a_sum8);
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

// -----------------------------------------------------------------------------
// Variant 2: Vectorized 32-bit load (uint2) where each thread processes 4 weights
// 8 threads per block (32 threads process 4 blocks = 128 K per iteration)
// -----------------------------------------------------------------------------
kernel void q4_0_vectorized_x8(
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
  device const uchar* row_blocks1 = row_blocks0 + col_stride;
  device const uchar* row_blocks2 = row_blocks1 + col_stride;
  device const uchar* row_blocks3 = row_blocks2 + col_stride;
  device const uchar* row_blocks4 = row_blocks3 + col_stride;
  device const uchar* row_blocks5 = row_blocks4 + col_stride;
  device const uchar* row_blocks6 = row_blocks5 + col_stride;
  device const uchar* row_blocks7 = row_blocks6 + col_stride;
  device const half* activation = input;

  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
  float accumulator2 = 0.0f;
  float accumulator3 = 0.0f;
  float accumulator4 = 0.0f;
  float accumulator5 = 0.0f;
  float accumulator6 = 0.0f;
  float accumulator7 = 0.0f;

  // 32 threads in simdgroup: 4 blocks per iteration (8 threads per block)
  // b_sub: 0..3 (which block within the 4-block cluster)
  // t_in_b: 0..7 (which pair of bytes within the 16 packed bytes)
  const uint b_sub = lane >> 3u; // 0..3
  const uint t_in_b = lane & 7u; // 0..7 (handles 2 bytes = 4 weights)
  const uint byte_offset = 2u + (t_in_b << 1u);

  const uint cluster_blocks = blocks >> 2u;
  for (uint c = 0u; c < cluster_blocks; ++c) {
    const uint block_idx = (c << 2u) + b_sub;
    const uint blk_byte_off = block_idx * 18u;

    device const uchar* blk0 = row_blocks0 + blk_byte_off;
    device const uchar* blk1 = row_blocks1 + blk_byte_off;
    device const uchar* blk2 = row_blocks2 + blk_byte_off;
    device const uchar* blk3 = row_blocks3 + blk_byte_off;
    device const uchar* blk4 = row_blocks4 + blk_byte_off;
    device const uchar* blk5 = row_blocks5 + blk_byte_off;
    device const uchar* blk6 = row_blocks6 + blk_byte_off;
    device const uchar* blk7 = row_blocks7 + blk_byte_off;

    const float scale0 = float(*reinterpret_cast<device const half*>(blk0));
    const float scale1 = float(*reinterpret_cast<device const half*>(blk1));
    const float scale2 = float(*reinterpret_cast<device const half*>(blk2));
    const float scale3 = float(*reinterpret_cast<device const half*>(blk3));
    const float scale4 = float(*reinterpret_cast<device const half*>(blk4));
    const float scale5 = float(*reinterpret_cast<device const half*>(blk5));
    const float scale6 = float(*reinterpret_cast<device const half*>(blk6));
    const float scale7 = float(*reinterpret_cast<device const half*>(blk7));

    // Load 2 bytes (16 bits) containing 4 nibbles
    const ushort p0 = *reinterpret_cast<device const ushort*>(blk0 + byte_offset);
    const ushort p1 = *reinterpret_cast<device const ushort*>(blk1 + byte_offset);
    const ushort p2 = *reinterpret_cast<device const ushort*>(blk2 + byte_offset);
    const ushort p3 = *reinterpret_cast<device const ushort*>(blk3 + byte_offset);
    const ushort p4 = *reinterpret_cast<device const ushort*>(blk4 + byte_offset);
    const ushort p5 = *reinterpret_cast<device const ushort*>(blk5 + byte_offset);
    const ushort p6 = *reinterpret_cast<device const ushort*>(blk6 + byte_offset);
    const ushort p7 = *reinterpret_cast<device const ushort*>(blk7 + byte_offset);

    // Activations for byte 0: l0 = (t_in_b << 1)
    const uint l0 = t_in_b << 1u;
    const uint k_idx0 = block_idx * 32u + l0;
    const float a0_lo = float(activation[k_idx0]);
    const float a0_hi = float(activation[k_idx0 + 16u]);

    // Activations for byte 1: l1 = (t_in_b << 1) + 1
    const uint k_idx1 = k_idx0 + 1u;
    const float a1_lo = float(activation[k_idx1]);
    const float a1_hi = float(activation[k_idx1 + 16u]);

    const float a_sum8 = 8.0f * (a0_lo + a0_hi + a1_lo + a1_hi);

    // Extract byte 0 nibbles and byte 1 nibbles
    // p is 16-bit: byte 0 is p & 0xFF, byte 1 is p >> 8
    #define DOT_PAIR(p) \\
        (a0_lo * float(p & 15u) + \\
         a0_hi * float((p >> 4u) & 15u) + \\
         a1_lo * float((p >> 8u) & 15u) + \\
         a1_hi * float((p >> 12u) & 15u) - a_sum8)

    accumulator0 += scale0 * DOT_PAIR(p0);
    accumulator1 += scale1 * DOT_PAIR(p1);
    accumulator2 += scale2 * DOT_PAIR(p2);
    accumulator3 += scale3 * DOT_PAIR(p3);
    accumulator4 += scale4 * DOT_PAIR(p4);
    accumulator5 += scale5 * DOT_PAIR(p5);
    accumulator6 += scale6 * DOT_PAIR(p6);
    accumulator7 += scale7 * DOT_PAIR(p7);
    #undef DOT_PAIR
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

// -----------------------------------------------------------------------------
// Variant 3: Vectorized 32-bit load (uint = 4 bytes = 8 weights per thread)
// 4 threads per block (32 threads process 8 blocks = 256 K per iteration)
// -----------------------------------------------------------------------------
kernel void q4_0_vectorized_4b_x8(
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
  device const uchar* row_blocks1 = row_blocks0 + col_stride;
  device const uchar* row_blocks2 = row_blocks1 + col_stride;
  device const uchar* row_blocks3 = row_blocks2 + col_stride;
  device const uchar* row_blocks4 = row_blocks3 + col_stride;
  device const uchar* row_blocks5 = row_blocks4 + col_stride;
  device const uchar* row_blocks6 = row_blocks5 + col_stride;
  device const uchar* row_blocks7 = row_blocks6 + col_stride;
  device const half* activation = input;

  float accumulator0 = 0.0f, accumulator1 = 0.0f, accumulator2 = 0.0f, accumulator3 = 0.0f;
  float accumulator4 = 0.0f, accumulator5 = 0.0f, accumulator6 = 0.0f, accumulator7 = 0.0f;

  const uint b_sub = lane >> 2u; // 0..7
  const uint t_in_b = lane & 3u; // 0..3 (4 bytes = 8 weights)
  const uint byte_offset = 2u + (t_in_b << 2u);

  const uint cluster_blocks = blocks >> 3u;
  for (uint c = 0u; c < cluster_blocks; ++c) {
    const uint block_idx = (c << 3u) + b_sub;
    const uint blk_byte_off = block_idx * 18u;

    device const uchar* blk0 = row_blocks0 + blk_byte_off;
    device const uchar* blk1 = row_blocks1 + blk_byte_off;
    device const uchar* blk2 = row_blocks2 + blk_byte_off;
    device const uchar* blk3 = row_blocks3 + blk_byte_off;
    device const uchar* blk4 = row_blocks4 + blk_byte_off;
    device const uchar* blk5 = row_blocks5 + blk_byte_off;
    device const uchar* blk6 = row_blocks6 + blk_byte_off;
    device const uchar* blk7 = row_blocks7 + blk_byte_off;

    const float scale0 = float(*reinterpret_cast<device const half*>(blk0));
    const float scale1 = float(*reinterpret_cast<device const half*>(blk1));
    const float scale2 = float(*reinterpret_cast<device const half*>(blk2));
    const float scale3 = float(*reinterpret_cast<device const half*>(blk3));
    const float scale4 = float(*reinterpret_cast<device const half*>(blk4));
    const float scale5 = float(*reinterpret_cast<device const half*>(blk5));
    const float scale6 = float(*reinterpret_cast<device const half*>(blk6));
    const float scale7 = float(*reinterpret_cast<device const half*>(blk7));

    const ushort p0_lo = *reinterpret_cast<device const ushort*>(blk0 + byte_offset);
    const ushort p0_hi = *reinterpret_cast<device const ushort*>(blk0 + byte_offset + 2u);
    const ushort p1_lo = *reinterpret_cast<device const ushort*>(blk1 + byte_offset);
    const ushort p1_hi = *reinterpret_cast<device const ushort*>(blk1 + byte_offset + 2u);
    const ushort p2_lo = *reinterpret_cast<device const ushort*>(blk2 + byte_offset);
    const ushort p2_hi = *reinterpret_cast<device const ushort*>(blk2 + byte_offset + 2u);
    const ushort p3_lo = *reinterpret_cast<device const ushort*>(blk3 + byte_offset);
    const ushort p3_hi = *reinterpret_cast<device const ushort*>(blk3 + byte_offset + 2u);
    const ushort p4_lo = *reinterpret_cast<device const ushort*>(blk4 + byte_offset);
    const ushort p4_hi = *reinterpret_cast<device const ushort*>(blk4 + byte_offset + 2u);
    const ushort p5_lo = *reinterpret_cast<device const ushort*>(blk5 + byte_offset);
    const ushort p5_hi = *reinterpret_cast<device const ushort*>(blk5 + byte_offset + 2u);
    const ushort p6_lo = *reinterpret_cast<device const ushort*>(blk6 + byte_offset);
    const ushort p6_hi = *reinterpret_cast<device const ushort*>(blk6 + byte_offset + 2u);
    const ushort p7_lo = *reinterpret_cast<device const ushort*>(blk7 + byte_offset);
    const ushort p7_hi = *reinterpret_cast<device const ushort*>(blk7 + byte_offset + 2u);

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

    #define DOT_QUAD(plo, phi) \\
        (a0_lo * float(plo & 15u) + \\
         a0_hi * float((plo >> 4u) & 15u) + \\
         a1_lo * float((plo >> 8u) & 15u) + \\
         a1_hi * float((plo >> 12u) & 15u) + \\
         a2_lo * float(phi & 15u) + \\
         a2_hi * float((phi >> 4u) & 15u) + \\
         a3_lo * float((phi >> 8u) & 15u) + \\
         a3_hi * float((phi >> 12u) & 15u) - a_sum8)

    accumulator0 += scale0 * DOT_QUAD(p0_lo, p0_hi);
    accumulator1 += scale1 * DOT_QUAD(p1_lo, p1_hi);
    accumulator2 += scale2 * DOT_QUAD(p2_lo, p2_hi);
    accumulator3 += scale3 * DOT_QUAD(p3_lo, p3_hi);
    accumulator4 += scale4 * DOT_QUAD(p4_lo, p4_hi);
    accumulator5 += scale5 * DOT_QUAD(p5_lo, p5_hi);
    accumulator6 += scale6 * DOT_QUAD(p6_lo, p6_hi);
    accumulator7 += scale7 * DOT_QUAD(p7_lo, p7_hi);
    #undef DOT_QUAD
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

// -----------------------------------------------------------------------------
// Variant 4: Vectorized 32-bit load with x16 columns per SIMD group
// 4 threads per block (32 threads process 8 blocks = 256 K per iteration)
// 16 columns per SIMD group (2x column tile width)
// -----------------------------------------------------------------------------
kernel void q4_0_vectorized_4b_x16(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint base_col = simd_idx << 4u;
  if (base_col >= params.n) return;
  const uint blocks = params.k / 32u;
  const uint col_stride = blocks * 18u;

  device const uchar* rows[16];
  for (uint i = 0u; i < 16u; ++i) {
    rows[i] = weight + (base_col + i) * col_stride;
  }
  device const half* activation = input;

  float acc[16] = {0.0f};

  const uint b_sub = lane >> 2u; // 0..7
  const uint t_in_b = lane & 3u; // 0..3 (4 bytes = 8 weights)
  const uint byte_offset = 2u + (t_in_b << 2u);

  const uint cluster_blocks = blocks >> 3u;
  for (uint c = 0u; c < cluster_blocks; ++c) {
    const uint block_idx = (c << 3u) + b_sub;
    const uint blk_byte_off = block_idx * 18u;

    float scales[16];
    ushort p_lo[16];
    ushort p_hi[16];
    for (uint i = 0u; i < 16u; ++i) {
      device const uchar* blk = rows[i] + blk_byte_off;
      scales[i] = float(*reinterpret_cast<device const half*>(blk));
      p_lo[i] = *reinterpret_cast<device const ushort*>(blk + byte_offset);
      p_hi[i] = *reinterpret_cast<device const ushort*>(blk + byte_offset + 2u);
    }

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

    #define DOT_QUAD(plo, phi) \\
        (a0_lo * float(plo & 15u) + \\
         a0_hi * float((plo >> 4u) & 15u) + \\
         a1_lo * float((plo >> 8u) & 15u) + \\
         a1_hi * float((plo >> 12u) & 15u) + \\
         a2_lo * float(phi & 15u) + \\
         a2_hi * float((phi >> 4u) & 15u) + \\
         a3_lo * float((phi >> 8u) & 15u) + \\
         a3_hi * float((phi >> 12u) & 15u) - a_sum8)

    for (uint i = 0u; i < 16u; ++i) {
      acc[i] += scales[i] * DOT_QUAD(p_lo[i], p_hi[i]);
    }
    #undef DOT_QUAD
  }

  for (uint i = 0u; i < 16u; ++i) {
    acc[i] = simd_sum(acc[i]);
  }

  if (lane == 0u) {
    for (uint i = 0u; i < 16u; ++i) {
      if (base_col + i < params.n) {
        output[base_col + i] = half(acc[i]);
      }
    }
  }
}
""";

let library = try device.makeLibrary(source: mslSource, options: nil)
let kernelNames = ["q4_0_baseline_x8", "q4_0_factored_x8", "q4_0_vectorized_x8", "q4_0_vectorized_4b_x8", "q4_0_vectorized_4b_x16"]
let pipelines = try kernelNames.map { name -> MTLComputePipelineState in
    guard let fn = library.makeFunction(name: name) else { fatalError("missing \(name)") }
    return try device.makeComputePipelineState(function: fn)
}

func buffer(_ length: Int) -> MTLBuffer {
    device.makeBuffer(length: length, options: .storageModeShared)!
}
func finish(_ command: MTLCommandBuffer) {
    command.commit()
    command.waitUntilCompleted()
    require(command.status == .completed, "GPU command failed: \\(String(describing: command.error))")
}

let eviction = buffer(256 * 1024 * 1024)
func perturbCache() {
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeBlitCommandEncoder()!
    encoder.fill(buffer: eviction, range: 0..<eviction.length, value: 0xa5)
    encoder.endEncoding()
    finish(command)
}

print("device=\(device.name)")
print("Benchmarking Q4_0 kernel variants across Gemma shapes (m=1, threads_per_group=256, samples=11)")
print("-----------------------------------------------------------------------------------------------")

for (n, k) in [(3840, 4096), (15360, 3840), (3840, 15360)] {
    autoreleasepool {
        let blocks = k / 32
        let weightBytes = n * blocks * 18
        let weight = buffer(weightBytes)
        let input = buffer(k * 2)
        let bias = buffer(n * 2)
        let output = buffer(n * 2)
        let xp = input.contents().bindMemory(to: Float16.self, capacity: k)
        let wp = weight.contents().bindMemory(to: UInt8.self, capacity: weightBytes)
        var x = [Int32](repeating: 0, count: k)
        var rng: UInt32 = 0x12345678
        func random() -> UInt32 {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng
        }
        for i in 0..<k {
            x[i] = Int32(random() % 17) - 8
            xp[i] = Float16(Float(x[i]) / 16)
        }
        for b in 0..<(n * blocks) {
            wp[b * 18] = 0
            wp[b * 18 + 1] = 0x28 // scale 1/32
            for j in 2..<18 { wp[b * 18 + j] = UInt8(truncatingIfNeeded: random()) }
        }
        var reference = [UInt16](repeating: 0, count: n)
        for col in 0..<n {
            var sum: Int32 = 0
            for b in 0..<blocks {
                let offset = (col * blocks + b) * 18 + 2
                for lane in 0..<16 {
                    let packed = wp[offset + lane]
                    sum += x[b * 32 + lane] * (Int32(packed & 15) - 8)
                    sum += x[b * 32 + lane + 16] * (Int32(packed >> 4) - 8)
                }
            }
            reference[col] = Float16(Float(sum) / 512).bitPattern
        }

        var results: [(name: String, median_us: Double, gbps: Double)] = []

        for (kIdx, kName) in kernelNames.enumerated() {
            let colTile = kName.hasSuffix("_x16") ? 16 : 8
            let groups = (n + colTile * 8 - 1) / (colTile * 8)
            var samples: [Double] = []
            let op = output.contents().bindMemory(to: UInt16.self, capacity: n)

            for iter in 0..<14 {
                perturbCache()
                for col in 0..<n { op[col] = 0x7e00 }
                let cmd = queue.makeCommandBuffer()!
                let enc = cmd.makeComputeCommandEncoder()!
                enc.setComputePipelineState(pipelines[kIdx])
                enc.setBuffer(input, offset: 0, index: 0)
                enc.setBuffer(weight, offset: 0, index: 1)
                enc.setBuffer(bias, offset: 0, index: 2)
                enc.setBuffer(output, offset: 0, index: 3)
                let params: [UInt32] = [1, UInt32(n), UInt32(k), 0]
                params.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 4) }
                enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()
                finish(cmd)

                // Verify bit-exact numerical parity against CPU reference
                for col in 0..<n {
                    require(op[col] == reference[col], "Parity mismatch in \\(kName) col \\(col): \\(op[col]) vs ref \\(reference[col])")
                }
                let sec = cmd.gpuEndTime - cmd.gpuStartTime
                if iter >= 3 { samples.append(sec) }
            }
            samples.sort()
            let med = samples[samples.count / 2]
            let gbps = Double(weightBytes) / med / 1e9
            results.append((name: kName, median_us: med * 1e6, gbps: gbps))
        }

        print(String(format: "Shape N=%5d K=%5d (Weight: %.1f MB):", n, k, Double(weightBytes) / 1024 / 1024))
        for r in results {
            print(String(format: "  %-22@: %7.2f µs | %6.2f GB/s (vs baseline: %+.1f%%)",
                         r.name as NSString, r.median_us, r.gbps,
                         ((results[0].median_us / r.median_us) - 1.0) * 100))
        }
    }
}
