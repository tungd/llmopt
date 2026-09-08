(** Metal Q6_K dequant shader text.  Split out of metal.ml (LOOP item 5). *)

let q6_k_source = {|
struct block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};

kernel void llmopt_dequant_q6_k(
    device const block_q6_K* blocks [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant QuantBlock256Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint sb_idx = gid >> 5;
  if (sb_idx >= params.total_superblocks) return;
  device const block_q6_K& b = blocks[sb_idx];
  const float d = float(b.d);
  device const uint8_t* ql = b.ql;
  device const uint8_t* qh = b.qh;
  device const int8_t* sc = b.scales;
  const int is = lane / 16;
  const int q1 = int(extract_bits(ql[lane], 0u, 4u) | (extract_bits(qh[lane], 0u, 2u) << 4)) - 32;
  const int q2 = int(extract_bits(ql[32 + lane], 0u, 4u) | (extract_bits(qh[lane], 2u, 2u) << 4)) - 32;
  const int q3 = int(extract_bits(ql[lane], 4u, 4u) | (extract_bits(qh[lane], 4u, 2u) << 4)) - 32;
  const int q4 = int(extract_bits(ql[32 + lane], 4u, 4u) | (extract_bits(qh[lane], 6u, 2u) << 4)) - 32;
  output[sb_idx * 256 + 0 + lane]  = half(d * float(sc[is + 0]) * float(q1));
  output[sb_idx * 256 + 32 + lane] = half(d * float(sc[is + 2]) * float(q2));
  output[sb_idx * 256 + 64 + lane] = half(d * float(sc[is + 4]) * float(q3));
  output[sb_idx * 256 + 96 + lane] = half(d * float(sc[is + 6]) * float(q4));

  const int q5 = int(extract_bits(ql[64 + lane], 0u, 4u) | (extract_bits(qh[32 + lane], 0u, 2u) << 4)) - 32;
  const int q6 = int(extract_bits(ql[96 + lane], 0u, 4u) | (extract_bits(qh[32 + lane], 2u, 2u) << 4)) - 32;
  const int q7 = int(extract_bits(ql[64 + lane], 4u, 4u) | (extract_bits(qh[32 + lane], 4u, 2u) << 4)) - 32;
  const int q8 = int(extract_bits(ql[96 + lane], 4u, 4u) | (extract_bits(qh[32 + lane], 6u, 2u) << 4)) - 32;
  output[sb_idx * 256 + 128 + lane] = half(d * float(sc[8 + is + 0]) * float(q5));
  output[sb_idx * 256 + 160 + lane] = half(d * float(sc[8 + is + 2]) * float(q6));
  output[sb_idx * 256 + 192 + lane] = half(d * float(sc[8 + is + 4]) * float(q7));
  output[sb_idx * 256 + 224 + lane] = half(d * float(sc[8 + is + 6]) * float(q8));
}


kernel void llmopt_q6_k_linear_f16_splitk(
    device const half* input [[buffer(0)]],
    device const block_q6_K* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_in_tg = tid >> 5;
  const uint col_in_tg = simd_in_tg >> 2;
  const uint split_idx = simd_in_tg & 3u;
  const uint tg_x = gid / 256u;
  const uint col = (tg_x << 1) + col_in_tg;
  if (col >= params.n) return;

  const uint num_superblocks = params.k >> 8;
  device const block_q6_K* row_blocks = weight + col * num_superblocks;
  device const half* in_ptr = input;

  threadgroup float partial_sums[2][4];

  float acc = 0.0f;
  for (uint sb = split_idx; sb < num_superblocks; sb += 4u) {
    device const block_q6_K& b = row_blocks[sb];
    const float d = float(b.d);
    device const uint8_t* ql = b.ql;
    device const uint8_t* qh = b.qh;
    device const int8_t* sc = b.scales;
    const int is = lane / 16;
    const int q1 = int(extract_bits(ql[lane], 0u, 4u) | (extract_bits(qh[lane], 0u, 2u) << 4)) - 32;
    const int q2 = int(extract_bits(ql[32 + lane], 0u, 4u) | (extract_bits(qh[lane], 2u, 2u) << 4)) - 32;
    const int q3 = int(extract_bits(ql[lane], 4u, 4u) | (extract_bits(qh[lane], 4u, 2u) << 4)) - 32;
    const int q4 = int(extract_bits(ql[32 + lane], 4u, 4u) | (extract_bits(qh[lane], 6u, 2u) << 4)) - 32;
    const uint base_idx = sb * 256 + lane;
    acc += float(in_ptr[base_idx + 0])  * (d * float(sc[is + 0]) * float(q1));
    acc += float(in_ptr[base_idx + 32]) * (d * float(sc[is + 2]) * float(q2));
    acc += float(in_ptr[base_idx + 64]) * (d * float(sc[is + 4]) * float(q3));
    acc += float(in_ptr[base_idx + 96]) * (d * float(sc[is + 6]) * float(q4));

    const int q5 = int(extract_bits(ql[64 + lane], 0u, 4u) | (extract_bits(qh[32 + lane], 0u, 2u) << 4)) - 32;
    const int q6 = int(extract_bits(ql[96 + lane], 0u, 4u) | (extract_bits(qh[32 + lane], 2u, 2u) << 4)) - 32;
    const int q7 = int(extract_bits(ql[64 + lane], 4u, 4u) | (extract_bits(qh[32 + lane], 4u, 2u) << 4)) - 32;
    const int q8 = int(extract_bits(ql[96 + lane], 4u, 4u) | (extract_bits(qh[32 + lane], 6u, 2u) << 4)) - 32;
    acc += float(in_ptr[base_idx + 128]) * (d * float(sc[8 + is + 0]) * float(q5));
    acc += float(in_ptr[base_idx + 160]) * (d * float(sc[8 + is + 2]) * float(q6));
    acc += float(in_ptr[base_idx + 192]) * (d * float(sc[8 + is + 4]) * float(q7));
    acc += float(in_ptr[base_idx + 224]) * (d * float(sc[8 + is + 6]) * float(q8));
  }
  acc = simd_sum(acc);
  if (lane == 0) {
    partial_sums[col_in_tg][split_idx] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (split_idx == 0 && lane == 0) {
    float total = partial_sums[col_in_tg][0] + partial_sums[col_in_tg][1] + partial_sums[col_in_tg][2] + partial_sums[col_in_tg][3];
    if (params.has_bias != 0u) total += float(bias[col]);
    output[col] = half(total);
  }
}

kernel void llmopt_q6_k_linear_f16_splitk_m2(
    device const half* input [[buffer(0)]],
    device const block_q6_K* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_in_tg = tid >> 5;
  const uint col_in_tg = simd_in_tg >> 2;
  const uint split_idx = simd_in_tg & 3u;
  const uint tg_x = gid / 256u;
  const uint col = (tg_x << 1) + col_in_tg;
  if (col >= params.n) return;

  const uint num_superblocks = params.k >> 8;
  device const block_q6_K* row_blocks = weight + col * num_superblocks;
  device const half* in0 = input;
  device const half* in1 = input + params.k;

  threadgroup float partial_sums0[2][4];
  threadgroup float partial_sums1[2][4];

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (uint sb = split_idx; sb < num_superblocks; sb += 4u) {
    device const block_q6_K& b = row_blocks[sb];
    const float d = float(b.d);
    device const uint8_t* ql = b.ql;
    device const uint8_t* qh = b.qh;
    device const int8_t* sc = b.scales;
    const int is = lane / 16;
    const uint base_idx = sb * 256 + lane;
    const float w0 = d * float(sc[is + 0]) * float(int(extract_bits(ql[lane], 0u, 4u) | (extract_bits(qh[lane], 0u, 2u) << 4)) - 32);
    const float w1 = d * float(sc[is + 2]) * float(int(extract_bits(ql[32 + lane], 0u, 4u) | (extract_bits(qh[lane], 2u, 2u) << 4)) - 32);
    const float w2 = d * float(sc[is + 4]) * float(int(extract_bits(ql[lane], 4u, 4u) | (extract_bits(qh[lane], 4u, 2u) << 4)) - 32);
    const float w3 = d * float(sc[is + 6]) * float(int(extract_bits(ql[32 + lane], 4u, 4u) | (extract_bits(qh[lane], 6u, 2u) << 4)) - 32);
    const float w4 = d * float(sc[8 + is + 0]) * float(int(extract_bits(ql[64 + lane], 0u, 4u) | (extract_bits(qh[32 + lane], 0u, 2u) << 4)) - 32);
    const float w5 = d * float(sc[8 + is + 2]) * float(int(extract_bits(ql[96 + lane], 0u, 4u) | (extract_bits(qh[32 + lane], 2u, 2u) << 4)) - 32);
    const float w6 = d * float(sc[8 + is + 4]) * float(int(extract_bits(ql[64 + lane], 4u, 4u) | (extract_bits(qh[32 + lane], 4u, 2u) << 4)) - 32);
    const float w7 = d * float(sc[8 + is + 6]) * float(int(extract_bits(ql[96 + lane], 4u, 4u) | (extract_bits(qh[32 + lane], 6u, 2u) << 4)) - 32);
    acc0 += float(in0[base_idx + 0]) * w0;
    acc0 += float(in0[base_idx + 32]) * w1;
    acc0 += float(in0[base_idx + 64]) * w2;
    acc0 += float(in0[base_idx + 96]) * w3;
    acc0 += float(in0[base_idx + 128]) * w4;
    acc0 += float(in0[base_idx + 160]) * w5;
    acc0 += float(in0[base_idx + 192]) * w6;
    acc0 += float(in0[base_idx + 224]) * w7;
    acc1 += float(in1[base_idx + 0]) * w0;
    acc1 += float(in1[base_idx + 32]) * w1;
    acc1 += float(in1[base_idx + 64]) * w2;
    acc1 += float(in1[base_idx + 96]) * w3;
    acc1 += float(in1[base_idx + 128]) * w4;
    acc1 += float(in1[base_idx + 160]) * w5;
    acc1 += float(in1[base_idx + 192]) * w6;
    acc1 += float(in1[base_idx + 224]) * w7;
  }
  acc0 = simd_sum(acc0);
  acc1 = simd_sum(acc1);
  if (lane == 0) {
    partial_sums0[col_in_tg][split_idx] = acc0;
    partial_sums1[col_in_tg][split_idx] = acc1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (split_idx == 0 && lane == 0) {
    float total0 = partial_sums0[col_in_tg][0] + partial_sums0[col_in_tg][1] + partial_sums0[col_in_tg][2] + partial_sums0[col_in_tg][3];
    float total1 = partial_sums1[col_in_tg][0] + partial_sums1[col_in_tg][1] + partial_sums1[col_in_tg][2] + partial_sums1[col_in_tg][3];
    const float bias_value = params.has_bias != 0u ? float(bias[col]) : 0.0f;
    output[col] = half(total0 + bias_value);
    output[params.n + col] = half(total1 + bias_value);
  }
}


kernel void llmopt_q6_k_linear_f16_llama_mv_add(
    device const half* input [[buffer(0)]],
    device const block_q6_K* weight [[buffer(1)]],
    device const half* residual [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row_base =
      (threadgroup_position.x * 2u + simdgroup) * 2u;
  const uint block_parity = lane & 1u;
  const uint lane_group = lane >> 1u;
  const uint half_index = lane_group >> 3u;
  const uint lane_in_half = lane_group & 7u;
  const uint lane_offset = lane_in_half << 2u;
  const uint scale_offset = (half_index << 3u) + (lane_offset >> 4u);
  const uint activation_offset = (half_index << 7u) + lane_offset;
  const uint low_offset = (half_index << 6u) + lane_offset;
  const uint high_offset = (half_index << 5u) + lane_offset;
  const uint num_superblocks = params.k >> 8u;
  float sums[2] = { 0.0f, 0.0f };

  for (uint sb = block_parity; sb < num_superblocks; sb += 2u) {
    device const half* activation = input + (sb << 8u) + activation_offset;
    float values[16];
    for (uint index = 0u; index < 4u; ++index) {
      values[4u * index + 0u] = float(activation[index + 0u]);
      values[4u * index + 1u] = float(activation[index + 32u]);
      values[4u * index + 2u] = float(activation[index + 64u]);
      values[4u * index + 3u] = float(activation[index + 96u]);
    }
    for (uint row = 0u; row < 2u; ++row) {
      const uint output_row = row_base + row;
      if (output_row >= params.n) continue;
      device const block_q6_K& block =
          weight[output_row * num_superblocks + sb];
      device const uchar* low0 = block.ql + low_offset;
      device const uchar* low1 = low0 + 32u;
      device const uchar* high = block.qh + high_offset;
      device const int8_t* scales = block.scales + scale_offset;
      float4 dot = float4(0.0f);
      for (uint index = 0u; index < 4u; ++index) {
        dot[0] += values[4u * index + 0u]
            * float(int((low0[index] & 0x0fu)
                | ((high[index] & 0x03u) << 4u)) - 32);
        dot[1] += values[4u * index + 1u]
            * float(int((low1[index] & 0x0fu)
                | ((high[index] & 0x0cu) << 2u)) - 32);
        dot[2] += values[4u * index + 2u]
            * float(int((low0[index] >> 4u)
                | (high[index] & 0x30u)) - 32);
        dot[3] += values[4u * index + 3u]
            * float(int((low1[index] >> 4u)
                | ((high[index] & 0xc0u) >> 2u)) - 32);
      }
      sums[row] += float(block.d)
          * (dot[0] * float(scales[0]) + dot[1] * float(scales[2])
            + dot[2] * float(scales[4]) + dot[3] * float(scales[6]));
    }
  }

  for (uint row = 0u; row < 2u; ++row) {
    const float total = simd_sum(sums[row]);
    const uint output_row = row_base + row;
    if (lane == 0u && output_row < params.n)
      output[output_row] = half(half(total) + residual[output_row]);
  }
}
|}
