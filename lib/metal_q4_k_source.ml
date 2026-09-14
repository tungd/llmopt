(** Metal Q4_K dequant shader text.  Split out of metal.ml (LOOP item 5). *)

let q4_k_source = {|
struct block_q4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

kernel void llmopt_dequant_q4_k(
    device const block_q4_K* blocks [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant QuantBlock256Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint sb_idx = gid >> 5;
  if (sb_idx >= params.total_superblocks) return;
  device const block_q4_K& b = blocks[sb_idx];
  const float d = float(b.d);
  const float dmin = float(b.dmin);
  float dl[8];
  float ml[8];
  for (int j = 0; j < 4; ++j) {
    const uint8_t sc0 = extract_bits(b.scales[j], 0u, 6u);
    const uint8_t m0 = extract_bits(b.scales[j + 4], 0u, 6u);
    const uint8_t sc1 = extract_bits(b.scales[j + 8], 0u, 4u) | (extract_bits(b.scales[j], 6u, 2u) << 4);
    const uint8_t m1 = extract_bits(b.scales[j + 8], 4u, 4u) | (extract_bits(b.scales[j + 4], 6u, 2u) << 4);
    dl[j] = d * float(sc0);
    ml[j] = dmin * float(m0);
    dl[j + 4] = d * float(sc1);
    ml[j + 4] = dmin * float(m1);
  }
  const uint8_t q0 = b.qs[lane];
  const uint8_t q1 = b.qs[32 + lane];
  const uint8_t q2 = b.qs[64 + lane];
  const uint8_t q3 = b.qs[96 + lane];
  output[sb_idx * 256 + 0 + lane]   = half(dl[0] * float(extract_bits(q0, 0u, 4u)) - ml[0]);
  output[sb_idx * 256 + 32 + lane]  = half(dl[1] * float(extract_bits(q0, 4u, 4u)) - ml[1]);
  output[sb_idx * 256 + 64 + lane]  = half(dl[2] * float(extract_bits(q1, 0u, 4u)) - ml[2]);
  output[sb_idx * 256 + 96 + lane]  = half(dl[3] * float(extract_bits(q1, 4u, 4u)) - ml[3]);
  output[sb_idx * 256 + 128 + lane] = half(dl[4] * float(extract_bits(q2, 0u, 4u)) - ml[4]);
  output[sb_idx * 256 + 160 + lane] = half(dl[5] * float(extract_bits(q2, 4u, 4u)) - ml[5]);
  output[sb_idx * 256 + 192 + lane] = half(dl[6] * float(extract_bits(q3, 0u, 4u)) - ml[6]);
  output[sb_idx * 256 + 224 + lane] = half(dl[7] * float(extract_bits(q3, 4u, 4u)) - ml[7]);
}


kernel void llmopt_q4_k_linear_f16_splitk(
    device const half* input [[buffer(0)]],
    device const block_q4_K* weight [[buffer(1)]],
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
  device const block_q4_K* row_blocks = weight + col * num_superblocks;
  device const half* in_ptr = input;

  threadgroup float partial_sums[2][4];

  float acc = 0.0f;
  for (uint sb = split_idx; sb < num_superblocks; sb += 4u) {
    device const block_q4_K& b = row_blocks[sb];
    const float d = float(b.d);
    const float dmin = float(b.dmin);
    float dl[8];
    float ml[8];
    for (int j = 0; j < 4; ++j) {
      const uint8_t sc0 = extract_bits(b.scales[j], 0u, 6u);
      const uint8_t m0 = extract_bits(b.scales[j + 4], 0u, 6u);
      const uint8_t sc1 = extract_bits(b.scales[j + 8], 0u, 4u) | (extract_bits(b.scales[j], 6u, 2u) << 4);
      const uint8_t m1 = extract_bits(b.scales[j + 8], 4u, 4u) | (extract_bits(b.scales[j + 4], 6u, 2u) << 4);
      dl[j] = d * float(sc0);
      ml[j] = dmin * float(m0);
      dl[j + 4] = d * float(sc1);
      ml[j + 4] = dmin * float(m1);
    }
    const uint8_t q0 = b.qs[lane];
    const uint8_t q1 = b.qs[32 + lane];
    const uint8_t q2 = b.qs[64 + lane];
    const uint8_t q3 = b.qs[96 + lane];
    const uint base_idx = sb * 256 + lane;
    acc += float(in_ptr[base_idx + 0])   * (dl[0] * float(extract_bits(q0, 0u, 4u)) - ml[0]);
    acc += float(in_ptr[base_idx + 32])  * (dl[1] * float(extract_bits(q0, 4u, 4u)) - ml[1]);
    acc += float(in_ptr[base_idx + 64])  * (dl[2] * float(extract_bits(q1, 0u, 4u)) - ml[2]);
    acc += float(in_ptr[base_idx + 96])  * (dl[3] * float(extract_bits(q1, 4u, 4u)) - ml[3]);
    acc += float(in_ptr[base_idx + 128]) * (dl[4] * float(extract_bits(q2, 0u, 4u)) - ml[4]);
    acc += float(in_ptr[base_idx + 160]) * (dl[5] * float(extract_bits(q2, 4u, 4u)) - ml[5]);
    acc += float(in_ptr[base_idx + 192]) * (dl[6] * float(extract_bits(q3, 0u, 4u)) - ml[6]);
    acc += float(in_ptr[base_idx + 224]) * (dl[7] * float(extract_bits(q3, 4u, 4u)) - ml[7]);
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

kernel void llmopt_q4_k_linear_f16_splitk_m2(
    device const half* input [[buffer(0)]],
    device const block_q4_K* weight [[buffer(1)]],
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
  device const block_q4_K* row_blocks = weight + col * num_superblocks;
  device const half* in0 = input;
  device const half* in1 = input + params.k;

  threadgroup float partial_sums0[2][4];
  threadgroup float partial_sums1[2][4];

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (uint sb = split_idx; sb < num_superblocks; sb += 4u) {
    device const block_q4_K& b = row_blocks[sb];
    const float d = float(b.d);
    const float dmin = float(b.dmin);
    float dl[8];
    float ml[8];
    for (int j = 0; j < 4; ++j) {
      const uint8_t sc0 = extract_bits(b.scales[j], 0u, 6u);
      const uint8_t m0 = extract_bits(b.scales[j + 4], 0u, 6u);
      const uint8_t sc1 = extract_bits(b.scales[j + 8], 0u, 4u) | (extract_bits(b.scales[j], 6u, 2u) << 4);
      const uint8_t m1 = extract_bits(b.scales[j + 8], 4u, 4u) | (extract_bits(b.scales[j + 4], 6u, 2u) << 4);
      dl[j] = d * float(sc0);
      ml[j] = dmin * float(m0);
      dl[j + 4] = d * float(sc1);
      ml[j + 4] = dmin * float(m1);
    }
    const uint8_t q0 = b.qs[lane];
    const uint8_t q1 = b.qs[32 + lane];
    const uint8_t q2 = b.qs[64 + lane];
    const uint8_t q3 = b.qs[96 + lane];
    const uint base_idx = sb * 256 + lane;
    const float w0 = dl[0] * float(extract_bits(q0, 0u, 4u)) - ml[0];
    const float w1 = dl[1] * float(extract_bits(q0, 4u, 4u)) - ml[1];
    const float w2 = dl[2] * float(extract_bits(q1, 0u, 4u)) - ml[2];
    const float w3 = dl[3] * float(extract_bits(q1, 4u, 4u)) - ml[3];
    const float w4 = dl[4] * float(extract_bits(q2, 0u, 4u)) - ml[4];
    const float w5 = dl[5] * float(extract_bits(q2, 4u, 4u)) - ml[5];
    const float w6 = dl[6] * float(extract_bits(q3, 0u, 4u)) - ml[6];
    const float w7 = dl[7] * float(extract_bits(q3, 4u, 4u)) - ml[7];
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

kernel void llmopt_q4_k_linear_f16_m2_x2_l32(
    device const half* input [[buffer(0)]],
    device const block_q4_K* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint columns_per_row = (params.n + 1u) >> 1;
  const uint task = gid >> 5;
  const uint total_tasks = params.m * columns_per_row;
  if (task >= total_tasks) return;

  const uint row = task / columns_per_row;
  const uint first_col = (task - row * columns_per_row) << 1;
  const uint superblock_quarter = lane >> 3;
  const uint local_lane = lane & 7u;
  const uint quant_half = local_lane >> 2;
  const uint quant_offset = local_lane & 3u;
  const uint num_superblocks = params.k >> 8;
  device const half* activation = input + row * params.k
    + (superblock_quarter << 8) + (quant_half << 6)
    + (quant_offset << 3);

  float low_values[16];
  float high_values[16];
  float accumulators[2] = { 0.0f, 0.0f };
  ushort packed_scales[4];
  thread const uchar* scale_bytes =
    reinterpret_cast<thread const uchar*>(packed_scales);

  for (uint sb = superblock_quarter; sb < num_superblocks; sb += 4u) {
    float4 activation_sums = float4(0.0f);
    for (uint index = 0; index < 8u; ++index) {
      low_values[index] = float(activation[index]);
      activation_sums[0] += low_values[index];
      low_values[index + 8u] = float(activation[index + 32u]);
      activation_sums[1] += low_values[index + 8u];
      high_values[index] = float(activation[index + 128u]);
      activation_sums[2] += high_values[index];
      high_values[index + 8u] = float(activation[index + 160u]);
      activation_sums[3] += high_values[index + 8u];
    }

    for (uint column_offset = 0; column_offset < 2u; ++column_offset) {
      const uint col = first_col + column_offset;
      if (col >= params.n) continue;
      device const block_q4_K& block =
        weight[col * num_superblocks + sb];
      device const ushort* scales =
        reinterpret_cast<device const ushort*>(block.scales) + quant_half;
      packed_scales[0] = scales[0] & 0x3f3fu;
      packed_scales[1] = scales[2] & 0x3f3fu;
      packed_scales[2] = ((scales[4] >> 0) & 0x0f0fu)
        | ((scales[0] & 0xc0c0u) >> 2);
      packed_scales[3] = ((scales[4] >> 4) & 0x0f0fu)
        | ((scales[2] & 0xc0c0u) >> 2);

      device const ushort* low_quant =
        reinterpret_cast<device const ushort*>(block.qs)
        + 16u * quant_half + 4u * quant_offset;
      device const ushort* high_quant = low_quant + 32u;
      float4 low_dot = float4(0.0f);
      float4 high_dot = float4(0.0f);
      for (uint index = 0; index < 4u; ++index) {
        low_dot[0] += low_values[2u * index] * float(low_quant[index] & 0x000fu);
        low_dot[1] += low_values[2u * index + 1u]
          * float(low_quant[index] & 0x0f00u);
        low_dot[2] += low_values[2u * index + 8u]
          * float(low_quant[index] & 0x00f0u);
        low_dot[3] += low_values[2u * index + 9u]
          * float(low_quant[index] & 0xf000u);
        high_dot[0] += high_values[2u * index]
          * float(high_quant[index] & 0x000fu);
        high_dot[1] += high_values[2u * index + 1u]
          * float(high_quant[index] & 0x0f00u);
        high_dot[2] += high_values[2u * index + 8u]
          * float(high_quant[index] & 0x00f0u);
        high_dot[3] += high_values[2u * index + 9u]
          * float(high_quant[index] & 0xf000u);
      }

      accumulators[column_offset] += float(block.d)
        * ((low_dot[0] + low_dot[1] / 256.0f) * float(scale_bytes[0])
          + (low_dot[2] + low_dot[3] / 256.0f) * float(scale_bytes[1]) / 16.0f
          + (high_dot[0] + high_dot[1] / 256.0f) * float(scale_bytes[4])
          + (high_dot[2] + high_dot[3] / 256.0f) * float(scale_bytes[5]) / 16.0f)
        - float(block.dmin)
        * (activation_sums[0] * float(scale_bytes[2])
          + activation_sums[1] * float(scale_bytes[3])
          + activation_sums[2] * float(scale_bytes[6])
          + activation_sums[3] * float(scale_bytes[7]));
    }
    activation += 1024u;
  }

  for (uint column_offset = 0; column_offset < 2u; ++column_offset) {
    const uint col = first_col + column_offset;
    const float sum = simd_sum(accumulators[column_offset]);
    if (lane == 0u && col < params.n) {
      const float bias_value = params.has_bias != 0u ? float(bias[col]) : 0.0f;
      output[row * params.n + col] = half(sum + bias_value);
    }
  }
}

kernel void llmopt_q4_k_linear_f16_m2_n2_l32(
    device const half* input [[buffer(0)]],
    device const block_q4_K* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint first_col = (gid >> 5) << 1;
  if (first_col >= params.n) return;

  const uint superblock_quarter = lane >> 3;
  const uint local_lane = lane & 7u;
  const uint quant_half = local_lane >> 2;
  const uint quant_offset = local_lane & 3u;
  const uint num_superblocks = params.k >> 8;
  device const half* activation0 = input
    + (superblock_quarter << 8) + (quant_half << 6)
    + (quant_offset << 3);
  device const half* activation1 = activation0 + params.k;

  float low_values0[16];
  float high_values0[16];
  float low_values1[16];
  float high_values1[16];
  float accumulators[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  ushort packed_scales[4];
  thread const uchar* scale_bytes =
    reinterpret_cast<thread const uchar*>(packed_scales);

  for (uint sb = superblock_quarter; sb < num_superblocks; sb += 4u) {
    float4 activation_sums0 = float4(0.0f);
    float4 activation_sums1 = float4(0.0f);
    for (uint index = 0; index < 8u; ++index) {
      low_values0[index] = float(activation0[index]);
      activation_sums0[0] += low_values0[index];
      low_values0[index + 8u] = float(activation0[index + 32u]);
      activation_sums0[1] += low_values0[index + 8u];
      high_values0[index] = float(activation0[index + 128u]);
      activation_sums0[2] += high_values0[index];
      high_values0[index + 8u] = float(activation0[index + 160u]);
      activation_sums0[3] += high_values0[index + 8u];
      low_values1[index] = float(activation1[index]);
      activation_sums1[0] += low_values1[index];
      low_values1[index + 8u] = float(activation1[index + 32u]);
      activation_sums1[1] += low_values1[index + 8u];
      high_values1[index] = float(activation1[index + 128u]);
      activation_sums1[2] += high_values1[index];
      high_values1[index + 8u] = float(activation1[index + 160u]);
      activation_sums1[3] += high_values1[index + 8u];
    }

    for (uint column_offset = 0; column_offset < 2u; ++column_offset) {
      const uint col = first_col + column_offset;
      if (col >= params.n) continue;
      device const block_q4_K& block =
        weight[col * num_superblocks + sb];
      device const ushort* scales =
        reinterpret_cast<device const ushort*>(block.scales) + quant_half;
      packed_scales[0] = scales[0] & 0x3f3fu;
      packed_scales[1] = scales[2] & 0x3f3fu;
      packed_scales[2] = ((scales[4] >> 0) & 0x0f0fu)
        | ((scales[0] & 0xc0c0u) >> 2);
      packed_scales[3] = ((scales[4] >> 4) & 0x0f0fu)
        | ((scales[2] & 0xc0c0u) >> 2);

      device const ushort* low_quant =
        reinterpret_cast<device const ushort*>(block.qs)
        + 16u * quant_half + 4u * quant_offset;
      device const ushort* high_quant = low_quant + 32u;
      float4 low_dot0 = float4(0.0f);
      float4 high_dot0 = float4(0.0f);
      float4 low_dot1 = float4(0.0f);
      float4 high_dot1 = float4(0.0f);
      for (uint index = 0; index < 4u; ++index) {
        const float q00 = float(low_quant[index] & 0x000fu);
        const float q01 = float(low_quant[index] & 0x0f00u);
        const float q02 = float(low_quant[index] & 0x00f0u);
        const float q03 = float(low_quant[index] & 0xf000u);
        const float q10 = float(high_quant[index] & 0x000fu);
        const float q11 = float(high_quant[index] & 0x0f00u);
        const float q12 = float(high_quant[index] & 0x00f0u);
        const float q13 = float(high_quant[index] & 0xf000u);
        low_dot0[0] += low_values0[2u * index] * q00;
        low_dot0[1] += low_values0[2u * index + 1u] * q01;
        low_dot0[2] += low_values0[2u * index + 8u] * q02;
        low_dot0[3] += low_values0[2u * index + 9u] * q03;
        high_dot0[0] += high_values0[2u * index] * q10;
        high_dot0[1] += high_values0[2u * index + 1u] * q11;
        high_dot0[2] += high_values0[2u * index + 8u] * q12;
        high_dot0[3] += high_values0[2u * index + 9u] * q13;
        low_dot1[0] += low_values1[2u * index] * q00;
        low_dot1[1] += low_values1[2u * index + 1u] * q01;
        low_dot1[2] += low_values1[2u * index + 8u] * q02;
        low_dot1[3] += low_values1[2u * index + 9u] * q03;
        high_dot1[0] += high_values1[2u * index] * q10;
        high_dot1[1] += high_values1[2u * index + 1u] * q11;
        high_dot1[2] += high_values1[2u * index + 8u] * q12;
        high_dot1[3] += high_values1[2u * index + 9u] * q13;
      }

      const float d = float(block.d);
      const float dmin = float(block.dmin);
      const uint accumulator = column_offset << 1;
      accumulators[accumulator] += d
        * ((low_dot0[0] + low_dot0[1] / 256.0f) * float(scale_bytes[0])
          + (low_dot0[2] + low_dot0[3] / 256.0f) * float(scale_bytes[1]) / 16.0f
          + (high_dot0[0] + high_dot0[1] / 256.0f) * float(scale_bytes[4])
          + (high_dot0[2] + high_dot0[3] / 256.0f) * float(scale_bytes[5]) / 16.0f)
        - dmin
        * (activation_sums0[0] * float(scale_bytes[2])
          + activation_sums0[1] * float(scale_bytes[3])
          + activation_sums0[2] * float(scale_bytes[6])
          + activation_sums0[3] * float(scale_bytes[7]));
      accumulators[accumulator + 1u] += d
        * ((low_dot1[0] + low_dot1[1] / 256.0f) * float(scale_bytes[0])
          + (low_dot1[2] + low_dot1[3] / 256.0f) * float(scale_bytes[1]) / 16.0f
          + (high_dot1[0] + high_dot1[1] / 256.0f) * float(scale_bytes[4])
          + (high_dot1[2] + high_dot1[3] / 256.0f) * float(scale_bytes[5]) / 16.0f)
        - dmin
        * (activation_sums1[0] * float(scale_bytes[2])
          + activation_sums1[1] * float(scale_bytes[3])
          + activation_sums1[2] * float(scale_bytes[6])
          + activation_sums1[3] * float(scale_bytes[7]));
    }
    activation0 += 1024u;
    activation1 += 1024u;
  }

  for (uint column_offset = 0; column_offset < 2u; ++column_offset) {
    const uint col = first_col + column_offset;
    const uint accumulator = column_offset << 1;
    const float sum0 = simd_sum(accumulators[accumulator]);
    const float sum1 = simd_sum(accumulators[accumulator + 1u]);
    if (lane == 0u && col < params.n) {
      const float bias_value = params.has_bias != 0u ? float(bias[col]) : 0.0f;
      output[col] = half(sum0 + bias_value);
      output[params.n + col] = half(sum1 + bias_value);
    }
  }
}


kernel void llmopt_q4_k_gated_linear_f16(
    device const half* normalized [[buffer(0)]],
    device const block_q4_K* gate_weight [[buffer(1)]],
    device const block_q4_K* up_weight [[buffer(2)]],
    device half* product [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5;
  const uint total_elements = params.m * params.n;
  if (simd_idx >= total_elements) return;
  const uint row = simd_idx / params.n;
  const uint col = simd_idx - row * params.n;
  const uint num_superblocks = params.k >> 8;
  device const block_q4_K* gate_row = gate_weight + col * num_superblocks;
  device const block_q4_K* up_row = up_weight + col * num_superblocks;
  device const half* in_ptr = normalized + row * params.k;
  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  for (uint sb = 0; sb < num_superblocks; ++sb) {
    device const block_q4_K& gb = gate_row[sb];
    device const block_q4_K& ub = up_row[sb];
    const float gd = float(gb.d);
    const float gdmin = float(gb.dmin);
    const float ud = float(ub.d);
    const float udmin = float(ub.dmin);
    float g_dl[8], g_ml[8], u_dl[8], u_ml[8];
    for (int j = 0; j < 4; ++j) {
      const uint8_t g_sc0 = extract_bits(gb.scales[j], 0u, 6u);
      const uint8_t g_m0 = extract_bits(gb.scales[j + 4], 0u, 6u);
      const uint8_t g_sc1 = extract_bits(gb.scales[j + 8], 0u, 4u) | (extract_bits(gb.scales[j], 6u, 2u) << 4);
      const uint8_t g_m1 = extract_bits(gb.scales[j + 8], 4u, 4u) | (extract_bits(gb.scales[j + 4], 6u, 2u) << 4);
      g_dl[j] = gd * float(g_sc0);
      g_ml[j] = gdmin * float(g_m0);
      g_dl[j + 4] = gd * float(g_sc1);
      g_ml[j + 4] = gdmin * float(g_m1);

      const uint8_t u_sc0 = extract_bits(ub.scales[j], 0u, 6u);
      const uint8_t u_m0 = extract_bits(ub.scales[j + 4], 0u, 6u);
      const uint8_t u_sc1 = extract_bits(ub.scales[j + 8], 0u, 4u) | (extract_bits(ub.scales[j], 6u, 2u) << 4);
      const uint8_t u_m1 = extract_bits(ub.scales[j + 8], 4u, 4u) | (extract_bits(ub.scales[j + 4], 6u, 2u) << 4);
      u_dl[j] = ud * float(u_sc0);
      u_ml[j] = udmin * float(u_m0);
      u_dl[j + 4] = ud * float(u_sc1);
      u_ml[j + 4] = udmin * float(u_m1);
    }
    const uint8_t g_q0 = gb.qs[lane];
    const uint8_t g_q1 = gb.qs[32 + lane];
    const uint8_t g_q2 = gb.qs[64 + lane];
    const uint8_t g_q3 = gb.qs[96 + lane];

    const uint8_t u_q0 = ub.qs[lane];
    const uint8_t u_q1 = ub.qs[32 + lane];
    const uint8_t u_q2 = ub.qs[64 + lane];
    const uint8_t u_q3 = ub.qs[96 + lane];

    const uint base_idx = sb * 256 + lane;
    const float in0 = float(in_ptr[base_idx + 0]);
    const float in1 = float(in_ptr[base_idx + 32]);
    const float in2 = float(in_ptr[base_idx + 64]);
    const float in3 = float(in_ptr[base_idx + 96]);
    const float in4 = float(in_ptr[base_idx + 128]);
    const float in5 = float(in_ptr[base_idx + 160]);
    const float in6 = float(in_ptr[base_idx + 192]);
    const float in7 = float(in_ptr[base_idx + 224]);

    gate_acc += in0 * (g_dl[0] * float(extract_bits(g_q0, 0u, 4u)) - g_ml[0])
              + in1 * (g_dl[1] * float(extract_bits(g_q0, 4u, 4u)) - g_ml[1])
              + in2 * (g_dl[2] * float(extract_bits(g_q1, 0u, 4u)) - g_ml[2])
              + in3 * (g_dl[3] * float(extract_bits(g_q1, 4u, 4u)) - g_ml[3])
              + in4 * (g_dl[4] * float(extract_bits(g_q2, 0u, 4u)) - g_ml[4])
              + in5 * (g_dl[5] * float(extract_bits(g_q2, 4u, 4u)) - g_ml[5])
              + in6 * (g_dl[6] * float(extract_bits(g_q3, 0u, 4u)) - g_ml[6])
              + in7 * (g_dl[7] * float(extract_bits(g_q3, 4u, 4u)) - g_ml[7]);

    up_acc   += in0 * (u_dl[0] * float(extract_bits(u_q0, 0u, 4u)) - u_ml[0])
              + in1 * (u_dl[1] * float(extract_bits(u_q0, 4u, 4u)) - u_ml[1])
              + in2 * (u_dl[2] * float(extract_bits(u_q1, 0u, 4u)) - u_ml[2])
              + in3 * (u_dl[3] * float(extract_bits(u_q1, 4u, 4u)) - u_ml[3])
              + in4 * (u_dl[4] * float(extract_bits(u_q2, 0u, 4u)) - u_ml[4])
              + in5 * (u_dl[5] * float(extract_bits(u_q2, 4u, 4u)) - u_ml[5])
              + in6 * (u_dl[6] * float(extract_bits(u_q3, 0u, 4u)) - u_ml[6])
              + in7 * (u_dl[7] * float(extract_bits(u_q3, 4u, 4u)) - u_ml[7]);
  }
  gate_acc = simd_sum(gate_acc);
  up_acc = simd_sum(up_acc);
  if (lane == 0) {
    product[row * params.n + col] =
      llmopt_gated_product(gate_acc, up_acc, params.has_bias);
  }
}

kernel void llmopt_q4_k_gated_linear_f16_m2_x1_l32(
    device const half* input [[buffer(0)]],
    device const block_q4_K* gate_weight [[buffer(1)]],
    device const block_q4_K* up_weight [[buffer(2)]],
    device half* product [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint task = gid >> 5;
  const uint total_tasks = params.m * params.n;
  if (task >= total_tasks) return;

  const uint row = task / params.n;
  const uint col = task - row * params.n;
  const uint superblock_quarter = lane >> 3;
  const uint local_lane = lane & 7u;
  const uint quant_half = local_lane >> 2;
  const uint quant_offset = local_lane & 3u;
  const uint num_superblocks = params.k >> 8;
  device const half* activation = input + row * params.k
    + (superblock_quarter << 8) + (quant_half << 6)
    + (quant_offset << 3);

  float low_values[16];
  float high_values[16];
  float accumulators[2] = { 0.0f, 0.0f };
  ushort packed_scales[4];
  thread const uchar* scale_bytes =
    reinterpret_cast<thread const uchar*>(packed_scales);

  for (uint sb = superblock_quarter; sb < num_superblocks; sb += 4u) {
    float4 activation_sums = float4(0.0f);
    for (uint index = 0; index < 8u; ++index) {
      low_values[index] = float(activation[index]);
      activation_sums[0] += low_values[index];
      low_values[index + 8u] = float(activation[index + 32u]);
      activation_sums[1] += low_values[index + 8u];
      high_values[index] = float(activation[index + 128u]);
      activation_sums[2] += high_values[index];
      high_values[index + 8u] = float(activation[index + 160u]);
      activation_sums[3] += high_values[index + 8u];
    }

    for (uint projection = 0; projection < 2u; ++projection) {
      device const block_q4_K* projection_weight =
        projection == 0u ? gate_weight : up_weight;
      device const block_q4_K& block =
        projection_weight[col * num_superblocks + sb];
      device const ushort* scales =
        reinterpret_cast<device const ushort*>(block.scales) + quant_half;
      packed_scales[0] = scales[0] & 0x3f3fu;
      packed_scales[1] = scales[2] & 0x3f3fu;
      packed_scales[2] = ((scales[4] >> 0) & 0x0f0fu)
        | ((scales[0] & 0xc0c0u) >> 2);
      packed_scales[3] = ((scales[4] >> 4) & 0x0f0fu)
        | ((scales[2] & 0xc0c0u) >> 2);

      device const ushort* low_quant =
        reinterpret_cast<device const ushort*>(block.qs)
        + 16u * quant_half + 4u * quant_offset;
      device const ushort* high_quant = low_quant + 32u;
      float4 low_dot = float4(0.0f);
      float4 high_dot = float4(0.0f);
      for (uint index = 0; index < 4u; ++index) {
        low_dot[0] += low_values[2u * index]
          * float(low_quant[index] & 0x000fu);
        low_dot[1] += low_values[2u * index + 1u]
          * float(low_quant[index] & 0x0f00u);
        low_dot[2] += low_values[2u * index + 8u]
          * float(low_quant[index] & 0x00f0u);
        low_dot[3] += low_values[2u * index + 9u]
          * float(low_quant[index] & 0xf000u);
        high_dot[0] += high_values[2u * index]
          * float(high_quant[index] & 0x000fu);
        high_dot[1] += high_values[2u * index + 1u]
          * float(high_quant[index] & 0x0f00u);
        high_dot[2] += high_values[2u * index + 8u]
          * float(high_quant[index] & 0x00f0u);
        high_dot[3] += high_values[2u * index + 9u]
          * float(high_quant[index] & 0xf000u);
      }

      accumulators[projection] += float(block.d)
        * ((low_dot[0] + low_dot[1] / 256.0f) * float(scale_bytes[0])
          + (low_dot[2] + low_dot[3] / 256.0f) * float(scale_bytes[1]) / 16.0f
          + (high_dot[0] + high_dot[1] / 256.0f) * float(scale_bytes[4])
          + (high_dot[2] + high_dot[3] / 256.0f) * float(scale_bytes[5]) / 16.0f)
        - float(block.dmin)
        * (activation_sums[0] * float(scale_bytes[2])
          + activation_sums[1] * float(scale_bytes[3])
          + activation_sums[2] * float(scale_bytes[6])
          + activation_sums[3] * float(scale_bytes[7]));
    }
    activation += 1024u;
  }

  const float gate_sum = simd_sum(accumulators[0]);
  const float up_sum = simd_sum(accumulators[1]);
  if (lane == 0u) {
    product[row * params.n + col] =
      llmopt_gated_product(gate_sum, up_sum, params.has_bias);
  }
}

kernel void llmopt_q4_k_linear_f16_llama_mv_add(
    device const half* input [[buffer(0)]],
    device const block_q4_K* weight [[buffer(1)]],
    device const half* residual [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row_base =
      (threadgroup_position.x * 2u + simdgroup) * 2u;
  const uint superblock_quarter = lane >> 3u;
  const uint local_lane = lane & 7u;
  const uint quant_half = local_lane >> 2u;
  const uint quant_offset = local_lane & 3u;
  const uint num_superblocks = params.k >> 8u;
  device const half* activation = input + (superblock_quarter << 8u)
      + (quant_half << 6u) + (quant_offset << 3u);
  float sums[2] = { 0.0f, 0.0f };
  ushort packed_scales[4];
  thread const uchar* scale_bytes =
      reinterpret_cast<thread const uchar*>(packed_scales);

  for (uint sb = superblock_quarter; sb < num_superblocks; sb += 4u) {
    float low_values[16];
    float high_values[16];
    float4 activation_sums = float4(0.0f);
    for (uint index = 0u; index < 8u; ++index) {
      low_values[index] = float(activation[index]);
      activation_sums[0] += low_values[index];
      low_values[index + 8u] = float(activation[index + 32u]);
      activation_sums[1] += low_values[index + 8u];
      high_values[index] = float(activation[index + 128u]);
      activation_sums[2] += high_values[index];
      high_values[index + 8u] = float(activation[index + 160u]);
      activation_sums[3] += high_values[index + 8u];
    }

    for (uint row = 0u; row < 2u; ++row) {
      const uint output_row = row_base + row;
      if (output_row >= params.n) continue;
      device const block_q4_K& block =
          weight[output_row * num_superblocks + sb];
      device const ushort* scales =
          reinterpret_cast<device const ushort*>(block.scales) + quant_half;
      packed_scales[0] = scales[0] & 0x3f3fu;
      packed_scales[1] = scales[2] & 0x3f3fu;
      packed_scales[2] = ((scales[4] >> 0) & 0x0f0fu)
          | ((scales[0] & 0xc0c0u) >> 2);
      packed_scales[3] = ((scales[4] >> 4) & 0x0f0fu)
          | ((scales[2] & 0xc0c0u) >> 2);
      device const ushort* low_quant =
          reinterpret_cast<device const ushort*>(block.qs)
          + 16u * quant_half + 4u * quant_offset;
      device const ushort* high_quant = low_quant + 32u;
      float4 low_dot = float4(0.0f);
      float4 high_dot = float4(0.0f);
      for (uint index = 0u; index < 4u; ++index) {
        low_dot[0] += low_values[2u * index]
            * float(low_quant[index] & 0x000fu);
        low_dot[1] += low_values[2u * index + 1u]
            * float(low_quant[index] & 0x0f00u);
        low_dot[2] += low_values[2u * index + 8u]
            * float(low_quant[index] & 0x00f0u);
        low_dot[3] += low_values[2u * index + 9u]
            * float(low_quant[index] & 0xf000u);
        high_dot[0] += high_values[2u * index]
            * float(high_quant[index] & 0x000fu);
        high_dot[1] += high_values[2u * index + 1u]
            * float(high_quant[index] & 0x0f00u);
        high_dot[2] += high_values[2u * index + 8u]
            * float(high_quant[index] & 0x00f0u);
        high_dot[3] += high_values[2u * index + 9u]
            * float(high_quant[index] & 0xf000u);
      }
      sums[row] += float(block.d)
          * ((low_dot[0] + low_dot[1] / 256.0f) * float(scale_bytes[0])
            + (low_dot[2] + low_dot[3] / 256.0f)
                * float(scale_bytes[1]) / 16.0f
            + (high_dot[0] + high_dot[1] / 256.0f)
                * float(scale_bytes[4])
            + (high_dot[2] + high_dot[3] / 256.0f)
                * float(scale_bytes[5]) / 16.0f)
          - float(block.dmin)
          * (activation_sums[0] * float(scale_bytes[2])
            + activation_sums[1] * float(scale_bytes[3])
            + activation_sums[2] * float(scale_bytes[6])
            + activation_sums[3] * float(scale_bytes[7]));
    }
    activation += 1024u;
  }

  for (uint row = 0u; row < 2u; ++row) {
    const float total = simd_sum(sums[row]);
    const uint output_row = row_base + row;
    if (lane == 0u && output_row < params.n)
      output[output_row] = half(half(total) + residual[output_row]);
  }
}
|}
