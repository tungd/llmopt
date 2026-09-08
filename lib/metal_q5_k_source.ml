(** Metal Q5_K dequant shader text.  Split out of metal.ml (LOOP item 5). *)

let q5_k_source = {|
struct block_q5_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

kernel void llmopt_dequant_q5_k(
    device const block_q5_K* blocks [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant QuantBlock256Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint sb_idx = gid >> 5;
  if (sb_idx >= params.total_superblocks) return;
  device const block_q5_K& b = blocks[sb_idx];
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
  const uint8_t qh = b.qh[lane];
  const uint8_t h0 = extract_bits(qh, 0u, 1u);
  const uint8_t h1 = extract_bits(qh, 1u, 1u);
  const uint8_t h2 = extract_bits(qh, 2u, 1u);
  const uint8_t h3 = extract_bits(qh, 3u, 1u);
  const uint8_t h4 = extract_bits(qh, 4u, 1u);
  const uint8_t h5 = extract_bits(qh, 5u, 1u);
  const uint8_t h6 = extract_bits(qh, 6u, 1u);
  const uint8_t h7 = extract_bits(qh, 7u, 1u);
  const uint8_t q0 = b.qs[lane];
  const uint8_t q1 = b.qs[32 + lane];
  const uint8_t q2 = b.qs[64 + lane];
  const uint8_t q3 = b.qs[96 + lane];
  output[sb_idx * 256 + 0 + lane]   = half(dl[0] * float(extract_bits(q0, 0u, 4u) | (h0 << 4)) - ml[0]);
  output[sb_idx * 256 + 32 + lane]  = half(dl[1] * float(extract_bits(q0, 4u, 4u) | (h1 << 4)) - ml[1]);
  output[sb_idx * 256 + 64 + lane]  = half(dl[2] * float(extract_bits(q1, 0u, 4u) | (h2 << 4)) - ml[2]);
  output[sb_idx * 256 + 96 + lane]  = half(dl[3] * float(extract_bits(q1, 4u, 4u) | (h3 << 4)) - ml[3]);
  output[sb_idx * 256 + 128 + lane] = half(dl[4] * float(extract_bits(q2, 0u, 4u) | (h4 << 4)) - ml[4]);
  output[sb_idx * 256 + 160 + lane] = half(dl[5] * float(extract_bits(q2, 4u, 4u) | (h5 << 4)) - ml[5]);
  output[sb_idx * 256 + 192 + lane] = half(dl[6] * float(extract_bits(q3, 0u, 4u) | (h6 << 4)) - ml[6]);
  output[sb_idx * 256 + 224 + lane] = half(dl[7] * float(extract_bits(q3, 4u, 4u) | (h7 << 4)) - ml[7]);
}


kernel void llmopt_q5_k_linear_f16_splitk(
    device const half* input [[buffer(0)]],
    device const block_q5_K* weight [[buffer(1)]],
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
  device const block_q5_K* row_blocks = weight + col * num_superblocks;
  device const half* in_ptr = input;

  threadgroup float partial_sums[2][4];

  float acc = 0.0f;
  for (uint sb = split_idx; sb < num_superblocks; sb += 4u) {
    device const block_q5_K& b = row_blocks[sb];
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
    const uint8_t qh = b.qh[lane];
    const uint8_t h0 = extract_bits(qh, 0u, 1u);
    const uint8_t h1 = extract_bits(qh, 1u, 1u);
    const uint8_t h2 = extract_bits(qh, 2u, 1u);
    const uint8_t h3 = extract_bits(qh, 3u, 1u);
    const uint8_t h4 = extract_bits(qh, 4u, 1u);
    const uint8_t h5 = extract_bits(qh, 5u, 1u);
    const uint8_t h6 = extract_bits(qh, 6u, 1u);
    const uint8_t h7 = extract_bits(qh, 7u, 1u);
    const uint8_t q0 = b.qs[lane];
    const uint8_t q1 = b.qs[32 + lane];
    const uint8_t q2 = b.qs[64 + lane];
    const uint8_t q3 = b.qs[96 + lane];
    const uint base_idx = sb * 256 + lane;
    acc += float(in_ptr[base_idx + 0])   * (dl[0] * float(extract_bits(q0, 0u, 4u) | (h0 << 4)) - ml[0]);
    acc += float(in_ptr[base_idx + 32])  * (dl[1] * float(extract_bits(q0, 4u, 4u) | (h1 << 4)) - ml[1]);
    acc += float(in_ptr[base_idx + 64])  * (dl[2] * float(extract_bits(q1, 0u, 4u) | (h2 << 4)) - ml[2]);
    acc += float(in_ptr[base_idx + 96])  * (dl[3] * float(extract_bits(q1, 4u, 4u) | (h3 << 4)) - ml[3]);
    acc += float(in_ptr[base_idx + 128]) * (dl[4] * float(extract_bits(q2, 0u, 4u) | (h4 << 4)) - ml[4]);
    acc += float(in_ptr[base_idx + 160]) * (dl[5] * float(extract_bits(q2, 4u, 4u) | (h5 << 4)) - ml[5]);
    acc += float(in_ptr[base_idx + 192]) * (dl[6] * float(extract_bits(q3, 0u, 4u) | (h6 << 4)) - ml[6]);
    acc += float(in_ptr[base_idx + 224]) * (dl[7] * float(extract_bits(q3, 4u, 4u) | (h7 << 4)) - ml[7]);
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

kernel void llmopt_q5_k_linear_f16_splitk_m2(
    device const half* input [[buffer(0)]],
    device const block_q5_K* weight [[buffer(1)]],
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
  device const block_q5_K* row_blocks = weight + col * num_superblocks;
  device const half* in0 = input;
  device const half* in1 = input + params.k;

  threadgroup float partial_sums0[2][4];
  threadgroup float partial_sums1[2][4];

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (uint sb = split_idx; sb < num_superblocks; sb += 4u) {
    device const block_q5_K& b = row_blocks[sb];
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
    const uint8_t qh = b.qh[lane];
    const uint8_t q0 = b.qs[lane];
    const uint8_t q1 = b.qs[32 + lane];
    const uint8_t q2 = b.qs[64 + lane];
    const uint8_t q3 = b.qs[96 + lane];
    const uint base_idx = sb * 256 + lane;
    const float w0 = dl[0] * float(extract_bits(q0, 0u, 4u) | (extract_bits(qh, 0u, 1u) << 4)) - ml[0];
    const float w1 = dl[1] * float(extract_bits(q0, 4u, 4u) | (extract_bits(qh, 1u, 1u) << 4)) - ml[1];
    const float w2 = dl[2] * float(extract_bits(q1, 0u, 4u) | (extract_bits(qh, 2u, 1u) << 4)) - ml[2];
    const float w3 = dl[3] * float(extract_bits(q1, 4u, 4u) | (extract_bits(qh, 3u, 1u) << 4)) - ml[3];
    const float w4 = dl[4] * float(extract_bits(q2, 0u, 4u) | (extract_bits(qh, 4u, 1u) << 4)) - ml[4];
    const float w5 = dl[5] * float(extract_bits(q2, 4u, 4u) | (extract_bits(qh, 5u, 1u) << 4)) - ml[5];
    const float w6 = dl[6] * float(extract_bits(q3, 0u, 4u) | (extract_bits(qh, 6u, 1u) << 4)) - ml[6];
    const float w7 = dl[7] * float(extract_bits(q3, 4u, 4u) | (extract_bits(qh, 7u, 1u) << 4)) - ml[7];
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

kernel void llmopt_q5_k_linear_f16_m2_x1_l32(
    device const half* input [[buffer(0)]],
    device const block_q5_K* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint task = gid >> 5;
  const uint total_tasks = params.m * params.n;
  if (task >= total_tasks) return;

  const uint row = task / params.n;
  const uint col = task - row * params.n;
  const uint lane_cluster = lane >> 2;
  const uint superblock_offset = lane & 3u;
  const uint quant_half = lane_cluster >> 2;
  const uint quant_chunk = lane_cluster & 3u;
  const uint element_offset = quant_chunk << 3;
  const uint quant_offset = (quant_half << 5) + element_offset;
  const uint activation_offset = (quant_half << 6) + element_offset;
  const uchar low_high_mask = uchar(1u << (quant_half << 1));
  const uchar high_high_mask = low_high_mask << 1;
  const uchar upper_low_high_mask = low_high_mask << 4;
  const uchar upper_high_high_mask = high_high_mask << 4;
  const uint num_superblocks = params.k >> 8;
  device const half* activation = input + row * params.k
    + (superblock_offset << 8) + activation_offset;

  float low_values[16];
  float high_values[16];
  float accumulator = 0.0f;
  ushort packed_scales[4];
  thread const uchar* scale_bytes =
    reinterpret_cast<thread const uchar*>(packed_scales);

  for (uint sb = superblock_offset; sb < num_superblocks; sb += 4u) {
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

    device const block_q5_K& block = weight[col * num_superblocks + sb];
    device const ushort* scales =
      reinterpret_cast<device const ushort*>(block.scales) + quant_half;
    packed_scales[0] = scales[0] & 0x3f3fu;
    packed_scales[1] = scales[2] & 0x3f3fu;
    packed_scales[2] = ((scales[4] >> 0) & 0x0f0fu)
      | ((scales[0] & 0xc0c0u) >> 2);
    packed_scales[3] = ((scales[4] >> 4) & 0x0f0fu)
      | ((scales[2] & 0xc0c0u) >> 2);

    device const uchar* low_quant = block.qs + quant_offset;
    device const uchar* high_quant = low_quant + 64u;
    device const uchar* high_bits = block.qh + element_offset;
    float4 quant_dots = float4(0.0f);
    float4 high_dots = float4(0.0f);
    for (uint index = 0; index < 8u; ++index) {
      const uchar high = high_bits[index];
      quant_dots[0] += low_values[index]
        * float(low_quant[index] & 0x0fu);
      quant_dots[1] += low_values[index + 8u]
        * float(low_quant[index] & 0xf0u);
      quant_dots[2] += high_values[index]
        * float(high_quant[index] & 0x0fu);
      quant_dots[3] += high_values[index + 8u]
        * float(high_quant[index] & 0xf0u);
      high_dots[0] += (high & low_high_mask) != 0u ? low_values[index] : 0.0f;
      high_dots[1] += (high & high_high_mask) != 0u
        ? low_values[index + 8u] : 0.0f;
      high_dots[2] += (high & upper_low_high_mask) != 0u
        ? high_values[index] : 0.0f;
      high_dots[3] += (high & upper_high_high_mask) != 0u
        ? high_values[index + 8u] : 0.0f;
    }

    accumulator += float(block.d)
      * (float(scale_bytes[0]) * (quant_dots[0] + 16.0f * high_dots[0])
        + float(scale_bytes[1]) * (quant_dots[1] / 16.0f
          + 16.0f * high_dots[1])
        + float(scale_bytes[4]) * (quant_dots[2] + 16.0f * high_dots[2])
        + float(scale_bytes[5]) * (quant_dots[3] / 16.0f
          + 16.0f * high_dots[3]))
      - float(block.dmin)
      * (activation_sums[0] * float(scale_bytes[2])
        + activation_sums[1] * float(scale_bytes[3])
        + activation_sums[2] * float(scale_bytes[6])
        + activation_sums[3] * float(scale_bytes[7]));
    activation += 1024u;
  }

  const float sum = simd_sum(accumulator);
  if (lane == 0u) {
    const float bias_value = params.has_bias != 0u ? float(bias[col]) : 0.0f;
    output[row * params.n + col] = half(sum + bias_value);
  }
}

kernel void llmopt_q5_k_linear_f16_m2_n1_l32(
    device const half* input [[buffer(0)]],
    device const block_q5_K* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint col = gid >> 5;
  if (col >= params.n) return;
  const uint lane_cluster = lane >> 2;
  const uint superblock_offset = lane & 3u;
  const uint quant_half = lane_cluster >> 2;
  const uint quant_chunk = lane_cluster & 3u;
  const uint element_offset = quant_chunk << 3;
  const uint quant_offset = (quant_half << 5) + element_offset;
  const uint activation_offset = (quant_half << 6) + element_offset;
  const uchar low_high_mask = uchar(1u << (quant_half << 1));
  const uchar high_high_mask = low_high_mask << 1;
  const uchar upper_low_high_mask = low_high_mask << 4;
  const uchar upper_high_high_mask = high_high_mask << 4;
  const uint num_superblocks = params.k >> 8;
  device const half* activation0 = input
    + (superblock_offset << 8) + activation_offset;
  device const half* activation1 = activation0 + params.k;

  float low_values0[16];
  float high_values0[16];
  float low_values1[16];
  float high_values1[16];
  float accumulators[2] = { 0.0f, 0.0f };
  ushort packed_scales[4];
  thread const uchar* scale_bytes =
    reinterpret_cast<thread const uchar*>(packed_scales);

  for (uint sb = superblock_offset; sb < num_superblocks; sb += 4u) {
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

    device const block_q5_K& block = weight[col * num_superblocks + sb];
    device const ushort* scales =
      reinterpret_cast<device const ushort*>(block.scales) + quant_half;
    packed_scales[0] = scales[0] & 0x3f3fu;
    packed_scales[1] = scales[2] & 0x3f3fu;
    packed_scales[2] = ((scales[4] >> 0) & 0x0f0fu)
      | ((scales[0] & 0xc0c0u) >> 2);
    packed_scales[3] = ((scales[4] >> 4) & 0x0f0fu)
      | ((scales[2] & 0xc0c0u) >> 2);

    device const uchar* low_quant = block.qs + quant_offset;
    device const uchar* high_quant = low_quant + 64u;
    device const uchar* high_bits = block.qh + element_offset;
    float4 quant_dots0 = float4(0.0f);
    float4 high_dots0 = float4(0.0f);
    float4 quant_dots1 = float4(0.0f);
    float4 high_dots1 = float4(0.0f);
    for (uint index = 0; index < 8u; ++index) {
      const uchar high = high_bits[index];
      const float low_quant_value = float(low_quant[index] & 0x0fu);
      const float low_quant_high = float(low_quant[index] & 0xf0u);
      const float high_quant_value = float(high_quant[index] & 0x0fu);
      const float high_quant_high = float(high_quant[index] & 0xf0u);
      quant_dots0[0] += low_values0[index] * low_quant_value;
      quant_dots0[1] += low_values0[index + 8u] * low_quant_high;
      quant_dots0[2] += high_values0[index] * high_quant_value;
      quant_dots0[3] += high_values0[index + 8u] * high_quant_high;
      quant_dots1[0] += low_values1[index] * low_quant_value;
      quant_dots1[1] += low_values1[index + 8u] * low_quant_high;
      quant_dots1[2] += high_values1[index] * high_quant_value;
      quant_dots1[3] += high_values1[index + 8u] * high_quant_high;
      if ((high & low_high_mask) != 0u) {
        high_dots0[0] += low_values0[index];
        high_dots1[0] += low_values1[index];
      }
      if ((high & high_high_mask) != 0u) {
        high_dots0[1] += low_values0[index + 8u];
        high_dots1[1] += low_values1[index + 8u];
      }
      if ((high & upper_low_high_mask) != 0u) {
        high_dots0[2] += high_values0[index];
        high_dots1[2] += high_values1[index];
      }
      if ((high & upper_high_high_mask) != 0u) {
        high_dots0[3] += high_values0[index + 8u];
        high_dots1[3] += high_values1[index + 8u];
      }
    }

    accumulators[0] += float(block.d)
      * (float(scale_bytes[0]) * (quant_dots0[0] + 16.0f * high_dots0[0])
        + float(scale_bytes[1]) * (quant_dots0[1] / 16.0f
          + 16.0f * high_dots0[1])
        + float(scale_bytes[4]) * (quant_dots0[2] + 16.0f * high_dots0[2])
        + float(scale_bytes[5]) * (quant_dots0[3] / 16.0f
          + 16.0f * high_dots0[3]))
      - float(block.dmin)
      * (activation_sums0[0] * float(scale_bytes[2])
        + activation_sums0[1] * float(scale_bytes[3])
        + activation_sums0[2] * float(scale_bytes[6])
        + activation_sums0[3] * float(scale_bytes[7]));
    accumulators[1] += float(block.d)
      * (float(scale_bytes[0]) * (quant_dots1[0] + 16.0f * high_dots1[0])
        + float(scale_bytes[1]) * (quant_dots1[1] / 16.0f
          + 16.0f * high_dots1[1])
        + float(scale_bytes[4]) * (quant_dots1[2] + 16.0f * high_dots1[2])
        + float(scale_bytes[5]) * (quant_dots1[3] / 16.0f
          + 16.0f * high_dots1[3]))
      - float(block.dmin)
      * (activation_sums1[0] * float(scale_bytes[2])
        + activation_sums1[1] * float(scale_bytes[3])
        + activation_sums1[2] * float(scale_bytes[6])
        + activation_sums1[3] * float(scale_bytes[7]));
    activation0 += 1024u;
    activation1 += 1024u;
  }

  const float sum0 = simd_sum(accumulators[0]);
  const float sum1 = simd_sum(accumulators[1]);
  if (lane == 0u) {
    const float bias_value = params.has_bias != 0u ? float(bias[col]) : 0.0f;
    output[col] = half(sum0 + bias_value);
    output[params.n + col] = half(sum1 + bias_value);
  }
}

kernel void llmopt_q5_k_gated_linear_f16_m2_x1_l32(
    device const half* input [[buffer(0)]],
    device const block_q5_K* gate_weight [[buffer(1)]],
    device const block_q5_K* up_weight [[buffer(2)]],
    device half* product [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint task = gid >> 5;
  const uint total_tasks = params.m * params.n;
  if (task >= total_tasks) return;

  const uint row = task / params.n;
  const uint col = task - row * params.n;
  const uint lane_cluster = lane >> 2;
  const uint superblock_offset = lane & 3u;
  const uint quant_half = lane_cluster >> 2;
  const uint quant_chunk = lane_cluster & 3u;
  const uint element_offset = quant_chunk << 3;
  const uint quant_offset = (quant_half << 5) + element_offset;
  const uint activation_offset = (quant_half << 6) + element_offset;
  const uchar low_high_mask = uchar(1u << (quant_half << 1));
  const uchar high_high_mask = low_high_mask << 1;
  const uchar upper_low_high_mask = low_high_mask << 4;
  const uchar upper_high_high_mask = high_high_mask << 4;
  const uint num_superblocks = params.k >> 8;
  device const half* activation = input + row * params.k
    + (superblock_offset << 8) + activation_offset;

  float low_values[16];
  float high_values[16];
  float accumulators[2] = { 0.0f, 0.0f };
  ushort packed_scales[4];
  thread const uchar* scale_bytes =
    reinterpret_cast<thread const uchar*>(packed_scales);

  for (uint sb = superblock_offset; sb < num_superblocks; sb += 4u) {
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
      device const block_q5_K* projection_weight =
        projection == 0u ? gate_weight : up_weight;
      device const block_q5_K& block =
        projection_weight[col * num_superblocks + sb];
      device const ushort* scales =
        reinterpret_cast<device const ushort*>(block.scales) + quant_half;
      packed_scales[0] = scales[0] & 0x3f3fu;
      packed_scales[1] = scales[2] & 0x3f3fu;
      packed_scales[2] = ((scales[4] >> 0) & 0x0f0fu)
        | ((scales[0] & 0xc0c0u) >> 2);
      packed_scales[3] = ((scales[4] >> 4) & 0x0f0fu)
        | ((scales[2] & 0xc0c0u) >> 2);

      device const uchar* low_quant = block.qs + quant_offset;
      device const uchar* high_quant = low_quant + 64u;
      device const uchar* high_bits = block.qh + element_offset;
      float4 quant_dots = float4(0.0f);
      float4 high_dots = float4(0.0f);
      for (uint index = 0; index < 8u; ++index) {
        const uchar high = high_bits[index];
        quant_dots[0] += low_values[index]
          * float(low_quant[index] & 0x0fu);
        quant_dots[1] += low_values[index + 8u]
          * float(low_quant[index] & 0xf0u);
        quant_dots[2] += high_values[index]
          * float(high_quant[index] & 0x0fu);
        quant_dots[3] += high_values[index + 8u]
          * float(high_quant[index] & 0xf0u);
        high_dots[0] += (high & low_high_mask) != 0u
          ? low_values[index] : 0.0f;
        high_dots[1] += (high & high_high_mask) != 0u
          ? low_values[index + 8u] : 0.0f;
        high_dots[2] += (high & upper_low_high_mask) != 0u
          ? high_values[index] : 0.0f;
        high_dots[3] += (high & upper_high_high_mask) != 0u
          ? high_values[index + 8u] : 0.0f;
      }

      accumulators[projection] += float(block.d)
        * (float(scale_bytes[0]) * (quant_dots[0] + 16.0f * high_dots[0])
          + float(scale_bytes[1]) * (quant_dots[1] / 16.0f
            + 16.0f * high_dots[1])
          + float(scale_bytes[4]) * (quant_dots[2] + 16.0f * high_dots[2])
          + float(scale_bytes[5]) * (quant_dots[3] / 16.0f
            + 16.0f * high_dots[3]))
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


|}
