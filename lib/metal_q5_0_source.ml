(** Metal Q5_0 dequant shader text.  Split out of metal.ml (LOOP item 5). *)

let q5_0_source = {|
struct block_q5_0 {
    half d;
    uint8_t qh[4];
    uint8_t qs[16];
};

kernel void llmopt_dequant_q5_0(
    device const block_q5_0* blocks [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant QuantBlock32Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint block_idx = gid >> 5;
  if (block_idx >= params.total_blocks) return;
  device const block_q5_0& b = blocks[block_idx];
  const half d = b.d;
  const uint8_t high_byte = b.qh[lane >> 3];
  const uint8_t high_bit = (high_byte >> (lane & 7u)) & 1u;
  const uint8_t low_byte = (lane < 16u) ? b.qs[lane] : b.qs[lane - 16u];
  const uint8_t low_nib = (lane < 16u) ? (low_byte & 0x0Fu) : (low_byte >> 4);
  const int q = int(low_nib | (high_bit << 4)) - 16;
  output[block_idx * 32 + lane] = half(float(d) * float(q));
}


kernel void llmopt_q5_0_gated_linear_f16(
    device const half* input [[buffer(0)]],
    device const block_q5_0* gate_weight [[buffer(1)]],
    device const block_q5_0* up_weight [[buffer(2)]],
    device half* product [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half projections[8];
  const uint local_output = simdgroup >> 1;
  const uint projection = simdgroup & 1u;
  const uint task = threadgroup_position.x * 4u + local_output;
  const uint total_tasks = params.m * params.n;
  const bool active = task < total_tasks;
  const uint row = active ? task / params.n : 0u;
  const uint col = active ? task - row * params.n : 0u;
  const uint num_blocks = params.k >> 5;
  device const block_q5_0* row_blocks =
    (projection == 0u ? gate_weight : up_weight) + col * num_blocks;
  device const half* activation = input + row * params.k;
  float acc = 0.0f;
  if (active) {
    for (uint b = 0; b < num_blocks; ++b) {
      device const block_q5_0& blk = row_blocks[b];
      const half d = blk.d;
      const uint8_t high_byte = blk.qh[lane >> 3];
      const uint8_t high_bit = (high_byte >> (lane & 7u)) & 1u;
      const uint8_t low_byte =
        lane < 16u ? blk.qs[lane] : blk.qs[lane - 16u];
      const uint8_t low_nib =
        lane < 16u ? (low_byte & 0x0Fu) : (low_byte >> 4);
      const int q = int(low_nib | (high_bit << 4)) - 16;
      const half in_val = activation[b * 32 + lane];
      acc += float(in_val) * (float(d) * float(q));
    }
  }
  acc = simd_sum(acc);
  if (lane == 0u) projections[simdgroup] = half(acc);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (active && projection == 0u && lane == 0u) {
    product[row * params.n + col] =
      llmopt_gated_product(
        float(projections[simdgroup]), float(projections[simdgroup + 1u]),
        params.has_bias);
  }
}

inline float llmopt_q5_0_llama_mv_dot(
    device const block_q5_0& block,
    thread const float* activation,
    float activation_sum,
    uint half_block) {
  const float d = float(block.d);
  device const uint16_t* quants =
      reinterpret_cast<device const uint16_t*>(&block) + 3u + half_block / 2u;
  const uint32_t high =
      *reinterpret_cast<device const uint32_t*>(block.qh);
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  for (uint index = 0u; index < 8u; index += 2u) {
    const uint16_t packed = quants[index / 2u];
    acc0 += activation[index]
        * float((packed & 0x000fu)
          | ((high >> (index + half_block) << 4u) & 0x0010u));
    acc1 += activation[index + 1u]
        * float((packed & 0x0f00u)
          | ((high >> (index + 1u + half_block) << 12u) & 0x1000u));
    acc2 += activation[index + 8u]
        * float((packed & 0x00f0u)
          | ((high >> (index + half_block + 16u) << 8u) & 0x0100u));
    acc3 += activation[index + 9u]
        * float((packed & 0xf000u)
          | ((high >> (index + 1u + half_block + 16u) << 16u) & 0x10000u));
  }
  return d * (activation_sum * -16.0f + acc0 + acc1 + acc2 + acc3);
}

kernel void llmopt_q5_0_linear_f16_llama_mv_add(
    device const half* input [[buffer(0)]],
    device const block_q5_0* weight [[buffer(1)]],
    device const half* residual [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row_base =
      (threadgroup_position.x * 2u + simdgroup) * 4u;
  const uint block_lane = lane >> 1u;
  const uint half_block = (lane & 1u) * 8u;
  const uint num_blocks = params.k >> 5u;
  float sums[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  device const half* activation_ptr = input + block_lane * 32u + half_block;
  for (uint block = block_lane; block < num_blocks; block += 16u) {
    float activation[16];
    float activation_sum = 0.0f;
    for (uint index = 0u; index < 8u; index += 2u) {
      const float x0 = float(activation_ptr[index]);
      const float x1 = float(activation_ptr[index + 1u]);
      const float x2 = float(activation_ptr[index + 16u]);
      const float x3 = float(activation_ptr[index + 17u]);
      activation_sum += x0 + x1 + x2 + x3;
      activation[index] = x0;
      activation[index + 1u] = x1 / 256.0f;
      activation[index + 8u] = x2 / 16.0f;
      activation[index + 9u] = x3 / 4096.0f;
    }
    for (uint row = 0u; row < 4u; ++row) {
      const uint output_row = row_base + row;
      if (output_row < params.n) {
        device const block_q5_0& quant =
            weight[output_row * num_blocks + block];
        sums[row] += llmopt_q5_0_llama_mv_dot(
            quant, activation, activation_sum, half_block);
      }
    }
    activation_ptr += 512u;
  }
  for (uint row = 0u; row < 4u; ++row) {
    const float total = simd_sum(sums[row]);
    const uint output_row = row_base + row;
    if (lane == 0u && output_row < params.n)
      output[output_row] = half(half(total) + residual[output_row]);
  }
}

kernel void llmopt_q5_0_gated_linear_f16_llama_mv(
    device const half* input [[buffer(0)]],
    device const block_q5_0* gate_weight [[buffer(1)]],
    device const block_q5_0* up_weight [[buffer(2)]],
    device half* product [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half projections[8];
  const uint row_base = threadgroup_position.x * 4u;
  const uint block_lane = lane >> 1u;
  const uint half_block = (lane & 1u) * 8u;
  const uint num_blocks = params.k >> 5u;
  device const block_q5_0* selected_weight =
      simdgroup == 0u ? gate_weight : up_weight;
  float sums[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  device const half* activation_ptr = input + block_lane * 32u + half_block;
  for (uint block = block_lane; block < num_blocks; block += 16u) {
    float activation[16];
    float activation_sum = 0.0f;
    for (uint index = 0u; index < 8u; index += 2u) {
      const float x0 = float(activation_ptr[index]);
      const float x1 = float(activation_ptr[index + 1u]);
      const float x2 = float(activation_ptr[index + 16u]);
      const float x3 = float(activation_ptr[index + 17u]);
      activation_sum += x0 + x1 + x2 + x3;
      activation[index] = x0;
      activation[index + 1u] = x1 / 256.0f;
      activation[index + 8u] = x2 / 16.0f;
      activation[index + 9u] = x3 / 4096.0f;
    }
    for (uint row = 0u; row < 4u; ++row) {
      const uint output_row = row_base + row;
      if (output_row < params.n) {
        device const block_q5_0& quant =
            selected_weight[output_row * num_blocks + block];
        sums[row] += llmopt_q5_0_llama_mv_dot(
            quant, activation, activation_sum, half_block);
      }
    }
    activation_ptr += 512u;
  }
  for (uint row = 0u; row < 4u; ++row) {
    const float total = simd_sum(sums[row]);
    if (lane == 0u) projections[simdgroup * 4u + row] = half(total);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup == 0u && lane < 4u && row_base + lane < params.n) {
    product[row_base + lane] = llmopt_gated_product(
        float(projections[lane]), float(projections[4u + lane]),
        params.has_bias);
  }
}
|}
