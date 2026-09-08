(** Metal Q8_0 dequant shader text.  Split out of metal.ml (LOOP item 5). *)

let q8_0_source = {|
struct block_q8_0 {
    half d;
    int8_t qs[32];
};

kernel void llmopt_dequant_q8_0(
    device const block_q8_0* blocks [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant QuantBlock32Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint block_idx = gid >> 5;
  if (block_idx >= params.total_blocks) return;
  const half d = blocks[block_idx].d;
  const int8_t q = blocks[block_idx].qs[lane];
  output[block_idx * 32 + lane] = half(float(d) * float(q));
}


kernel void llmopt_q8_0_linear_f16_m1_r2_sg4(
    device const half* input [[buffer(0)]],
    device const block_q8_0* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simdgroup = tid >> 5;
  const uint column_base = (gid / 128u) << 1;
  const uint num_blocks = params.k >> 5;
  threadgroup float partial_sums[2][4];

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (uint block = simdgroup; block < num_blocks; block += 4u) {
    const uint input_index = block * 32u + lane;
    const float input_value = float(input[input_index]);
    device const block_q8_0& value0 =
        weight[column_base * num_blocks + block];
    acc0 += input_value * (float(value0.d) * float(value0.qs[lane]));
    if (column_base + 1u < params.n) {
      device const block_q8_0& value1 =
          weight[(column_base + 1u) * num_blocks + block];
      acc1 += input_value * (float(value1.d) * float(value1.qs[lane]));
    }
  }
  acc0 = simd_sum(acc0);
  acc1 = simd_sum(acc1);
  if (lane == 0u) {
    partial_sums[0][simdgroup] = acc0;
    partial_sums[1][simdgroup] = acc1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup == 0u && lane < 2u) {
    const uint column = column_base + lane;
    if (column < params.n) {
      float total = partial_sums[lane][0] + partial_sums[lane][1]
          + partial_sums[lane][2] + partial_sums[lane][3];
      if (params.has_bias != 0u) total += float(bias[column]);
      output[column] = half(total);
    }
  }
}

kernel void llmopt_q8_0_linear_f16_llama_mv(
    device const half* input [[buffer(0)]],
    device const block_q8_0* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row_base = threadgroup_position.x * 2u;
  const uint block_in_group = lane >> 2u;
  const uint quant_chunk = lane & 3u;
  const uint block_start = simdgroup * 8u + block_in_group;
  const uint num_blocks = params.k >> 5u;
  float sums[2] = { 0.0f, 0.0f };
  threadgroup float partial_sums[2][4];

  for (uint block = block_start; block < num_blocks; block += 32u) {
    const uint input_offset = block * 32u + quant_chunk * 8u;
    float values[8];
    for (uint index = 0u; index < 8u; ++index)
      values[index] = float(input[input_offset + index]);
    for (uint row = 0u; row < 2u; ++row) {
      const uint output_row = row_base + row;
      if (output_row >= params.n) continue;
      device const block_q8_0& quant =
          weight[output_row * num_blocks + block];
      device const int8_t* quants = quant.qs + quant_chunk * 8u;
      float dot = 0.0f;
      for (uint index = 0u; index < 8u; ++index)
        dot += values[index] * float(quants[index]);
      sums[row] += dot * float(quant.d);
    }
  }

  for (uint row = 0u; row < 2u; ++row) {
    sums[row] = simd_sum(sums[row]);
    if (lane == 0u) partial_sums[row][simdgroup] = sums[row];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup == 0u && lane < 2u) {
    const uint output_row = row_base + lane;
    if (output_row < params.n) {
      float total = partial_sums[lane][0] + partial_sums[lane][1]
          + partial_sums[lane][2] + partial_sums[lane][3];
      if (params.has_bias != 0u) total += float(bias[output_row]);
      output[output_row] = half(total);
    }
  }
}


kernel void llmopt_q8_0_linear_f16_splitk(
    device const half* input [[buffer(0)]],
    device const block_q8_0* weight [[buffer(1)]],
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

  const uint num_blocks = params.k >> 5;
  device const block_q8_0* row_blocks = weight + col * num_blocks;
  device const half* in_ptr = input;

  threadgroup float partial_sums[2][4];

  float acc = 0.0f;
  for (uint b = split_idx; b < num_blocks; b += 4u) {
    const half d = row_blocks[b].d;
    const int8_t q = row_blocks[b].qs[lane];
    const half in_val = in_ptr[b * 32 + lane];
    acc += float(in_val) * (float(d) * float(q));
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

kernel void llmopt_q8_0_linear_f16_splitk_m2(
    device const half* input [[buffer(0)]],
    device const block_q8_0* weight [[buffer(1)]],
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

  const uint num_blocks = params.k >> 5;
  device const block_q8_0* row_blocks = weight + col * num_blocks;
  device const half* in0 = input;
  device const half* in1 = input + params.k;

  threadgroup float partial_sums0[2][4];
  threadgroup float partial_sums1[2][4];

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (uint b = split_idx; b < num_blocks; b += 4u) {
    const float w = float(row_blocks[b].d) * float(row_blocks[b].qs[lane]);
    const uint index = b * 32 + lane;
    acc0 += float(in0[index]) * w;
    acc1 += float(in1[index]) * w;
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

kernel void llmopt_q8_0_dual_swiglu_f16(
    device const half* normalized [[buffer(0)]],
    device const block_q8_0* gate_weight [[buffer(1)]],
    device const block_q8_0* up_weight [[buffer(2)]],
    device half* product [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5;
  const uint total_elements = params.m * params.n;
  if (simd_idx >= total_elements) return;
  const uint row = simd_idx / params.n;
  const uint col = simd_idx - row * params.n;
  const uint num_blocks = params.k >> 5;
  device const block_q8_0* gate_row = gate_weight + col * num_blocks;
  device const block_q8_0* up_row = up_weight + col * num_blocks;
  device const half* in_ptr = normalized + row * params.k;
  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  for (uint b = 0; b < num_blocks; ++b) {
    const half gd = gate_row[b].d;
    const int8_t gq = gate_row[b].qs[lane];
    const half ud = up_row[b].d;
    const int8_t uq = up_row[b].qs[lane];
    const half in_val = in_ptr[b * 32 + lane];
    gate_acc += float(in_val) * (float(gd) * float(gq));
    up_acc   += float(in_val) * (float(ud) * float(uq));
  }
  gate_acc = simd_sum(gate_acc);
  up_acc = simd_sum(up_acc);
  if (lane == 0) {
    const float silu_gate = gate_acc / (1.0f + exp(-gate_acc));
    product[row * params.n + col] = half(silu_gate * up_acc);
  }
}

kernel void llmopt_q8_0_down_add_f16(
    device const half* product [[buffer(0)]],
    device const block_q8_0* down_weight [[buffer(1)]],
    device const half* residual [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5;
  const uint total_elements = params.m * params.k;
  if (simd_idx >= total_elements) return;
  const uint row = simd_idx / params.k;
  const uint col = simd_idx - row * params.k;
  const uint num_blocks = params.n >> 5;
  device const block_q8_0* down_row = down_weight + col * num_blocks;
  device const half* in_ptr = product + row * params.n;
  float down_acc = 0.0f;
  for (uint b = 0; b < num_blocks; ++b) {
    const half d = down_row[b].d;
    const int8_t q = down_row[b].qs[lane];
    const half in_val = in_ptr[b * 32 + lane];
    down_acc += float(in_val) * (float(d) * float(q));
  }
  down_acc = simd_sum(down_acc);
  if (lane == 0) {
    output[row * params.k + col] = half(down_acc + float(residual[row * params.k + col]));
  }
}
|}
