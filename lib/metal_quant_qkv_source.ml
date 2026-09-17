(** Metal quant QKV shader catalog: params + Q4_0/Q5_0 QKV base sources,
   assembled sources and ABI entries.  Split out of metal.ml (LOOP item 5). *)

open Metal_source_common

let quant_qkv_params_source = {|
struct QuantQKVParams {
  uint m;
  uint k;
  uint n_q;
  uint n_k;
  uint n_v;
};
|}

let q4_0_qkv_base_source = {|
kernel void llmopt_q4_0_qkv_linear_f16(
    device const half* input [[buffer(0)]],
    device const block_q4_0* q_weight [[buffer(1)]],
    device const block_q4_0* k_weight [[buffer(2)]],
    device const block_q4_0* v_weight [[buffer(3)]],
    device half* q_output [[buffer(4)]],
    device half* k_output [[buffer(5)]],
    device half* v_output [[buffer(6)]],
    constant QuantQKVParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint tg_simd_idx = (gid - tid) >> 5u;
  const uint q_groups = (params.n_q + 3u) >> 2u;
  const uint k_groups = (params.n_k + 3u) >> 2u;
  const uint v_groups = (params.n_v + 3u) >> 2u;
  const uint total_groups = q_groups + k_groups + v_groups;
  const uint total_elements = params.m * total_groups;
  if (tg_simd_idx >= total_elements) return;

  const uint row = simd_idx / total_groups;
  const uint group_idx = simd_idx - row * total_groups;

  device const half* activation = input + row * params.k;
  threadgroup half s_activation[4096];
  for (uint i = tid; i < params.k; i += 256u) {
    s_activation[i] = activation[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_idx >= total_elements) return;

  uint base_col;
  device const uchar* weight_ptr;
  device half* output;
  uint stride;
  uint limit;
  if (group_idx < q_groups) {
    base_col = group_idx << 2u;
    weight_ptr = reinterpret_cast<device const uchar*>(q_weight);
    output = q_output;
    stride = params.n_q;
    limit = params.n_q;
  } else if (group_idx < q_groups + k_groups) {
    base_col = (group_idx - q_groups) << 2u;
    weight_ptr = reinterpret_cast<device const uchar*>(k_weight);
    output = k_output;
    stride = params.n_k;
    limit = params.n_k;
  } else {
    base_col = (group_idx - q_groups - k_groups) << 2u;
    weight_ptr = reinterpret_cast<device const uchar*>(v_weight);
    output = v_output;
    stride = params.n_v;
    limit = params.n_v;
  }

  const uint blocks = params.k >> 5u;
  const uint col_stride = blocks * 18u;
  device const uchar* row_blocks0 = weight_ptr + base_col * col_stride;
  device const uchar* row_blocks1 = weight_ptr + min(base_col + 1u, limit - 1u) * col_stride;
  device const uchar* row_blocks2 = weight_ptr + min(base_col + 2u, limit - 1u) * col_stride;
  device const uchar* row_blocks3 = weight_ptr + min(base_col + 3u, limit - 1u) * col_stride;

  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
  float accumulator2 = 0.0f;
  float accumulator3 = 0.0f;

  const uint half_blocks = blocks >> 1u;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;

  for (uint b = 0u; b < half_blocks; ++b) {
    const uint block_idx = (b << 1u) + b_offset;
    const uint blk_byte_off = block_idx * 18u;
    device const uchar* blk0 = row_blocks0 + blk_byte_off;
    device const uchar* blk1 = row_blocks1 + blk_byte_off;
    device const uchar* blk2 = row_blocks2 + blk_byte_off;
    device const uchar* blk3 = row_blocks3 + blk_byte_off;

    const half scale0 = *reinterpret_cast<device const half*>(blk0);
    const half scale1 = *reinterpret_cast<device const half*>(blk1);
    const half scale2 = *reinterpret_cast<device const half*>(blk2);
    const half scale3 = *reinterpret_cast<device const half*>(blk3);

    const uchar packed0 = blk0[2u + l];
    const uchar packed1 = blk1[2u + l];
    const uchar packed2 = blk2[2u + l];
    const uchar packed3 = blk3[2u + l];

    const uint k_idx = block_idx * 32u + l;
    const float a_low = float(s_activation[k_idx]);
    const float a_high = float(s_activation[k_idx + 16u]);

    const float q0_low = float(int(packed0 & 15u) - 8);
    const float q0_high = float(int(packed0 >> 4u) - 8);
    const float q1_low = float(int(packed1 & 15u) - 8);
    const float q1_high = float(int(packed1 >> 4u) - 8);
    const float q2_low = float(int(packed2 & 15u) - 8);
    const float q2_high = float(int(packed2 >> 4u) - 8);
    const float q3_low = float(int(packed3 & 15u) - 8);
    const float q3_high = float(int(packed3 >> 4u) - 8);

    accumulator0 += float(scale0) * (a_low * q0_low + a_high * q0_high);
    accumulator1 += float(scale1) * (a_low * q1_low + a_high * q1_high);
    accumulator2 += float(scale2) * (a_low * q2_low + a_high * q2_high);
    accumulator3 += float(scale3) * (a_low * q3_low + a_high * q3_high);
  }

  if ((blocks & 1u) != 0u) {
    const uint block_idx = blocks - 1u;
    const uint blk_byte_off = block_idx * 18u;
    device const uchar* blk0 = row_blocks0 + blk_byte_off;
    device const uchar* blk1 = row_blocks1 + blk_byte_off;
    device const uchar* blk2 = row_blocks2 + blk_byte_off;
    device const uchar* blk3 = row_blocks3 + blk_byte_off;

    const half scale0 = *reinterpret_cast<device const half*>(blk0);
    const half scale1 = *reinterpret_cast<device const half*>(blk1);
    const half scale2 = *reinterpret_cast<device const half*>(blk2);
    const half scale3 = *reinterpret_cast<device const half*>(blk3);

    const uint8_t byte0 = (lane < 16u) ? blk0[2u + lane] : blk0[2u + lane - 16u];
    const uint8_t byte1 = (lane < 16u) ? blk1[2u + lane] : blk1[2u + lane - 16u];
    const uint8_t byte2 = (lane < 16u) ? blk2[2u + lane] : blk2[2u + lane - 16u];
    const uint8_t byte3 = (lane < 16u) ? blk3[2u + lane] : blk3[2u + lane - 16u];

    const int q0 = int((lane < 16u) ? (byte0 & 0x0Fu) : (byte0 >> 4)) - 8;
    const int q1 = int((lane < 16u) ? (byte1 & 0x0Fu) : (byte1 >> 4)) - 8;
    const int q2 = int((lane < 16u) ? (byte2 & 0x0Fu) : (byte2 >> 4)) - 8;
    const int q3 = int((lane < 16u) ? (byte3 & 0x0Fu) : (byte3 >> 4)) - 8;

    const float a = float(s_activation[block_idx * 32u + lane]);
    accumulator0 += float(scale0) * float(q0) * a;
    accumulator1 += float(scale1) * float(q1) * a;
    accumulator2 += float(scale2) * float(q2) * a;
    accumulator3 += float(scale3) * float(q3) * a;
  }

  accumulator0 = simd_sum(accumulator0);
  accumulator1 = simd_sum(accumulator1);
  accumulator2 = simd_sum(accumulator2);
  accumulator3 = simd_sum(accumulator3);

  if (lane == 0u) {
    device half* out_row = output + row * stride;
    out_row[base_col] = half(accumulator0);
    if (base_col + 1u < limit) out_row[base_col + 1u] = half(accumulator1);
    if (base_col + 2u < limit) out_row[base_col + 2u] = half(accumulator2);
    if (base_col + 3u < limit) out_row[base_col + 3u] = half(accumulator3);
  }
}

kernel void llmopt_q4_0_qkv_linear_f16_m4(
    device const half* input [[buffer(0)]],
    device const block_q4_0* q_weight [[buffer(1)]],
    device const block_q4_0* k_weight [[buffer(2)]],
    device const block_q4_0* v_weight [[buffer(3)]],
    device half* q_output [[buffer(4)]],
    device half* k_output [[buffer(5)]],
    device half* v_output [[buffer(6)]],
    constant QuantQKVParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint total_cols = params.n_q + params.n_k + params.n_v;
  const uint m_tiles = (params.m + 3u) >> 2u;
  const uint total_simd = m_tiles * total_cols;
  if (simd_idx >= total_simd) return;
  const uint m_tile = simd_idx / total_cols;
  const uint col_idx = simd_idx - m_tile * total_cols;
  const uint row0 = m_tile << 2u;
  const uint row1 = row0 + 1u;
  const uint row2 = row0 + 2u;
  const uint row3 = row0 + 3u;
  const bool valid1 = row1 < params.m;
  const bool valid2 = row2 < params.m;
  const bool valid3 = row3 < params.m;

  uint col;
  device const uchar* weight_ptr;
  device half* out_ptr;
  uint out_stride;
  if (col_idx < params.n_q) {
    col = col_idx;
    out_stride = params.n_q;
    weight_ptr = reinterpret_cast<device const uchar*>(q_weight);
    out_ptr = q_output;
  } else if (col_idx < params.n_q + params.n_k) {
    col = col_idx - params.n_q;
    out_stride = params.n_k;
    weight_ptr = reinterpret_cast<device const uchar*>(k_weight);
    out_ptr = k_output;
  } else {
    col = col_idx - (params.n_q + params.n_k);
    out_stride = params.n_v;
    weight_ptr = reinterpret_cast<device const uchar*>(v_weight);
    out_ptr = v_output;
  }

  const uint blocks = params.k >> 5u;
  device const uchar* row_blocks = weight_ptr + col * blocks * 18u;

  device const half* in0 = input + row0 * params.k;
  device const half* in1 = valid1 ? input + row1 * params.k : in0;
  device const half* in2 = valid2 ? input + row2 * params.k : in0;
  device const half* in3 = valid3 ? input + row3 * params.k : in0;

  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
  float accumulator2 = 0.0f;
  float accumulator3 = 0.0f;

  const uint half_blocks = blocks >> 1u;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;

  for (uint b = 0u; b < half_blocks; ++b) {
    const uint block_idx = (b << 1u) + b_offset;
    device const uchar* blk = row_blocks + block_idx * 18u;
    const half scale = *reinterpret_cast<device const half*>(blk);
    const uchar packed = blk[2u + l];
    const float s = float(scale);
    const uint k = block_idx * 32u + l;
    const float q_low = float(int(packed & 15u) - 8);
    const float q_high = float(int(packed >> 4u) - 8);

    const float in0_low = float(in0[k]);
    const float in0_high = float(in0[k + 16u]);
    accumulator0 += s * (in0_low * q_low + in0_high * q_high);

    const float in1_low = float(in1[k]);
    const float in1_high = float(in1[k + 16u]);
    accumulator1 += s * (in1_low * q_low + in1_high * q_high);

    const float in2_low = float(in2[k]);
    const float in2_high = float(in2[k + 16u]);
    accumulator2 += s * (in2_low * q_low + in2_high * q_high);

    const float in3_low = float(in3[k]);
    const float in3_high = float(in3[k + 16u]);
    accumulator3 += s * (in3_low * q_low + in3_high * q_high);
  }

  if ((blocks & 1u) != 0u) {
    const uint block_idx = blocks - 1u;
    device const uchar* blk = row_blocks + block_idx * 18u;
    const half scale = *reinterpret_cast<device const half*>(blk);
    const uint8_t byte_val = (lane < 16u) ? blk[2u + lane] : blk[2u + lane - 16u];
    const uint8_t nib = (lane < 16u) ? (byte_val & 0x0Fu) : (byte_val >> 4u);
    const float decoded = float(scale) * float(int(nib) - 8);
    const uint k = block_idx * 32u + lane;
    accumulator0 += float(in0[k]) * decoded;
    accumulator1 += float(in1[k]) * decoded;
    accumulator2 += float(in2[k]) * decoded;
    accumulator3 += float(in3[k]) * decoded;
  }

  accumulator0 = simd_sum(accumulator0);
  accumulator1 = simd_sum(accumulator1);
  accumulator2 = simd_sum(accumulator2);
  accumulator3 = simd_sum(accumulator3);

  if (lane == 0u) {
    out_ptr[row0 * out_stride + col] = half(accumulator0);
    if (valid1) out_ptr[row1 * out_stride + col] = half(accumulator1);
    if (valid2) out_ptr[row2 * out_stride + col] = half(accumulator2);
    if (valid3) out_ptr[row3 * out_stride + col] = half(accumulator3);
  }
}

kernel void llmopt_q4_0_qkv_linear_f16_m16(
    device const half* input [[buffer(0)]],
    device const block_q4_0* q_weight [[buffer(1)]],
    device const block_q4_0* k_weight [[buffer(2)]],
    device const block_q4_0* v_weight [[buffer(3)]],
    device half* q_output [[buffer(4)]],
    device half* k_output [[buffer(5)]],
    device half* v_output [[buffer(6)]],
    constant QuantQKVParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint total_cols = params.n_q + params.n_k + params.n_v;
  const uint m_tiles = (params.m + 15u) >> 4u;
  const uint total_simd = m_tiles * total_cols;
  if (simd_idx >= total_simd) return;
  const uint m_tile = simd_idx / total_cols;
  const uint col_idx = simd_idx - m_tile * total_cols;
  const uint row_base = m_tile << 4u;

  uint col;
  device const uchar* weight_ptr;
  device half* out_ptr;
  uint out_stride;
  if (col_idx < params.n_q) {
    col = col_idx;
    out_stride = params.n_q;
    weight_ptr = reinterpret_cast<device const uchar*>(q_weight);
    out_ptr = q_output;
  } else if (col_idx < params.n_q + params.n_k) {
    col = col_idx - params.n_q;
    out_stride = params.n_k;
    weight_ptr = reinterpret_cast<device const uchar*>(k_weight);
    out_ptr = k_output;
  } else {
    col = col_idx - (params.n_q + params.n_k);
    out_stride = params.n_v;
    weight_ptr = reinterpret_cast<device const uchar*>(v_weight);
    out_ptr = v_output;
  }

  const uint blocks = params.k >> 5u;
  device const uchar* row_blocks = weight_ptr + col * blocks * 18u;

  device const half* in_ptrs[16];
  #pragma unroll
  for (uint r = 0; r < 16; ++r) {
    in_ptrs[r] = (row_base + r < params.m) ? input + (row_base + r) * params.k : input;
  }

  float accumulator[16] = {0.0f};

  const uint half_blocks = blocks >> 1u;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;

  for (uint b = 0u; b < half_blocks; ++b) {
    const uint block_idx = (b << 1u) + b_offset;
    device const uchar* blk = row_blocks + block_idx * 18u;
    const half scale = *reinterpret_cast<device const half*>(blk);
    const uchar packed = blk[2u + l];
    const float s = float(scale);
    const uint k = block_idx * 32u + l;
    const float q_low = float(int(packed & 15u) - 8);
    const float q_high = float(int(packed >> 4u) - 8);
    const float w_low = s * q_low;
    const float w_high = s * q_high;

    #pragma unroll
    for (uint r = 0; r < 16; ++r) {
      accumulator[r] += float(in_ptrs[r][k]) * w_low + float(in_ptrs[r][k + 16u]) * w_high;
    }
  }

  if ((blocks & 1u) != 0u) {
    const uint block_idx = blocks - 1u;
    device const uchar* blk = row_blocks + block_idx * 18u;
    const half scale = *reinterpret_cast<device const half*>(blk);
    const uint8_t byte_val = (lane < 16u) ? blk[2u + lane] : blk[2u + lane - 16u];
    const uint8_t nib = (lane < 16u) ? (byte_val & 0x0Fu) : (byte_val >> 4u);
    const float decoded = float(scale) * float(int(nib) - 8);
    const uint k = block_idx * 32u + lane;
    #pragma unroll
    for (uint r = 0; r < 16; ++r) {
      accumulator[r] += float(in_ptrs[r][k]) * decoded;
    }
  }

  #pragma unroll
  for (uint r = 0; r < 16; ++r) {
    accumulator[r] = simd_sum(accumulator[r]);
  }

  if (lane == 0u) {
    #pragma unroll
    for (uint r = 0; r < 16; ++r) {
      if (row_base + r < params.m) {
        out_ptr[(row_base + r) * out_stride + col] = half(accumulator[r]);
      }
    }
  }
}
|}

let q4_0_qkv_source =
  let schema =
    match Block_schema.of_storage (Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_0) with
    | Ok s -> s
    | Error err -> failwith (Block_schema.Error.to_string err)
  in
  let generated =
    match Metal_block_codegen.qkv_kernels ~target:Target_hardware.default schema with
    | Ok s -> s
    | Error err -> failwith err
  in
  q4_0_qkv_base_source ^ "\n" ^ generated

let q5_0_qkv_homogeneous_source = {|
kernel void llmopt_q5_0_qkv_linear_f16(
    device const half* input [[buffer(0)]],
    device const block_q5_0* q_weight [[buffer(1)]],
    device const block_q5_0* k_weight [[buffer(2)]],
    device const block_q5_0* v_weight [[buffer(3)]],
    device half* q_output [[buffer(4)]],
    device half* k_output [[buffer(5)]],
    device half* v_output [[buffer(6)]],
    constant QuantQKVParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5;
  const uint total_n = params.n_q + params.n_k + params.n_v;
  const uint total_elements = params.m * total_n;
  if (simd_idx >= total_elements) return;
  const uint row = simd_idx / total_n;
  const uint col_idx = simd_idx - row * total_n;
  uint col;
  device const block_q5_0* row_blocks;
  device half* output;
  uint stride;
  if (col_idx < params.n_q) {
    col = col_idx;
    row_blocks = q_weight + col * (params.k >> 5);
    output = q_output;
    stride = params.n_q;
  } else if (col_idx < params.n_q + params.n_k) {
    col = col_idx - params.n_q;
    row_blocks = k_weight + col * (params.k >> 5);
    output = k_output;
    stride = params.n_k;
  } else {
    col = col_idx - params.n_q - params.n_k;
    row_blocks = v_weight + col * (params.k >> 5);
    output = v_output;
    stride = params.n_v;
  }
  device const half* in_ptr = input + row * params.k;
  float acc = 0.0f;
  for (uint b = 0; b < (params.k >> 5); ++b) {
    device const block_q5_0& blk = row_blocks[b];
    const uint8_t high_byte = blk.qh[lane >> 3];
    const uint8_t high_bit = (high_byte >> (lane & 7u)) & 1u;
    const uint8_t low_byte = lane < 16u ? blk.qs[lane] : blk.qs[lane - 16u];
    const uint8_t low_nib = lane < 16u ? (low_byte & 0x0fu) : (low_byte >> 4);
    const int q = int(low_nib | (high_bit << 4)) - 16;
    acc += float(in_ptr[b * 32u + lane]) * (float(blk.d) * float(q));
  }
  acc = simd_sum(acc);
  if (lane == 0u) output[row * stride + col] = half(acc);
}

kernel void llmopt_q5_0_qkv_linear_f16_llama_mv(
    device const half* input [[buffer(0)]],
    device const block_q5_0* q_weight [[buffer(1)]],
    device const block_q5_0* k_weight [[buffer(2)]],
    device const block_q5_0* v_weight [[buffer(3)]],
    device half* q_output [[buffer(4)]],
    device half* k_output [[buffer(5)]],
    device half* v_output [[buffer(6)]],
    constant QuantQKVParams& params [[buffer(7)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint output_base =
      (threadgroup_position.x * 4u + simdgroup) * 4u;
  const uint total_outputs = params.n_q + params.n_k + params.n_v;
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
      const uint output_row = output_base + row;
      if (output_row >= total_outputs) continue;
      device const block_q5_0* selected_weight;
      uint selected_row;
      if (output_row < params.n_q) {
        selected_weight = q_weight;
        selected_row = output_row;
      } else if (output_row < params.n_q + params.n_k) {
        selected_weight = k_weight;
        selected_row = output_row - params.n_q;
      } else {
        selected_weight = v_weight;
        selected_row = output_row - params.n_q - params.n_k;
      }
      device const block_q5_0& quant =
          selected_weight[selected_row * num_blocks + block];
      sums[row] += llmopt_q5_0_llama_mv_dot(
          quant, activation, activation_sum, half_block);
    }
    activation_ptr += 512u;
  }

  for (uint row = 0u; row < 4u; ++row) {
    const float total = simd_sum(sums[row]);
    const uint output_row = output_base + row;
    if (lane != 0u || output_row >= total_outputs) continue;
    if (output_row < params.n_q) {
      q_output[output_row] = half(total);
    } else if (output_row < params.n_q + params.n_k) {
      k_output[output_row - params.n_q] = half(total);
    } else {
      v_output[output_row - params.n_q - params.n_k] = half(total);
    }
  }
}

|}

let q5_0_qkv_mixed_source = {|
kernel void llmopt_q5_0_qkv_linear_vq8_f16(
    device const half* input [[buffer(0)]],
    device const block_q5_0* q_weight [[buffer(1)]],
    device const block_q5_0* k_weight [[buffer(2)]],
    device const block_q8_0* v_weight [[buffer(3)]],
    device half* q_output [[buffer(4)]],
    device half* k_output [[buffer(5)]],
    device half* v_output [[buffer(6)]],
    constant QuantQKVParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5;
  const uint total_n = params.n_q + params.n_k + params.n_v;
  const uint total_elements = params.m * total_n;
  if (simd_idx >= total_elements) return;
  const uint row = simd_idx / total_n;
  const uint col_idx = simd_idx - row * total_n;
  const bool is_v = col_idx >= params.n_q + params.n_k;
  const uint col = is_v ? col_idx - params.n_q - params.n_k
      : (col_idx < params.n_q ? col_idx : col_idx - params.n_q);
  device const half* in_ptr = input + row * params.k;
  float acc = 0.0f;
  if (is_v) {
    device const block_q8_0* row_blocks = v_weight + col * (params.k >> 5);
    for (uint b = 0; b < (params.k >> 5); ++b) {
      device const block_q8_0& blk = row_blocks[b];
      acc += float(in_ptr[b * 32u + lane])
          * (float(blk.d) * float(blk.qs[lane]));
    }
  } else {
    device const block_q5_0* weight = col_idx < params.n_q ? q_weight : k_weight;
    device const block_q5_0* row_blocks = weight + col * (params.k >> 5);
    for (uint b = 0; b < (params.k >> 5); ++b) {
      device const block_q5_0& blk = row_blocks[b];
      const uint8_t high_byte = blk.qh[lane >> 3];
      const uint8_t high_bit = (high_byte >> (lane & 7u)) & 1u;
      const uint8_t low_byte = lane < 16u ? blk.qs[lane] : blk.qs[lane - 16u];
      const uint8_t low_nib = lane < 16u ? (low_byte & 0x0fu) : (low_byte >> 4);
      const int q = int(low_nib | (high_bit << 4)) - 16;
      acc += float(in_ptr[b * 32u + lane]) * (float(blk.d) * float(q));
    }
  }
  acc = simd_sum(acc);
  if (lane == 0u) {
    if (col_idx < params.n_q) q_output[row * params.n_q + col] = half(acc);
    else if (!is_v) k_output[row * params.n_k + col] = half(acc);
    else v_output[row * params.n_v + col] = half(acc);
  }
}

kernel void llmopt_q5_0_qkv_linear_vq8_f16_llama_mv(
    device const half* input [[buffer(0)]],
    device const block_q5_0* q_weight [[buffer(1)]],
    device const block_q5_0* k_weight [[buffer(2)]],
    device const block_q8_0* v_weight [[buffer(3)]],
    device half* q_output [[buffer(4)]],
    device half* k_output [[buffer(5)]],
    device half* v_output [[buffer(6)]],
    constant QuantQKVParams& params [[buffer(7)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint output_base =
      (threadgroup_position.x * 4u + simdgroup) * 4u;
  const uint total_outputs = params.n_q + params.n_k + params.n_v;
  const uint block_lane = lane >> 1u;
  const uint half_block = (lane & 1u) * 8u;
  const uint num_blocks = params.k >> 5u;
  float sums[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  device const half* activation_ptr = input + block_lane * 32u + half_block;

  for (uint block = block_lane; block < num_blocks; block += 16u) {
    float activation[16];
    float raw_activation[16];
    float activation_sum = 0.0f;
    for (uint index = 0u; index < 8u; index += 2u) {
      const float x0 = float(activation_ptr[index]);
      const float x1 = float(activation_ptr[index + 1u]);
      const float x2 = float(activation_ptr[index + 16u]);
      const float x3 = float(activation_ptr[index + 17u]);
      activation_sum += x0 + x1 + x2 + x3;
      raw_activation[index] = x0;
      raw_activation[index + 1u] = x1;
      raw_activation[index + 8u] = x2;
      raw_activation[index + 9u] = x3;
      activation[index] = x0;
      activation[index + 1u] = x1 / 256.0f;
      activation[index + 8u] = x2 / 16.0f;
      activation[index + 9u] = x3 / 4096.0f;
    }
    for (uint row = 0u; row < 4u; ++row) {
      const uint output_row = output_base + row;
      if (output_row >= total_outputs) continue;
      if (output_row < params.n_q + params.n_k) {
        device const block_q5_0* selected_weight =
            output_row < params.n_q ? q_weight : k_weight;
        const uint selected_row = output_row < params.n_q
            ? output_row : output_row - params.n_q;
        device const block_q5_0& quant =
            selected_weight[selected_row * num_blocks + block];
        sums[row] += llmopt_q5_0_llama_mv_dot(
            quant, activation, activation_sum, half_block);
      } else {
        const uint selected_row = output_row - params.n_q - params.n_k;
        device const block_q8_0& quant =
            v_weight[selected_row * num_blocks + block];
        float dot = 0.0f;
        for (uint index = 0u; index < 8u; ++index) {
          dot += raw_activation[index]
              * float(quant.qs[half_block + index]);
          dot += raw_activation[index + 8u]
              * float(quant.qs[16u + half_block + index]);
        }
        sums[row] += dot * float(quant.d);
      }
    }
    activation_ptr += 512u;
  }

  for (uint row = 0u; row < 4u; ++row) {
    const float total = simd_sum(sums[row]);
    const uint output_row = output_base + row;
    if (lane != 0u || output_row >= total_outputs) continue;
    if (output_row < params.n_q) {
      q_output[output_row] = half(total);
    } else if (output_row < params.n_q + params.n_k) {
      k_output[output_row - params.n_q] = half(total);
    } else {
      v_output[output_row - params.n_q - params.n_k] = half(total);
    }
  }
}
|}

let q4_0_qkv_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_qkv_linear_f16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_qkv_linear_f16_m4"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_qkv_linear_f16_m16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_qkv_linear_f16_x8"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_qkv_linear_f16_x16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_qkv_linear_f16_m32"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16 ]

let q5_0_qkv_homogeneous_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q5_0_qkv_linear_f16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q5_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(128, 1, 1)
      ~name:"llmopt_q5_0_qkv_linear_f16_llama_mv"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q5_0) ~output_dtype:Ir.Dtype.Float16 ]

let q5_0_qkv_mixed_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q5_0_qkv_linear_vq8_f16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q5_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(128, 1, 1)
      ~name:"llmopt_q5_0_qkv_linear_vq8_f16_llama_mv"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q5_0) ~output_dtype:Ir.Dtype.Float16 ]

let q5_0_qkv_source = q5_0_qkv_homogeneous_source ^ q5_0_qkv_mixed_source
let q5_0_qkv_entries = q5_0_qkv_homogeneous_entries @ q5_0_qkv_mixed_entries

let quant_qkv_entries = q4_0_qkv_entries @ q5_0_qkv_entries
let quant_qkv_source = quant_qkv_params_source ^ q4_0_qkv_source ^ q5_0_qkv_source
