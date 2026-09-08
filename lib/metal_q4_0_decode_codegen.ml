open Metal_q4_0_epilogue_codegen

let emit_q4_0_linear_f16_x4 ?(name = "llmopt_q4_0_linear_f16_x4")
    ?(epilogue = Epilogue.none) ~scale_offset ~quant_offset ~zero ~block_elements
    ~block_bytes () =
  let flat =
    Printf.sprintf {|
kernel void llmopt_q4_0_linear_f16_x4(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint base_col = simd_idx << 2u;
  if (base_col >= params.n) return;
  const uint blocks = params.k / %du;
  const uint col_stride = blocks * %du;
  device const uchar* row_blocks0 = weight + base_col * col_stride;
  device const uchar* row_blocks1 = row_blocks0 + col_stride;
  device const uchar* row_blocks2 = row_blocks1 + col_stride;
  device const uchar* row_blocks3 = row_blocks2 + col_stride;
  device const half* activation = input;
  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
  float accumulator2 = 0.0f;
  float accumulator3 = 0.0f;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;

  const uint half_blocks = blocks >> 1u;
  for (uint b = 0u; b < half_blocks; ++b) {
    const uint block_idx = (b << 1u) + b_offset;
    const uint blk_byte_off = block_idx * %du;
    device const uchar* blk0 = row_blocks0 + blk_byte_off;
    device const uchar* blk1 = row_blocks1 + blk_byte_off;
    device const uchar* blk2 = row_blocks2 + blk_byte_off;
    device const uchar* blk3 = row_blocks3 + blk_byte_off;
    const half scale0 = *reinterpret_cast<device const half*>(blk0 + %du);
    const half scale1 = *reinterpret_cast<device const half*>(blk1 + %du);
    const half scale2 = *reinterpret_cast<device const half*>(blk2 + %du);
    const half scale3 = *reinterpret_cast<device const half*>(blk3 + %du);
    const uchar packed0 = blk0[%du + l];
    const uchar packed1 = blk1[%du + l];
    const uchar packed2 = blk2[%du + l];
    const uchar packed3 = blk3[%du + l];
    const uint k_idx = block_idx * %du + l;
    const float a_low = float(activation[k_idx]);
    const float a_high = float(activation[k_idx + 16u]);
    const float q0_low = float(int(packed0 & 15u) - %d);
    const float q0_high = float(int(packed0 >> 4u) - %d);
    const float q1_low = float(int(packed1 & 15u) - %d);
    const float q1_high = float(int(packed1 >> 4u) - %d);
    const float q2_low = float(int(packed2 & 15u) - %d);
    const float q2_high = float(int(packed2 >> 4u) - %d);
    const float q3_low = float(int(packed3 & 15u) - %d);
    const float q3_high = float(int(packed3 >> 4u) - %d);
    accumulator0 += float(scale0) * (a_low * q0_low + a_high * q0_high);
    accumulator1 += float(scale1) * (a_low * q1_low + a_high * q1_high);
    accumulator2 += float(scale2) * (a_low * q2_low + a_high * q2_high);
    accumulator3 += float(scale3) * (a_low * q3_low + a_high * q3_high);
  }
  if ((blocks & 1u) != 0u) {
    const uint block_idx = blocks - 1u;
    const uint blk_byte_off = block_idx * %du;
    const uint k_idx = block_idx * %du + lane;
    const float a = float(activation[k_idx]);
    accumulator0 += a * llmopt_schema_q4_0_value(row_blocks0 + blk_byte_off, lane);
    accumulator1 += a * llmopt_schema_q4_0_value(row_blocks1 + blk_byte_off, lane);
    accumulator2 += a * llmopt_schema_q4_0_value(row_blocks2 + blk_byte_off, lane);
    accumulator3 += a * llmopt_schema_q4_0_value(row_blocks3 + blk_byte_off, lane);
  }
  accumulator0 = simd_sum(accumulator0);
  accumulator1 = simd_sum(accumulator1);
  accumulator2 = simd_sum(accumulator2);
  accumulator3 = simd_sum(accumulator3);
  if (lane == 0u && base_col < params.n) {
    if (params.has_bias != 0u) {
      accumulator0 += float(bias[base_col]);
      if (base_col + 1u < params.n) accumulator1 += float(bias[base_col + 1u]);
      if (base_col + 2u < params.n) accumulator2 += float(bias[base_col + 2u]);
      if (base_col + 3u < params.n) accumulator3 += float(bias[base_col + 3u]);
    }
    output[base_col] = half(accumulator0);
    if (base_col + 1u < params.n) output[base_col + 1u] = half(accumulator1);
    if (base_col + 2u < params.n) output[base_col + 2u] = half(accumulator2);
    if (base_col + 3u < params.n) output[base_col + 3u] = half(accumulator3);
  }
}
|}
    block_elements block_bytes block_bytes scale_offset scale_offset
    scale_offset scale_offset quant_offset quant_offset quant_offset
    quant_offset block_elements zero zero zero zero zero zero zero zero
    block_bytes block_elements
  in
  if Epilogue.equal epilogue Epilogue.none then flat
  else
    match q4_0_x4_digest_of_flat ~name epilogue flat with
    | Ok text -> text
    | Error message ->
        invalid_arg
          (Printf.sprintf "Metal_block_codegen q4_0_linear_f16_x4: %s" message)

let emit_q4_0_linear_f16_x8 ?(split_k = 1) ?(name = "llmopt_q4_0_linear_f16_x8")
    ?(epilogue = Epilogue.none) ~scale_offset ~quant_offset ~zero ~block_elements
    ~block_bytes () =
  let flat_split1 =
    Printf.sprintf {|
kernel void llmopt_q4_0_linear_f16_x8(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_in_tg = tid >> 5u;
  const uint tg_idx = (gid - tid) >> 8u;
  const uint col_group = tg_idx * 8u + simd_in_tg;
  const uint base_col = col_group << 3u;
  if (base_col >= params.n) return;
  const uint blocks = params.k / %du;
  const uint col_stride = blocks * %du;
  device const uchar* row_blocks0 = weight + min(base_col, params.n - 1u) * col_stride;
  device const uchar* row_blocks1 = weight + min(base_col + 1u, params.n - 1u) * col_stride;
  device const uchar* row_blocks2 = weight + min(base_col + 2u, params.n - 1u) * col_stride;
  device const uchar* row_blocks3 = weight + min(base_col + 3u, params.n - 1u) * col_stride;
  device const uchar* row_blocks4 = weight + min(base_col + 4u, params.n - 1u) * col_stride;
  device const uchar* row_blocks5 = weight + min(base_col + 5u, params.n - 1u) * col_stride;
  device const uchar* row_blocks6 = weight + min(base_col + 6u, params.n - 1u) * col_stride;
  device const uchar* row_blocks7 = weight + min(base_col + 7u, params.n - 1u) * col_stride;
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
    const uint blk_byte_off = block_idx * %du;
    device const uchar* blk0 = row_blocks0 + blk_byte_off;
    device const uchar* blk1 = row_blocks1 + blk_byte_off;
    device const uchar* blk2 = row_blocks2 + blk_byte_off;
    device const uchar* blk3 = row_blocks3 + blk_byte_off;
    device const uchar* blk4 = row_blocks4 + blk_byte_off;
    device const uchar* blk5 = row_blocks5 + blk_byte_off;
    device const uchar* blk6 = row_blocks6 + blk_byte_off;
    device const uchar* blk7 = row_blocks7 + blk_byte_off;
    const half scale0 = *reinterpret_cast<device const half*>(blk0 + %du);
    const half scale1 = *reinterpret_cast<device const half*>(blk1 + %du);
    const half scale2 = *reinterpret_cast<device const half*>(blk2 + %du);
    const half scale3 = *reinterpret_cast<device const half*>(blk3 + %du);
    const half scale4 = *reinterpret_cast<device const half*>(blk4 + %du);
    const half scale5 = *reinterpret_cast<device const half*>(blk5 + %du);
    const half scale6 = *reinterpret_cast<device const half*>(blk6 + %du);
    const half scale7 = *reinterpret_cast<device const half*>(blk7 + %du);
    const uchar packed0 = blk0[%du + l];
    const uchar packed1 = blk1[%du + l];
    const uchar packed2 = blk2[%du + l];
    const uchar packed3 = blk3[%du + l];
    const uchar packed4 = blk4[%du + l];
    const uchar packed5 = blk5[%du + l];
    const uchar packed6 = blk6[%du + l];
    const uchar packed7 = blk7[%du + l];
    const uint k_idx = block_idx * %du + l;
    const float a_low = float(activation[k_idx]);
    const float a_high = float(activation[k_idx + 16u]);
    const float q0_low = float(int(packed0 & 15u) - %d);
    const float q0_high = float(int(packed0 >> 4u) - %d);
    const float q1_low = float(int(packed1 & 15u) - %d);
    const float q1_high = float(int(packed1 >> 4u) - %d);
    const float q2_low = float(int(packed2 & 15u) - %d);
    const float q2_high = float(int(packed2 >> 4u) - %d);
    const float q3_low = float(int(packed3 & 15u) - %d);
    const float q3_high = float(int(packed3 >> 4u) - %d);
    const float q4_low = float(int(packed4 & 15u) - %d);
    const float q4_high = float(int(packed4 >> 4u) - %d);
    const float q5_low = float(int(packed5 & 15u) - %d);
    const float q5_high = float(int(packed5 >> 4u) - %d);
    const float q6_low = float(int(packed6 & 15u) - %d);
    const float q6_high = float(int(packed6 >> 4u) - %d);
    const float q7_low = float(int(packed7 & 15u) - %d);
    const float q7_high = float(int(packed7 >> 4u) - %d);
    accumulator0 += float(scale0) * (a_low * q0_low + a_high * q0_high);
    accumulator1 += float(scale1) * (a_low * q1_low + a_high * q1_high);
    accumulator2 += float(scale2) * (a_low * q2_low + a_high * q2_high);
    accumulator3 += float(scale3) * (a_low * q3_low + a_high * q3_high);
    accumulator4 += float(scale4) * (a_low * q4_low + a_high * q4_high);
    accumulator5 += float(scale5) * (a_low * q5_low + a_high * q5_high);
    accumulator6 += float(scale6) * (a_low * q6_low + a_high * q6_high);
    accumulator7 += float(scale7) * (a_low * q7_low + a_high * q7_high);
  }
  if ((blocks & 1u) != 0u) {
    const uint block_idx = blocks - 1u;
    const uint blk_byte_off = block_idx * %du;
    const uint k_idx = block_idx * %du + lane;
    const float a = float(activation[k_idx]);
    accumulator0 += a * llmopt_schema_q4_0_value(row_blocks0 + blk_byte_off, lane);
    accumulator1 += a * llmopt_schema_q4_0_value(row_blocks1 + blk_byte_off, lane);
    accumulator2 += a * llmopt_schema_q4_0_value(row_blocks2 + blk_byte_off, lane);
    accumulator3 += a * llmopt_schema_q4_0_value(row_blocks3 + blk_byte_off, lane);
    accumulator4 += a * llmopt_schema_q4_0_value(row_blocks4 + blk_byte_off, lane);
    accumulator5 += a * llmopt_schema_q4_0_value(row_blocks5 + blk_byte_off, lane);
    accumulator6 += a * llmopt_schema_q4_0_value(row_blocks6 + blk_byte_off, lane);
    accumulator7 += a * llmopt_schema_q4_0_value(row_blocks7 + blk_byte_off, lane);
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
    if (params.has_bias != 0u) {
      accumulator0 += float(bias[base_col]);
      if (base_col + 1u < params.n) accumulator1 += float(bias[base_col + 1u]);
      if (base_col + 2u < params.n) accumulator2 += float(bias[base_col + 2u]);
      if (base_col + 3u < params.n) accumulator3 += float(bias[base_col + 3u]);
      if (base_col + 4u < params.n) accumulator4 += float(bias[base_col + 4u]);
      if (base_col + 5u < params.n) accumulator5 += float(bias[base_col + 5u]);
      if (base_col + 6u < params.n) accumulator6 += float(bias[base_col + 6u]);
      if (base_col + 7u < params.n) accumulator7 += float(bias[base_col + 7u]);
    }
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
|}
    block_elements block_bytes block_bytes
    scale_offset scale_offset scale_offset scale_offset
    scale_offset scale_offset scale_offset scale_offset
    quant_offset quant_offset quant_offset quant_offset
    quant_offset quant_offset quant_offset quant_offset
    block_elements
    zero zero zero zero zero zero zero zero
    zero zero zero zero zero zero zero zero
    block_bytes block_elements
  in
  if Epilogue.equal epilogue Epilogue.none then
    if split_k = 1 then flat_split1
    else
      Printf.sprintf {|
kernel void llmopt_q4_0_linear_f16_x8(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_in_tg = tid >> 5u;
  const uint pair_in_tg = simd_in_tg >> 1u;
  const uint simd_in_pair = simd_in_tg & 1u;
  const uint tg_idx = (gid - tid) >> 8u;
  const uint col_group = tg_idx * 4u + pair_in_tg;
  const uint base_col = col_group << 3u;
  const uint tg_base_col = (tg_idx * 4u) << 3u;
  if (tg_base_col >= params.n) return;
  const uint blocks = params.k / %du;
  const uint col_stride = blocks * %du;
  device const uchar* row_blocks0 = weight + min(base_col, params.n - 1u) * col_stride;
  device const uchar* row_blocks1 = weight + min(base_col + 1u, params.n - 1u) * col_stride;
  device const uchar* row_blocks2 = weight + min(base_col + 2u, params.n - 1u) * col_stride;
  device const uchar* row_blocks3 = weight + min(base_col + 3u, params.n - 1u) * col_stride;
  device const uchar* row_blocks4 = weight + min(base_col + 4u, params.n - 1u) * col_stride;
  device const uchar* row_blocks5 = weight + min(base_col + 5u, params.n - 1u) * col_stride;
  device const uchar* row_blocks6 = weight + min(base_col + 6u, params.n - 1u) * col_stride;
  device const uchar* row_blocks7 = weight + min(base_col + 7u, params.n - 1u) * col_stride;
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

  threadgroup half s_activation[4096];
  for (uint k_start = 0u; k_start < params.k; k_start += 4096u) {
    const uint chunk_len = min(4096u, params.k - k_start);
    for (uint i = tid; i < chunk_len; i += 256u) {
      s_activation[i] = activation[k_start + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint chunk_blocks = chunk_len / %du;
    const uint chunk_half_blocks = chunk_blocks >> 1u;
    const uint split_half_blocks = chunk_half_blocks >> 1u;
    const uint b_start_pair = (simd_in_pair == 0u) ? 0u : split_half_blocks;
    const uint b_end_pair = (simd_in_pair == 0u) ? split_half_blocks : chunk_half_blocks;
    const uint b_start = k_start / %du;

    for (uint b = b_start_pair; b < b_end_pair; ++b) {
      const uint block_idx = b_start + (b << 1u) + b_offset;
      const uint blk_byte_off = block_idx * %du;
      device const uchar* blk0 = row_blocks0 + blk_byte_off;
      device const uchar* blk1 = row_blocks1 + blk_byte_off;
      device const uchar* blk2 = row_blocks2 + blk_byte_off;
      device const uchar* blk3 = row_blocks3 + blk_byte_off;
      device const uchar* blk4 = row_blocks4 + blk_byte_off;
      device const uchar* blk5 = row_blocks5 + blk_byte_off;
      device const uchar* blk6 = row_blocks6 + blk_byte_off;
      device const uchar* blk7 = row_blocks7 + blk_byte_off;
      const half scale0 = *reinterpret_cast<device const half*>(blk0 + %du);
      const half scale1 = *reinterpret_cast<device const half*>(blk1 + %du);
      const half scale2 = *reinterpret_cast<device const half*>(blk2 + %du);
      const half scale3 = *reinterpret_cast<device const half*>(blk3 + %du);
      const half scale4 = *reinterpret_cast<device const half*>(blk4 + %du);
      const half scale5 = *reinterpret_cast<device const half*>(blk5 + %du);
      const half scale6 = *reinterpret_cast<device const half*>(blk6 + %du);
      const half scale7 = *reinterpret_cast<device const half*>(blk7 + %du);
      const uchar packed0 = blk0[%du + l];
      const uchar packed1 = blk1[%du + l];
      const uchar packed2 = blk2[%du + l];
      const uchar packed3 = blk3[%du + l];
      const uchar packed4 = blk4[%du + l];
      const uchar packed5 = blk5[%du + l];
      const uchar packed6 = blk6[%du + l];
      const uchar packed7 = blk7[%du + l];
      const uint s_k = ((b << 1u) + b_offset) * %du + l;
      const float a_low = float(s_activation[s_k]);
      const float a_high = float(s_activation[s_k + 16u]);
      const float q0_low = float(int(packed0 & 15u) - %d);
      const float q0_high = float(int(packed0 >> 4u) - %d);
      const float q1_low = float(int(packed1 & 15u) - %d);
      const float q1_high = float(int(packed1 >> 4u) - %d);
      const float q2_low = float(int(packed2 & 15u) - %d);
      const float q2_high = float(int(packed2 >> 4u) - %d);
      const float q3_low = float(int(packed3 & 15u) - %d);
      const float q3_high = float(int(packed3 >> 4u) - %d);
      const float q4_low = float(int(packed4 & 15u) - %d);
      const float q4_high = float(int(packed4 >> 4u) - %d);
      const float q5_low = float(int(packed5 & 15u) - %d);
      const float q5_high = float(int(packed5 >> 4u) - %d);
      const float q6_low = float(int(packed6 & 15u) - %d);
      const float q6_high = float(int(packed6 >> 4u) - %d);
      const float q7_low = float(int(packed7 & 15u) - %d);
      const float q7_high = float(int(packed7 >> 4u) - %d);
      accumulator0 += float(scale0) * (a_low * q0_low + a_high * q0_high);
      accumulator1 += float(scale1) * (a_low * q1_low + a_high * q1_high);
      accumulator2 += float(scale2) * (a_low * q2_low + a_high * q2_high);
      accumulator3 += float(scale3) * (a_low * q3_low + a_high * q3_high);
      accumulator4 += float(scale4) * (a_low * q4_low + a_high * q4_high);
      accumulator5 += float(scale5) * (a_low * q5_low + a_high * q5_high);
      accumulator6 += float(scale6) * (a_low * q6_low + a_high * q6_high);
      accumulator7 += float(scale7) * (a_low * q7_low + a_high * q7_high);
    }
    if (simd_in_pair == 1u && ((chunk_blocks & 1u) != 0u)) {
      const uint block_idx = b_start + chunk_blocks - 1u;
      const uint blk_byte_off = block_idx * %du;
      const uint s_k = (chunk_blocks - 1u) * %du + lane;
      const float a = float(s_activation[s_k]);
      accumulator0 += a * llmopt_schema_q4_0_value(row_blocks0 + blk_byte_off, lane);
      accumulator1 += a * llmopt_schema_q4_0_value(row_blocks1 + blk_byte_off, lane);
      accumulator2 += a * llmopt_schema_q4_0_value(row_blocks2 + blk_byte_off, lane);
      accumulator3 += a * llmopt_schema_q4_0_value(row_blocks3 + blk_byte_off, lane);
      accumulator4 += a * llmopt_schema_q4_0_value(row_blocks4 + blk_byte_off, lane);
      accumulator5 += a * llmopt_schema_q4_0_value(row_blocks5 + blk_byte_off, lane);
      accumulator6 += a * llmopt_schema_q4_0_value(row_blocks6 + blk_byte_off, lane);
      accumulator7 += a * llmopt_schema_q4_0_value(row_blocks7 + blk_byte_off, lane);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  accumulator0 = simd_sum(accumulator0);
  accumulator1 = simd_sum(accumulator1);
  accumulator2 = simd_sum(accumulator2);
  accumulator3 = simd_sum(accumulator3);
  accumulator4 = simd_sum(accumulator4);
  accumulator5 = simd_sum(accumulator5);
  accumulator6 = simd_sum(accumulator6);
  accumulator7 = simd_sum(accumulator7);

  threadgroup float tg_acc[8][8];
  if (lane == 0u) {
    tg_acc[simd_in_tg][0] = accumulator0;
    tg_acc[simd_in_tg][1] = accumulator1;
    tg_acc[simd_in_tg][2] = accumulator2;
    tg_acc[simd_in_tg][3] = accumulator3;
    tg_acc[simd_in_tg][4] = accumulator4;
    tg_acc[simd_in_tg][5] = accumulator5;
    tg_acc[simd_in_tg][6] = accumulator6;
    tg_acc[simd_in_tg][7] = accumulator7;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_in_pair == 0u && lane == 0u && base_col < params.n) {
    const uint partner = simd_in_tg + 1u;
    accumulator0 = tg_acc[simd_in_tg][0] + tg_acc[partner][0];
    accumulator1 = tg_acc[simd_in_tg][1] + tg_acc[partner][1];
    accumulator2 = tg_acc[simd_in_tg][2] + tg_acc[partner][2];
    accumulator3 = tg_acc[simd_in_tg][3] + tg_acc[partner][3];
    accumulator4 = tg_acc[simd_in_tg][4] + tg_acc[partner][4];
    accumulator5 = tg_acc[simd_in_tg][5] + tg_acc[partner][5];
    accumulator6 = tg_acc[simd_in_tg][6] + tg_acc[partner][6];
    accumulator7 = tg_acc[simd_in_tg][7] + tg_acc[partner][7];

    if (params.has_bias != 0u) {
      accumulator0 += float(bias[base_col]);
      if (base_col + 1u < params.n) accumulator1 += float(bias[base_col + 1u]);
      if (base_col + 2u < params.n) accumulator2 += float(bias[base_col + 2u]);
      if (base_col + 3u < params.n) accumulator3 += float(bias[base_col + 3u]);
      if (base_col + 4u < params.n) accumulator4 += float(bias[base_col + 4u]);
      if (base_col + 5u < params.n) accumulator5 += float(bias[base_col + 5u]);
      if (base_col + 6u < params.n) accumulator6 += float(bias[base_col + 6u]);
      if (base_col + 7u < params.n) accumulator7 += float(bias[base_col + 7u]);
    }
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
|}
    block_elements block_bytes block_elements block_elements block_bytes
    scale_offset scale_offset scale_offset scale_offset
    scale_offset scale_offset scale_offset scale_offset
    quant_offset quant_offset quant_offset quant_offset
    quant_offset quant_offset quant_offset quant_offset
    block_elements
    zero zero zero zero zero zero zero zero
    zero zero zero zero zero zero zero zero
    block_bytes block_elements
  else if split_k = 1 then
    match q4_0_x8_digest_of_flat ~name epilogue flat_split1 with
    | Ok text -> text
    | Error message ->
        invalid_arg
          (Printf.sprintf
             "Metal_block_codegen q4_0_linear_f16_x8: %s" message)
  else
    invalid_arg
      "Metal_block_codegen q4_0_linear_f16_x8: epilogue emission requires \
       split_k = 1"

let emit_q4_0_gated_linear_f16_x8 ~scale_offset ~quant_offset ~zero ~block_elements
    ~block_bytes =
  Printf.sprintf {|
kernel void llmopt_q4_0_gated_linear_f16_x8(
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

  const uint blocks = params.k / %du;
  const uint col_stride = blocks * %du;
  device const uchar* gate_base = gate_weight;
  device const uchar* up_base = up_weight;

  device const uchar* gate_row0 = gate_base + base_col * col_stride;
  device const uchar* gate_row1 = gate_base + min(base_col + 1u, params.n - 1u) * col_stride;
  device const uchar* gate_row2 = gate_base + min(base_col + 2u, params.n - 1u) * col_stride;
  device const uchar* gate_row3 = gate_base + min(base_col + 3u, params.n - 1u) * col_stride;
  device const uchar* gate_row4 = gate_base + min(base_col + 4u, params.n - 1u) * col_stride;
  device const uchar* gate_row5 = gate_base + min(base_col + 5u, params.n - 1u) * col_stride;
  device const uchar* gate_row6 = gate_base + min(base_col + 6u, params.n - 1u) * col_stride;
  device const uchar* gate_row7 = gate_base + min(base_col + 7u, params.n - 1u) * col_stride;

  device const uchar* up_row0 = up_base + base_col * col_stride;
  device const uchar* up_row1 = up_base + min(base_col + 1u, params.n - 1u) * col_stride;
  device const uchar* up_row2 = up_base + min(base_col + 2u, params.n - 1u) * col_stride;
  device const uchar* up_row3 = up_base + min(base_col + 3u, params.n - 1u) * col_stride;
  device const uchar* up_row4 = up_base + min(base_col + 4u, params.n - 1u) * col_stride;
  device const uchar* up_row5 = up_base + min(base_col + 5u, params.n - 1u) * col_stride;
  device const uchar* up_row6 = up_base + min(base_col + 6u, params.n - 1u) * col_stride;
  device const uchar* up_row7 = up_base + min(base_col + 7u, params.n - 1u) * col_stride;

  float gate_acc0 = 0.0f, gate_acc1 = 0.0f, gate_acc2 = 0.0f, gate_acc3 = 0.0f;
  float gate_acc4 = 0.0f, gate_acc5 = 0.0f, gate_acc6 = 0.0f, gate_acc7 = 0.0f;
  float up_acc0 = 0.0f, up_acc1 = 0.0f, up_acc2 = 0.0f, up_acc3 = 0.0f;
  float up_acc4 = 0.0f, up_acc5 = 0.0f, up_acc6 = 0.0f, up_acc7 = 0.0f;

  const uint half_blocks = blocks >> 1u;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;

  for (uint b = 0u; b < half_blocks; ++b) {
    const uint block_idx = (b << 1u) + b_offset;
    const uint blk_byte_off = block_idx * %du;

    device const uchar* g_blk0 = gate_row0 + blk_byte_off;
    device const uchar* g_blk1 = gate_row1 + blk_byte_off;
    device const uchar* g_blk2 = gate_row2 + blk_byte_off;
    device const uchar* g_blk3 = gate_row3 + blk_byte_off;
    device const uchar* g_blk4 = gate_row4 + blk_byte_off;
    device const uchar* g_blk5 = gate_row5 + blk_byte_off;
    device const uchar* g_blk6 = gate_row6 + blk_byte_off;
    device const uchar* g_blk7 = gate_row7 + blk_byte_off;

    device const uchar* u_blk0 = up_row0 + blk_byte_off;
    device const uchar* u_blk1 = up_row1 + blk_byte_off;
    device const uchar* u_blk2 = up_row2 + blk_byte_off;
    device const uchar* u_blk3 = up_row3 + blk_byte_off;
    device const uchar* u_blk4 = up_row4 + blk_byte_off;
    device const uchar* u_blk5 = up_row5 + blk_byte_off;
    device const uchar* u_blk6 = up_row6 + blk_byte_off;
    device const uchar* u_blk7 = up_row7 + blk_byte_off;

    const half g_scale0 = *reinterpret_cast<device const half*>(g_blk0 + %du);
    const half g_scale1 = *reinterpret_cast<device const half*>(g_blk1 + %du);
    const half g_scale2 = *reinterpret_cast<device const half*>(g_blk2 + %du);
    const half g_scale3 = *reinterpret_cast<device const half*>(g_blk3 + %du);
    const half g_scale4 = *reinterpret_cast<device const half*>(g_blk4 + %du);
    const half g_scale5 = *reinterpret_cast<device const half*>(g_blk5 + %du);
    const half g_scale6 = *reinterpret_cast<device const half*>(g_blk6 + %du);
    const half g_scale7 = *reinterpret_cast<device const half*>(g_blk7 + %du);

    const half u_scale0 = *reinterpret_cast<device const half*>(u_blk0 + %du);
    const half u_scale1 = *reinterpret_cast<device const half*>(u_blk1 + %du);
    const half u_scale2 = *reinterpret_cast<device const half*>(u_blk2 + %du);
    const half u_scale3 = *reinterpret_cast<device const half*>(u_blk3 + %du);
    const half u_scale4 = *reinterpret_cast<device const half*>(u_blk4 + %du);
    const half u_scale5 = *reinterpret_cast<device const half*>(u_blk5 + %du);
    const half u_scale6 = *reinterpret_cast<device const half*>(u_blk6 + %du);
    const half u_scale7 = *reinterpret_cast<device const half*>(u_blk7 + %du);

    const uchar g_packed0 = g_blk0[%du + l];
    const uchar g_packed1 = g_blk1[%du + l];
    const uchar g_packed2 = g_blk2[%du + l];
    const uchar g_packed3 = g_blk3[%du + l];
    const uchar g_packed4 = g_blk4[%du + l];
    const uchar g_packed5 = g_blk5[%du + l];
    const uchar g_packed6 = g_blk6[%du + l];
    const uchar g_packed7 = g_blk7[%du + l];

    const uchar u_packed0 = u_blk0[%du + l];
    const uchar u_packed1 = u_blk1[%du + l];
    const uchar u_packed2 = u_blk2[%du + l];
    const uchar u_packed3 = u_blk3[%du + l];
    const uchar u_packed4 = u_blk4[%du + l];
    const uchar u_packed5 = u_blk5[%du + l];
    const uchar u_packed6 = u_blk6[%du + l];
    const uchar u_packed7 = u_blk7[%du + l];

    const uint k_idx = block_idx * %du + l;
    const float a_low = float(in_ptr[k_idx]);
    const float a_high = float(in_ptr[k_idx + 16u]);

    const float g_q0_l = float(int(g_packed0 & 15u) - %d);
    const float g_q0_h = float(int(g_packed0 >> 4u) - %d);
    const float g_q1_l = float(int(g_packed1 & 15u) - %d);
    const float g_q1_h = float(int(g_packed1 >> 4u) - %d);
    const float g_q2_l = float(int(g_packed2 & 15u) - %d);
    const float g_q2_h = float(int(g_packed2 >> 4u) - %d);
    const float g_q3_l = float(int(g_packed3 & 15u) - %d);
    const float g_q3_h = float(int(g_packed3 >> 4u) - %d);
    const float g_q4_l = float(int(g_packed4 & 15u) - %d);
    const float g_q4_h = float(int(g_packed4 >> 4u) - %d);
    const float g_q5_l = float(int(g_packed5 & 15u) - %d);
    const float g_q5_h = float(int(g_packed5 >> 4u) - %d);
    const float g_q6_l = float(int(g_packed6 & 15u) - %d);
    const float g_q6_h = float(int(g_packed6 >> 4u) - %d);
    const float g_q7_l = float(int(g_packed7 & 15u) - %d);
    const float g_q7_h = float(int(g_packed7 >> 4u) - %d);

    const float u_q0_l = float(int(u_packed0 & 15u) - %d);
    const float u_q0_h = float(int(u_packed0 >> 4u) - %d);
    const float u_q1_l = float(int(u_packed1 & 15u) - %d);
    const float u_q1_h = float(int(u_packed1 >> 4u) - %d);
    const float u_q2_l = float(int(u_packed2 & 15u) - %d);
    const float u_q2_h = float(int(u_packed2 >> 4u) - %d);
    const float u_q3_l = float(int(u_packed3 & 15u) - %d);
    const float u_q3_h = float(int(u_packed3 >> 4u) - %d);
    const float u_q4_l = float(int(u_packed4 & 15u) - %d);
    const float u_q4_h = float(int(u_packed4 >> 4u) - %d);
    const float u_q5_l = float(int(u_packed5 & 15u) - %d);
    const float u_q5_h = float(int(u_packed5 >> 4u) - %d);
    const float u_q6_l = float(int(u_packed6 & 15u) - %d);
    const float u_q6_h = float(int(u_packed6 >> 4u) - %d);
    const float u_q7_l = float(int(u_packed7 & 15u) - %d);
    const float u_q7_h = float(int(u_packed7 >> 4u) - %d);

    gate_acc0 += float(g_scale0) * (a_low * g_q0_l + a_high * g_q0_h);
    gate_acc1 += float(g_scale1) * (a_low * g_q1_l + a_high * g_q1_h);
    gate_acc2 += float(g_scale2) * (a_low * g_q2_l + a_high * g_q2_h);
    gate_acc3 += float(g_scale3) * (a_low * g_q3_l + a_high * g_q3_h);
    gate_acc4 += float(g_scale4) * (a_low * g_q4_l + a_high * g_q4_h);
    gate_acc5 += float(g_scale5) * (a_low * g_q5_l + a_high * g_q5_h);
    gate_acc6 += float(g_scale6) * (a_low * g_q6_l + a_high * g_q6_h);
    gate_acc7 += float(g_scale7) * (a_low * g_q7_l + a_high * g_q7_h);

    up_acc0 += float(u_scale0) * (a_low * u_q0_l + a_high * u_q0_h);
    up_acc1 += float(u_scale1) * (a_low * u_q1_l + a_high * u_q1_h);
    up_acc2 += float(u_scale2) * (a_low * u_q2_l + a_high * u_q2_h);
    up_acc3 += float(u_scale3) * (a_low * u_q3_l + a_high * u_q3_h);
    up_acc4 += float(u_scale4) * (a_low * u_q4_l + a_high * u_q4_h);
    up_acc5 += float(u_scale5) * (a_low * u_q5_l + a_high * u_q5_h);
    up_acc6 += float(u_scale6) * (a_low * u_q6_l + a_high * u_q6_h);
    up_acc7 += float(u_scale7) * (a_low * u_q7_l + a_high * u_q7_h);
  }

  gate_acc0 = simd_sum(gate_acc0);
  gate_acc1 = simd_sum(gate_acc1);
  gate_acc2 = simd_sum(gate_acc2);
  gate_acc3 = simd_sum(gate_acc3);
  gate_acc4 = simd_sum(gate_acc4);
  gate_acc5 = simd_sum(gate_acc5);
  gate_acc6 = simd_sum(gate_acc6);
  gate_acc7 = simd_sum(gate_acc7);

  up_acc0 = simd_sum(up_acc0);
  up_acc1 = simd_sum(up_acc1);
  up_acc2 = simd_sum(up_acc2);
  up_acc3 = simd_sum(up_acc3);
  up_acc4 = simd_sum(up_acc4);
  up_acc5 = simd_sum(up_acc5);
  up_acc6 = simd_sum(up_acc6);
  up_acc7 = simd_sum(up_acc7);

  if (lane == 0u) {
    device half* out_row = product + row * params.n;
    out_row[base_col] = llmopt_gated_product(gate_acc0, up_acc0, params.has_bias);
    if (base_col + 1u < params.n)
      out_row[base_col + 1u] = llmopt_gated_product(gate_acc1, up_acc1, params.has_bias);
    if (base_col + 2u < params.n)
      out_row[base_col + 2u] = llmopt_gated_product(gate_acc2, up_acc2, params.has_bias);
    if (base_col + 3u < params.n)
      out_row[base_col + 3u] = llmopt_gated_product(gate_acc3, up_acc3, params.has_bias);
    if (base_col + 4u < params.n)
      out_row[base_col + 4u] = llmopt_gated_product(gate_acc4, up_acc4, params.has_bias);
    if (base_col + 5u < params.n)
      out_row[base_col + 5u] = llmopt_gated_product(gate_acc5, up_acc5, params.has_bias);
    if (base_col + 6u < params.n)
      out_row[base_col + 6u] = llmopt_gated_product(gate_acc6, up_acc6, params.has_bias);
    if (base_col + 7u < params.n)
      out_row[base_col + 7u] = llmopt_gated_product(gate_acc7, up_acc7, params.has_bias);
  }
}
|}
    block_elements block_bytes block_bytes
    scale_offset scale_offset scale_offset scale_offset
    scale_offset scale_offset scale_offset scale_offset
    scale_offset scale_offset scale_offset scale_offset
    scale_offset scale_offset scale_offset scale_offset
    quant_offset quant_offset quant_offset quant_offset
    quant_offset quant_offset quant_offset quant_offset
    quant_offset quant_offset quant_offset quant_offset
    quant_offset quant_offset quant_offset quant_offset
    block_elements
    zero zero zero zero zero zero zero zero
    zero zero zero zero zero zero zero zero
    zero zero zero zero zero zero zero zero
    zero zero zero zero zero zero zero zero
