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
kernel void %s(
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

  const uint cluster_blocks = (blocks + 7u) >> 3u;
  const uint b_sub = lane >> 2u;
  const uint t_in_b = lane & 3u;
  const uint byte_offset = %du + (t_in_b << 2u);

  for (uint c = 0u; c < cluster_blocks; ++c) {
    const uint block_idx = (c << 3u) + b_sub;
    if (block_idx < blocks) {
      const uint blk_byte_off = block_idx * %du;
      const uint l0 = t_in_b << 2u;
      const uint k_idx = block_idx * %du + l0;
      const float a0_lo = float(activation[k_idx]);
      const float a0_hi = float(activation[k_idx + 16u]);
      const float a1_lo = float(activation[k_idx + 1u]);
      const float a1_hi = float(activation[k_idx + 17u]);
      const float a2_lo = float(activation[k_idx + 2u]);
      const float a2_hi = float(activation[k_idx + 18u]);
      const float a3_lo = float(activation[k_idx + 3u]);
      const float a3_hi = float(activation[k_idx + 19u]);

      const float a_sum8 = float(%d) * (a0_lo + a0_hi + a1_lo + a1_hi + a2_lo + a2_hi + a3_lo + a3_hi);

      #define ACC_ROW(r_blk, acc) \
        { \
          device const uchar* blk = r_blk + blk_byte_off; \
          const float scale = float(*reinterpret_cast<device const half*>(blk + %du)); \
          const ushort p_lo = *reinterpret_cast<device const ushort*>(blk + byte_offset); \
          const ushort p_hi = *reinterpret_cast<device const ushort*>(blk + byte_offset + 2u); \
          acc += scale * ( \
              a0_lo * float(p_lo & 15u) + \
              a0_hi * float((p_lo >> 4u) & 15u) + \
              a1_lo * float((p_lo >> 8u) & 15u) + \
              a1_hi * float((p_lo >> 12u) & 15u) + \
              a2_lo * float(p_hi & 15u) + \
              a2_hi * float((p_hi >> 4u) & 15u) + \
              a3_lo * float((p_hi >> 8u) & 15u) + \
              a3_hi * float((p_hi >> 12u) & 15u) - a_sum8); \
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
    name block_elements block_bytes
    quant_offset block_bytes block_elements zero
    scale_offset
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

let emit_q4_0_linear_f16_x16 ?(split_k = 1) ?(name = "llmopt_q4_0_linear_f16_x16")
    ?(epilogue = Epilogue.none) ~scale_offset ~quant_offset ~zero ~block_elements
    ~block_bytes () =
  let row_blocks_decls =
    List.init 16 (fun j ->
        if j = 0 then
          "  device const uchar* row_blocks0 = weight + min(base_col, params.n - 1u) * col_stride;"
        else
          Printf.sprintf
            "  device const uchar* row_blocks%d = weight + min(base_col + %du, params.n - 1u) * col_stride;"
            j j)
    |> String.concat "\n"
  in
  let accumulators_decls =
    List.init 16 (fun j -> Printf.sprintf "  float accumulator%d = 0.0f;" j)
    |> String.concat "\n"
  in
  let oct_accumulates =
    List.init 16 (fun j ->
        Printf.sprintf "    Q4_0_OCT_ACCUMULATE(accumulator%d, row_blocks%d);" j j)
    |> String.concat "\n"
  in
  let rem_accumulates =
    List.init 16 (fun j ->
        Printf.sprintf "    Q4_0_PAIR_ACCUMULATE(accumulator%d, row_blocks%d);" j j)
    |> String.concat "\n"
  in
  let odd_accumulates =
    List.init 16 (fun j ->
        Printf.sprintf
          "    accumulator%d += a * llmopt_schema_q4_0_value(row_blocks%d + blk_byte_off, lane);"
          j j)
    |> String.concat "\n"
  in
  let simd_sums =
    List.init 16 (fun j ->
        Printf.sprintf "  accumulator%d = simd_sum(accumulator%d);" j j)
    |> String.concat "\n"
  in
  let bias_adds =
    List.init 16 (fun j ->
        if j = 0 then "      accumulator0 += float(bias[base_col]);"
        else
          Printf.sprintf
            "      if (base_col + %du < params.n) accumulator%d += float(bias[base_col + %du]);"
            j j j)
    |> String.concat "\n"
  in
  let stores =
    List.init 16 (fun j ->
        if j = 0 then "    output[base_col] = half(accumulator0);"
        else
          Printf.sprintf
            "    if (base_col + %du < params.n) output[base_col + %du] = half(accumulator%d);"
            j j j)
    |> String.concat "\n"
  in
  let flat_split1 =
    Printf.sprintf {|
#define Q4_0_OCT_ACCUMULATE(col_acc, row_ptr) \
  { \
    device const uchar* blk0 = (row_ptr) + blk_byte_off; \
    device const uchar* blk1 = blk0 + 36u; \
    device const uchar* blk2 = blk0 + 72u; \
    device const uchar* blk3 = blk0 + 108u; \
    const half s0 = *reinterpret_cast<device const half*>(blk0 + %du); \
    const half s1 = *reinterpret_cast<device const half*>(blk1 + %du); \
    const half s2 = *reinterpret_cast<device const half*>(blk2 + %du); \
    const half s3 = *reinterpret_cast<device const half*>(blk3 + %du); \
    const uchar p0 = blk0[%du + l]; \
    const uchar p1 = blk1[%du + l]; \
    const uchar p2 = blk2[%du + l]; \
    const uchar p3 = blk3[%du + l]; \
    const float q0_l = float(int(p0 & 15u) - %d); \
    const float q0_h = float(int(p0 >> 4u) - %d); \
    const float q1_l = float(int(p1 & 15u) - %d); \
    const float q1_h = float(int(p1 >> 4u) - %d); \
    const float q2_l = float(int(p2 & 15u) - %d); \
    const float q2_h = float(int(p2 >> 4u) - %d); \
    const float q3_l = float(int(p3 & 15u) - %d); \
    const float q3_h = float(int(p3 >> 4u) - %d); \
    (col_acc) += float(s0) * (a0_low * q0_l + a0_high * q0_h) \
               + float(s1) * (a1_low * q1_l + a1_high * q1_h) \
               + float(s2) * (a2_low * q2_l + a2_high * q2_h) \
               + float(s3) * (a3_low * q3_l + a3_high * q3_h); \
  }

#define Q4_0_PAIR_ACCUMULATE(col_acc, row_ptr) \
  { \
    device const uchar* blk = (row_ptr) + blk_byte_off; \
    const half s = *reinterpret_cast<device const half*>(blk + %du); \
    const uchar p = blk[%du + l]; \
    const float q_l = float(int(p & 15u) - %d); \
    const float q_h = float(int(p >> 4u) - %d); \
    (col_acc) += float(s) * (a_low * q_l + a_high * q_h); \
  }

kernel void %s(
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
  const uint base_col = col_group << 4u;
  if (base_col >= params.n) return;
  const uint blocks = params.k / %du;
  const uint col_stride = blocks * %du;
%s
  device const half* activation = input;
%s
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;

  const uint oct_blocks = blocks >> 3u;
  for (uint b = 0u; b < oct_blocks; ++b) {
    const uint block_idx = (b << 3u) + b_offset;
    const uint blk_byte_off = block_idx * %du;
    const uint k_idx0 = block_idx * %du + l;
    const float a0_low = float(activation[k_idx0]);
    const float a0_high = float(activation[k_idx0 + 16u]);
    const float a1_low = float(activation[k_idx0 + 64u]);
    const float a1_high = float(activation[k_idx0 + 80u]);
    const float a2_low = float(activation[k_idx0 + 128u]);
    const float a2_high = float(activation[k_idx0 + 144u]);
    const float a3_low = float(activation[k_idx0 + 192u]);
    const float a3_high = float(activation[k_idx0 + 208u]);

%s
  }

  const uint rem_start = oct_blocks << 3u;
  const uint rem_half = (blocks - rem_start) >> 1u;
  for (uint b = 0u; b < rem_half; ++b) {
    const uint block_idx = rem_start + (b << 1u) + b_offset;
    const uint blk_byte_off = block_idx * %du;
    const uint k_idx = block_idx * %du + l;
    const float a_low = float(activation[k_idx]);
    const float a_high = float(activation[k_idx + 16u]);

%s
  }

  if ((blocks & 1u) != 0u) {
    const uint block_idx = blocks - 1u;
    const uint blk_byte_off = block_idx * %du;
    const uint k_idx = block_idx * %du + lane;
    const float a = float(activation[k_idx]);
%s
  }

%s

  if (lane == 0u && base_col < params.n) {
    if (params.has_bias != 0u) {
%s
    }
%s
  }
}

#undef Q4_0_OCT_ACCUMULATE
#undef Q4_0_PAIR_ACCUMULATE
|}
      scale_offset scale_offset scale_offset scale_offset
      quant_offset quant_offset quant_offset quant_offset
      zero zero zero zero zero zero zero zero
      scale_offset quant_offset zero zero
      name block_elements block_bytes
      row_blocks_decls accumulators_decls
      block_bytes block_elements oct_accumulates
      block_bytes block_elements rem_accumulates
      block_bytes block_elements odd_accumulates
      simd_sums bias_adds stores
  in
  if Epilogue.equal epilogue Epilogue.none then
    if split_k = 1 then flat_split1
    else invalid_arg "Metal_block_codegen q4_0_linear_f16_x16: split_k > 1 not supported"
  else if split_k = 1 then
    match q4_0_x16_digest_of_flat ~name epilogue flat_split1 with
    | Ok text -> text
    | Error message ->
        invalid_arg
          (Printf.sprintf
             "Metal_block_codegen q4_0_linear_f16_x16: %s" message)
  else
    invalid_arg
      "Metal_block_codegen q4_0_linear_f16_x16: epilogue emission requires \
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

  const uint cluster_blocks = (blocks + 7u) >> 3u;
  const uint b_sub = lane >> 2u;
  const uint t_in_b = lane & 3u;
  const uint byte_offset = %du + (t_in_b << 2u);

  for (uint c = 0u; c < cluster_blocks; ++c) {
    const uint block_idx = (c << 3u) + b_sub;
    if (block_idx < blocks) {
      const uint blk_byte_off = block_idx * %du;
      const uint l0 = t_in_b << 2u;
      const uint k_idx = block_idx * %du + l0;
      const float a0_lo = float(in_ptr[k_idx]);
      const float a0_hi = float(in_ptr[k_idx + 16u]);
      const float a1_lo = float(in_ptr[k_idx + 1u]);
      const float a1_hi = float(in_ptr[k_idx + 17u]);
      const float a2_lo = float(in_ptr[k_idx + 2u]);
      const float a2_hi = float(in_ptr[k_idx + 18u]);
      const float a3_lo = float(in_ptr[k_idx + 3u]);
      const float a3_hi = float(in_ptr[k_idx + 19u]);

      const float a_sum8 = float(%d) * (a0_lo + a0_hi + a1_lo + a1_hi + a2_lo + a2_hi + a3_lo + a3_hi);

      #define ACC_ROW(g_row, u_row, g_acc, u_acc) \
        { \
          device const uchar* g_blk = g_row + blk_byte_off; \
          device const uchar* u_blk = u_row + blk_byte_off; \
          const float g_scale = float(*reinterpret_cast<device const half*>(g_blk + %du)); \
          const float u_scale = float(*reinterpret_cast<device const half*>(u_blk + %du)); \
          const ushort g_lo = *reinterpret_cast<device const ushort*>(g_blk + byte_offset); \
          const ushort g_hi = *reinterpret_cast<device const ushort*>(g_blk + byte_offset + 2u); \
          const ushort u_lo = *reinterpret_cast<device const ushort*>(u_blk + byte_offset); \
          const ushort u_hi = *reinterpret_cast<device const ushort*>(u_blk + byte_offset + 2u); \
          g_acc += g_scale * ( \
              a0_lo * float(g_lo & 15u) + \
              a0_hi * float((g_lo >> 4u) & 15u) + \
              a1_lo * float((g_lo >> 8u) & 15u) + \
              a1_hi * float((g_lo >> 12u) & 15u) + \
              a2_lo * float(g_hi & 15u) + \
              a2_hi * float((g_hi >> 4u) & 15u) + \
              a3_lo * float((g_hi >> 8u) & 15u) + \
              a3_hi * float((g_hi >> 12u) & 15u) - a_sum8); \
          u_acc += u_scale * ( \
              a0_lo * float(u_lo & 15u) + \
              a0_hi * float((u_lo >> 4u) & 15u) + \
              a1_lo * float((u_lo >> 8u) & 15u) + \
              a1_hi * float((u_lo >> 12u) & 15u) + \
              a2_lo * float(u_hi & 15u) + \
              a2_hi * float((u_hi >> 4u) & 15u) + \
              a3_lo * float((u_hi >> 8u) & 15u) + \
              a3_hi * float((u_hi >> 12u) & 15u) - a_sum8); \
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
    block_elements block_bytes
    quant_offset block_bytes block_elements zero
    scale_offset scale_offset

let emit_q4_0_multi_linear_f16_x8 ~name ~epilogue ?(num_slots = 2)
    ~scale_offset ~quant_offset ~zero ~block_elements ~block_bytes () =
  let canonical_epilogue = Applicative_expr.canonicalize epilogue in
  let aux_values = Applicative_expr.required_values canonical_epilogue in
  let num_aux = List.length aux_values in
  let buf_idx_out = 1 + num_slots + num_aux in
  let buf_idx_params = 2 + num_slots + num_aux in
  let weight_buffers =
    List.init num_slots (fun s ->
        Printf.sprintf "    device const uchar* weight_%d [[buffer(%d)]]," s
          (1 + s))
    |> String.concat "\n"
  in
  let aux_buffers =
    if num_aux = 0 then ""
    else
      aux_values
      |> List.mapi (fun k v ->
             let dtype_str =
               match Ir.Value.dtype v with
               | Ir.Dtype.Float16 -> "half"
               | _ -> "float"
             in
             Printf.sprintf "    device const %s* buf_%d [[buffer(%d)]],\n"
               dtype_str k (1 + num_slots + k))
      |> String.concat ""
  in
  let weight_bases =
    List.init num_slots (fun s ->
        let rows =
          List.init 8 (fun j ->
              if j = 0 then
                Printf.sprintf
                  "  device const uchar* row_%d_0 = base_%d + base_col * col_stride;"
                  s s
              else
                Printf.sprintf
                  "  device const uchar* row_%d_%d = base_%d + min(base_col + %du, params.n - 1u) * col_stride;"
                  s j s j)
          |> String.concat "\n"
        in
        Printf.sprintf "  device const uchar* base_%d = weight_%d;\n%s" s s rows)
    |> String.concat "\n\n"
  in
  let accumulators =
    List.init num_slots (fun s ->
        Printf.sprintf
          "  float acc%d_0 = 0.0f, acc%d_1 = 0.0f, acc%d_2 = 0.0f, acc%d_3 = 0.0f;\n\
          \  float acc%d_4 = 0.0f, acc%d_5 = 0.0f, acc%d_6 = 0.0f, acc%d_7 = 0.0f;"
          s s s s s s s s)
    |> String.concat "\n"
  in
  let cluster_block_accs =
    List.init num_slots (fun s ->
        List.init 8 (fun j ->
            Printf.sprintf
              "      {\n\
              \        device const uchar* blk = row_%d_%d + blk_byte_off;\n\
              \        const float scale = float(*reinterpret_cast<device const half*>(blk + %du));\n\
              \        const ushort p_lo = *reinterpret_cast<device const ushort*>(blk + byte_offset);\n\
              \        const ushort p_hi = *reinterpret_cast<device const ushort*>(blk + byte_offset + 2u);\n\
              \        acc%d_%d += scale * (\n\
              \            a0_lo * float(p_lo & 15u) +\n\
              \            a0_hi * float((p_lo >> 4u) & 15u) +\n\
              \            a1_lo * float((p_lo >> 8u) & 15u) +\n\
              \            a1_hi * float((p_lo >> 12u) & 15u) +\n\
              \            a2_lo * float(p_hi & 15u) +\n\
              \            a2_hi * float((p_hi >> 4u) & 15u) +\n\
              \            a3_lo * float((p_hi >> 8u) & 15u) +\n\
              \            a3_hi * float((p_hi >> 12u) & 15u) - a_sum8);\n\
              \      }"
              s j scale_offset s j)
        |> String.concat "\n")
    |> String.concat "\n\n"
  in
  let simd_sums =
    List.init num_slots (fun s ->
        List.init 8 (fun j ->
            Printf.sprintf "  acc%d_%d = simd_sum(acc%d_%d);" s j s j)
        |> String.concat "\n")
    |> String.concat "\n\n"
  in
  let in_bufs =
    if num_aux = 0 then ""
    else
      aux_values
      |> List.mapi (fun k v ->
             let dtype_str =
               match Ir.Value.dtype v with
               | Ir.Dtype.Float16 -> "half"
               | _ -> "float"
             in
             Printf.sprintf
               "    device const %s* in_buf_%d = buf_%d + row * params.n;\n"
               dtype_str k k)
      |> String.concat ""
  in
  let value_var v =
    Printf.sprintf "in_buf_%d" (Ir.Value_id.to_int (Ir.Value.id v))
  in
  let writebacks =
    List.init 8 (fun j ->
        let idx =
          if j = 0 then "base_col" else Printf.sprintf "(base_col + %du)" j
        in
        let expr =
          Applicative_expr.emit_msl_expr ~value_var
            ~slot_emitter:(fun s -> Printf.sprintf "acc%d_%d" s j)
            ~index_expr:idx canonical_epilogue
        in
        if j = 0 then
          Printf.sprintf "    out_row[base_col] = half(%s);" expr
        else
          Printf.sprintf
            "    if (base_col + %du < params.n) out_row[base_col + %du] = half(%s);"
            j j expr)
    |> String.concat "\n"
  in
  Printf.sprintf
    {|
kernel void %s(
    device const half* input [[buffer(0)]],
%s
%s    device half* output [[buffer(%d)]],
    constant QuantLinearParams& params [[buffer(%d)]],
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
%s

%s

  const uint cluster_blocks = (blocks + 7u) >> 3u;
  const uint b_sub = lane >> 2u;
  const uint t_in_b = lane & 3u;
  const uint byte_offset = %du + (t_in_b << 2u);

  for (uint c = 0u; c < cluster_blocks; ++c) {
    const uint block_idx = (c << 3u) + b_sub;
    if (block_idx < blocks) {
      const uint blk_byte_off = block_idx * %du;
      const uint l0 = t_in_b << 2u;
      const uint k_idx = block_idx * %du + l0;
      const float a0_lo = float(in_ptr[k_idx]);
      const float a0_hi = float(in_ptr[k_idx + 16u]);
      const float a1_lo = float(in_ptr[k_idx + 1u]);
      const float a1_hi = float(in_ptr[k_idx + 17u]);
      const float a2_lo = float(in_ptr[k_idx + 2u]);
      const float a2_hi = float(in_ptr[k_idx + 18u]);
      const float a3_lo = float(in_ptr[k_idx + 3u]);
      const float a3_hi = float(in_ptr[k_idx + 19u]);

      const float a_sum8 = float(%d) * (a0_lo + a0_hi + a1_lo + a1_hi + a2_lo + a2_hi + a3_lo + a3_hi);

%s
    }
  }

%s

  if (lane == 0u) {
    device half* out_row = output + row * params.n;
%s%s
  }
}
|}
    name weight_buffers aux_buffers buf_idx_out buf_idx_params
    block_elements block_bytes weight_bases accumulators
    quant_offset block_bytes block_elements zero cluster_block_accs
    simd_sums in_bufs writebacks

let emit_q4_0_multi_linear_f16_x16 ~name ~epilogue ?(num_slots = 2)
    ~scale_offset ~quant_offset ~zero ~block_elements ~block_bytes () =
  let canonical_epilogue = Applicative_expr.canonicalize epilogue in
  let aux_values = Applicative_expr.required_values canonical_epilogue in
  let num_aux = List.length aux_values in
  let buf_idx_out = 1 + num_slots + num_aux in
  let buf_idx_params = 2 + num_slots + num_aux in
  let weight_buffers =
    List.init num_slots (fun s ->
        Printf.sprintf "    device const uchar* weight_%d [[buffer(%d)]]," s
          (1 + s))
    |> String.concat "\n"
  in
  let aux_buffers =
    if num_aux = 0 then ""
    else
      aux_values
      |> List.mapi (fun k v ->
             let dtype_str =
               match Ir.Value.dtype v with
               | Ir.Dtype.Float16 -> "half"
               | _ -> "float"
             in
             Printf.sprintf "    device const %s* buf_%d [[buffer(%d)]],\n"
               dtype_str k (1 + num_slots + k))
      |> String.concat ""
  in
  let weight_bases =
    List.init num_slots (fun s ->
        let rows =
          List.init 16 (fun j ->
              if j = 0 then
                Printf.sprintf
                  "  device const uchar* row_%d_0 = base_%d + base_col * col_stride;"
                  s s
              else
                Printf.sprintf
                  "  device const uchar* row_%d_%d = base_%d + min(base_col + %du, params.n - 1u) * col_stride;"
                  s j s j)
          |> String.concat "\n"
        in
        Printf.sprintf "  device const uchar* base_%d = weight_%d;\n%s" s s rows)
    |> String.concat "\n\n"
  in
  let accumulators =
    List.init num_slots (fun s ->
        let acc_decls =
          List.init 16 (fun j -> Printf.sprintf "acc%d_%d = 0.0f" s j)
          |> String.concat ", "
        in
        Printf.sprintf "  float %s;" acc_decls)
    |> String.concat "\n"
  in
  let oct_accumulates =
    List.init num_slots (fun s ->
        List.init 16 (fun j ->
            Printf.sprintf "    Q4_0_OCT_ACCUMULATE(acc%d_%d, row_%d_%d);" s j s j)
        |> String.concat "\n")
    |> String.concat "\n"
  in
  let rem_accumulates =
    List.init num_slots (fun s ->
        List.init 16 (fun j ->
            Printf.sprintf "    Q4_0_PAIR_ACCUMULATE(acc%d_%d, row_%d_%d);" s j s j)
        |> String.concat "\n")
    |> String.concat "\n"
  in
  let odd_accumulates =
    List.init num_slots (fun s ->
        List.init 16 (fun j ->
            Printf.sprintf
              "    acc%d_%d += a * llmopt_schema_q4_0_value(row_%d_%d + blk_byte_off, lane);"
              s j s j)
        |> String.concat "\n")
    |> String.concat "\n"
  in
  let simd_sums =
    List.init num_slots (fun s ->
        List.init 16 (fun j ->
            Printf.sprintf "  acc%d_%d = simd_sum(acc%d_%d);" s j s j)
        |> String.concat "\n")
    |> String.concat "\n\n"
  in
  let in_bufs =
    if num_aux = 0 then ""
    else
      aux_values
      |> List.mapi (fun k v ->
             let dtype_str =
               match Ir.Value.dtype v with
               | Ir.Dtype.Float16 -> "half"
               | _ -> "float"
             in
             Printf.sprintf
               "    device const %s* in_buf_%d = buf_%d + row * params.n;\n"
               dtype_str k k)
      |> String.concat ""
  in
  let value_var v =
    Printf.sprintf "in_buf_%d" (Ir.Value_id.to_int (Ir.Value.id v))
  in
  let writebacks =
    List.init 16 (fun j ->
        let idx =
          if j = 0 then "base_col" else Printf.sprintf "(base_col + %du)" j
        in
        let expr =
          Applicative_expr.emit_msl_expr ~value_var
            ~slot_emitter:(fun s -> Printf.sprintf "acc%d_%d" s j)
            ~index_expr:idx canonical_epilogue
        in
        if j = 0 then
          Printf.sprintf "    out_row[base_col] = half(%s);" expr
        else
          Printf.sprintf
            "    if (base_col + %du < params.n) out_row[base_col + %du] = half(%s);"
            j j expr)
    |> String.concat "\n"
  in
  Printf.sprintf
    {|
#define Q4_0_OCT_ACCUMULATE(col_acc, row_ptr) \
  { \
    device const uchar* blk0 = (row_ptr) + blk_byte_off; \
    device const uchar* blk1 = blk0 + 36u; \
    device const uchar* blk2 = blk0 + 72u; \
    device const uchar* blk3 = blk0 + 108u; \
    const half s0 = *reinterpret_cast<device const half*>(blk0 + %du); \
    const half s1 = *reinterpret_cast<device const half*>(blk1 + %du); \
    const half s2 = *reinterpret_cast<device const half*>(blk2 + %du); \
    const half s3 = *reinterpret_cast<device const half*>(blk3 + %du); \
    const uchar p0 = blk0[%du + l]; \
    const uchar p1 = blk1[%du + l]; \
    const uchar p2 = blk2[%du + l]; \
    const uchar p3 = blk3[%du + l]; \
    const float q0_l = float(int(p0 & 15u) - %d); \
    const float q0_h = float(int(p0 >> 4u) - %d); \
    const float q1_l = float(int(p1 & 15u) - %d); \
    const float q1_h = float(int(p1 >> 4u) - %d); \
    const float q2_l = float(int(p2 & 15u) - %d); \
    const float q2_h = float(int(p2 >> 4u) - %d); \
    const float q3_l = float(int(p3 & 15u) - %d); \
    const float q3_h = float(int(p3 >> 4u) - %d); \
    (col_acc) += float(s0) * (a0_low * q0_l + a0_high * q0_h) \
               + float(s1) * (a1_low * q1_l + a1_high * q1_h) \
               + float(s2) * (a2_low * q2_l + a2_high * q2_h) \
               + float(s3) * (a3_low * q3_l + a3_high * q3_h); \
  }

#define Q4_0_PAIR_ACCUMULATE(col_acc, row_ptr) \
  { \
    device const uchar* blk = (row_ptr) + blk_byte_off; \
    const half s = *reinterpret_cast<device const half*>(blk + %du); \
    const uchar p = blk[%du + l]; \
    const float q_l = float(int(p & 15u) - %d); \
    const float q_h = float(int(p >> 4u) - %d); \
    (col_acc) += float(s) * (a_low * q_l + a_high * q_h); \
  }

kernel void %s(
    device const half* input [[buffer(0)]],
%s
%s    device half* output [[buffer(%d)]],
    constant QuantLinearParams& params [[buffer(%d)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint col_groups = (params.n + 15u) >> 4u;
  const uint total_elements = params.m * col_groups;
  if (simd_idx >= total_elements) return;

  const uint row = simd_idx / col_groups;
  const uint group_idx = simd_idx - row * col_groups;
  const uint base_col = group_idx << 4u;

  device const half* in_ptr = input + row * params.k;

  const uint blocks = params.k / %du;
  const uint col_stride = blocks * %du;
%s

%s

  const uint oct_blocks = blocks >> 3u;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;

  for (uint b = 0u; b < oct_blocks; ++b) {
    const uint block_idx = (b << 3u) + b_offset;
    const uint blk_byte_off = block_idx * %du;

    const uint k_idx0 = block_idx * %du + l;
    const float a0_low = float(in_ptr[k_idx0]);
    const float a0_high = float(in_ptr[k_idx0 + 16u]);
    const float a1_low = float(in_ptr[k_idx0 + 64u]);
    const float a1_high = float(in_ptr[k_idx0 + 80u]);
    const float a2_low = float(in_ptr[k_idx0 + 128u]);
    const float a2_high = float(in_ptr[k_idx0 + 144u]);
    const float a3_low = float(in_ptr[k_idx0 + 192u]);
    const float a3_high = float(in_ptr[k_idx0 + 208u]);

%s
  }

  const uint rem_start = oct_blocks << 3u;
  const uint rem_half = (blocks - rem_start) >> 1u;
  for (uint b = 0u; b < rem_half; ++b) {
    const uint block_idx = rem_start + (b << 1u) + b_offset;
    const uint blk_byte_off = block_idx * %du;
    const uint k_idx = block_idx * %du + l;
    const float a_low = float(in_ptr[k_idx]);
    const float a_high = float(in_ptr[k_idx + 16u]);

%s
  }

  if ((blocks & 1u) != 0u) {
    const uint block_idx = blocks - 1u;
    const uint blk_byte_off = block_idx * %du;
    const uint k_idx = block_idx * %du + lane;
    const float a = float(in_ptr[k_idx]);
%s
  }

%s

  if (lane == 0u) {
    device half* out_row = output + row * params.n;
%s%s
  }
}

#undef Q4_0_OCT_ACCUMULATE
#undef Q4_0_PAIR_ACCUMULATE
|}
    scale_offset scale_offset scale_offset scale_offset
    quant_offset quant_offset quant_offset quant_offset
    zero zero zero zero zero zero zero zero
    scale_offset quant_offset zero zero
    name weight_buffers aux_buffers buf_idx_out buf_idx_params
    block_elements block_bytes weight_bases accumulators
    block_bytes block_elements oct_accumulates
    block_bytes block_elements rem_accumulates
    block_bytes block_elements odd_accumulates
    simd_sums in_bufs writebacks

let emit_q4_0_multi_export_linear_f16_x8 ~name ~epilogues ?(num_slots = 2)
    ~scale_offset ~quant_offset ~zero ~block_elements ~block_bytes () =
  let num_exports = List.length epilogues in
  let canonical_epilogues = List.map Applicative_expr.canonicalize epilogues in
  let all_aux_values =
    List.concat_map Applicative_expr.required_values canonical_epilogues
    |> List.fold_left
         (fun acc v ->
           if List.exists (fun v' -> Ir.Value.equal v v') acc then acc
           else acc @ [ v ])
         []
  in
  let num_aux = List.length all_aux_values in
  let first_output_buf = 1 + num_slots + num_aux in
  let buf_idx_params = first_output_buf + num_exports in
  let weight_buffers =
    List.init num_slots (fun s ->
        Printf.sprintf "    device const uchar* weight_%d [[buffer(%d)]]," s
          (1 + s))
    |> String.concat "\n"
  in
  let aux_buffers =
    if num_aux = 0 then ""
    else
      all_aux_values
      |> List.mapi (fun k v ->
             let dtype_str =
               match Ir.Value.dtype v with
               | Ir.Dtype.Float16 -> "half"
               | _ -> "float"
             in
             Printf.sprintf "    device const %s* buf_%d [[buffer(%d)]],\n"
               dtype_str k (1 + num_slots + k))
      |> String.concat ""
  in
  let output_buffers =
    List.init num_exports (fun e ->
        Printf.sprintf "    device half* output_%d [[buffer(%d)]]," e
          (first_output_buf + e))
    |> String.concat "\n"
  in
  let weight_bases =
    List.init num_slots (fun s ->
        let rows =
          List.init 8 (fun j ->
              if j = 0 then
                Printf.sprintf
                  "  device const uchar* row_%d_0 = base_%d + base_col * col_stride;"
                  s s
              else
                Printf.sprintf
                  "  device const uchar* row_%d_%d = base_%d + min(base_col + %du, params.n - 1u) * col_stride;"
                  s j s j)
          |> String.concat "\n"
        in
        Printf.sprintf "  device const uchar* base_%d = weight_%d;\n%s" s s rows)
    |> String.concat "\n\n"
  in
  let accumulators =
    List.init num_slots (fun s ->
        Printf.sprintf
          "  float acc%d_0 = 0.0f, acc%d_1 = 0.0f, acc%d_2 = 0.0f, acc%d_3 = 0.0f;\n\
          \  float acc%d_4 = 0.0f, acc%d_5 = 0.0f, acc%d_6 = 0.0f, acc%d_7 = 0.0f;"
          s s s s s s s s)
    |> String.concat "\n"
  in
  let cluster_block_accs =
    List.init num_slots (fun s ->
        List.init 8 (fun j ->
            Printf.sprintf
              "      {\n\
              \        device const uchar* blk = row_%d_%d + blk_byte_off;\n\
              \        const float scale = float(*reinterpret_cast<device const half*>(blk + %du));\n\
              \        const ushort p_lo = *reinterpret_cast<device const ushort*>(blk + byte_offset);\n\
              \        const ushort p_hi = *reinterpret_cast<device const ushort*>(blk + byte_offset + 2u);\n\
              \        acc%d_%d += scale * (\n\
              \            a0_lo * float(p_lo & 15u) +\n\
              \            a0_hi * float((p_lo >> 4u) & 15u) +\n\
              \            a1_lo * float((p_lo >> 8u) & 15u) +\n\
              \            a1_hi * float((p_lo >> 12u) & 15u) +\n\
              \            a2_lo * float(p_hi & 15u) +\n\
              \            a2_hi * float((p_hi >> 4u) & 15u) +\n\
              \            a3_lo * float((p_hi >> 8u) & 15u) +\n\
              \            a3_hi * float((p_hi >> 12u) & 15u) - a_sum8);\n\
              \      }"
              s j scale_offset s j)
        |> String.concat "\n")
    |> String.concat "\n\n"
  in
  let simd_sums =
    List.init num_slots (fun s ->
        List.init 8 (fun j ->
            Printf.sprintf "  acc%d_%d = simd_sum(acc%d_%d);" s j s j)
        |> String.concat "\n")
    |> String.concat "\n\n"
  in
  let in_bufs =
    if num_aux = 0 then ""
    else
      all_aux_values
      |> List.mapi (fun k v ->
             let dtype_str =
               match Ir.Value.dtype v with
               | Ir.Dtype.Float16 -> "half"
               | _ -> "float"
             in
             Printf.sprintf
               "    device const %s* in_buf_%d = buf_%d + row * params.n;\n"
               dtype_str k k)
      |> String.concat ""
  in
  let value_var v =
    Printf.sprintf "in_buf_%d" (Ir.Value_id.to_int (Ir.Value.id v))
  in
  let export_writebacks =
    canonical_epilogues
    |> List.mapi (fun e canonical ->
        let out_row = Printf.sprintf "out_row_%d" e in
        List.init 8 (fun j ->
            let idx =
              if j = 0 then "base_col"
              else Printf.sprintf "(base_col + %du)" j
            in
            let expr =
              Applicative_expr.emit_msl_expr ~value_var
                ~slot_emitter:(fun s -> Printf.sprintf "acc%d_%d" s j)
                ~index_expr:idx canonical
            in
            if j = 0 then
              Printf.sprintf "    %s[base_col] = half(%s);" out_row expr
            else
              Printf.sprintf
                "    if (base_col + %du < params.n) %s[base_col + %du] = half(%s);"
                j out_row j expr)
        |> String.concat "\n")
    |> String.concat "\n\n"
  in
  let out_row_decls =
    List.init num_exports (fun e ->
        Printf.sprintf "    device half* out_row_%d = output_%d + row * params.n;" e e)
    |> String.concat "\n"
  in
  String.concat "\n"
    [ Printf.sprintf "kernel void %s(" name;
      "    device const half* input [[buffer(0)]],";
      weight_buffers;
      aux_buffers;
      output_buffers;
      Printf.sprintf "    constant QuantLinearParams& params [[buffer(%d)]]," buf_idx_params;
      "    uint gid [[thread_position_in_grid]],";
      "    uint tid [[thread_index_in_threadgroup]],";
      "    uint lane [[thread_index_in_simdgroup]]) {";
      "  const uint simd_idx = gid >> 5u;";
      "  const uint col_groups = (params.n + 7u) >> 3u;";
      "  const uint total_elements = params.m * col_groups;";
      "  if (simd_idx >= total_elements) return;";
      "";
      "  const uint row = simd_idx / col_groups;";
      "  const uint group_idx = simd_idx - row * col_groups;";
      "  const uint base_col = group_idx << 3u;";
      "";
      "  device const half* in_ptr = input + row * params.k;";
      "";
      Printf.sprintf "  const uint blocks = params.k / %du;" block_elements;
      Printf.sprintf "  const uint col_stride = blocks * %du;" block_bytes;
      weight_bases;
      "";
      accumulators;
      "";
      "  const uint cluster_blocks = (blocks + 7u) >> 3u;";
      "  const uint b_sub = lane >> 2u;";
      "  const uint t_in_b = lane & 3u;";
      Printf.sprintf "  const uint byte_offset = %du + (t_in_b << 2u);" quant_offset;
      "";
      "  for (uint c = 0u; c < cluster_blocks; ++c) {";
      "    const uint block_idx = (c << 3u) + b_sub;";
      "    if (block_idx < blocks) {";
      Printf.sprintf "      const uint blk_byte_off = block_idx * %du;" block_bytes;
      "      const uint l0 = t_in_b << 2u;";
      Printf.sprintf "      const uint k_idx = block_idx * %du + l0;" block_elements;
      "      const float a0_lo = float(in_ptr[k_idx]);";
      "      const float a0_hi = float(in_ptr[k_idx + 16u]);";
      "      const float a1_lo = float(in_ptr[k_idx + 1u]);";
      "      const float a1_hi = float(in_ptr[k_idx + 17u]);";
      "      const float a2_lo = float(in_ptr[k_idx + 2u]);";
      "      const float a2_hi = float(in_ptr[k_idx + 18u]);";
      "      const float a3_lo = float(in_ptr[k_idx + 3u]);";
      "      const float a3_hi = float(in_ptr[k_idx + 19u]);";
      "";
      Printf.sprintf "      const float a_sum8 = float(%d) * (a0_lo + a0_hi + a1_lo + a1_hi + a2_lo + a2_hi + a3_lo + a3_hi);" zero;
      "";
      cluster_block_accs;
      "    }";
      "  }";
      "";
      simd_sums;
      "";
      in_bufs;
      "  if (lane == 0u) {";
      out_row_decls;
      export_writebacks;
      "  }";
      "}" ]

let emit_q4_0_multi_export_linear_f16_x16 ~name ~epilogues ?(num_slots = 2)
    ~scale_offset ~quant_offset ~zero ~block_elements ~block_bytes () =
  let num_exports = List.length epilogues in
  let canonical_epilogues = List.map Applicative_expr.canonicalize epilogues in
  let all_aux_values =
    List.concat_map Applicative_expr.required_values canonical_epilogues
    |> List.fold_left
         (fun acc v ->
           if List.exists (fun v' -> Ir.Value.equal v v') acc then acc
           else acc @ [ v ])
         []
  in
  let num_aux = List.length all_aux_values in
  let first_output_buf = 1 + num_slots + num_aux in
  let buf_idx_params = first_output_buf + num_exports in
  let weight_buffers =
    List.init num_slots (fun s ->
        Printf.sprintf "    device const uchar* weight_%d [[buffer(%d)]]," s
          (1 + s))
    |> String.concat "\n"
  in
  let aux_buffers =
    if num_aux = 0 then ""
    else
      all_aux_values
      |> List.mapi (fun k v ->
             let dtype_str =
               match Ir.Value.dtype v with
               | Ir.Dtype.Float16 -> "half"
               | _ -> "float"
             in
             Printf.sprintf "    device const %s* buf_%d [[buffer(%d)]],\n"
               dtype_str k (1 + num_slots + k))
      |> String.concat ""
  in
  let output_buffers =
    List.init num_exports (fun e ->
        Printf.sprintf "    device half* output_%d [[buffer(%d)]]," e
          (first_output_buf + e))
    |> String.concat "\n"
  in
  let weight_bases =
    List.init num_slots (fun s ->
        let rows =
          List.init 16 (fun j ->
              if j = 0 then
                Printf.sprintf
                  "  device const uchar* row_%d_0 = base_%d + base_col * col_stride;"
                  s s
              else
                Printf.sprintf
                  "  device const uchar* row_%d_%d = base_%d + min(base_col + %du, params.n - 1u) * col_stride;"
                  s j s j)
          |> String.concat "\n"
        in
        Printf.sprintf "  device const uchar* base_%d = weight_%d;\n%s" s s rows)
    |> String.concat "\n\n"
  in
  let accumulators =
    List.init num_slots (fun s ->
        let acc_decls =
          List.init 16 (fun j -> Printf.sprintf "acc%d_%d = 0.0f" s j)
          |> String.concat ", "
        in
        Printf.sprintf "  float %s;" acc_decls)
    |> String.concat "\n"
  in
  let oct_accumulates =
    List.init num_slots (fun s ->
        List.init 16 (fun j ->
            Printf.sprintf "    Q4_0_OCT_ACCUMULATE(acc%d_%d, row_%d_%d);" s j s j)
        |> String.concat "\n")
    |> String.concat "\n"
  in
  let rem_accumulates =
    List.init num_slots (fun s ->
        List.init 16 (fun j ->
            Printf.sprintf "    Q4_0_PAIR_ACCUMULATE(acc%d_%d, row_%d_%d);" s j s j)
        |> String.concat "\n")
    |> String.concat "\n"
  in
  let odd_accumulates =
    List.init num_slots (fun s ->
        List.init 16 (fun j ->
            Printf.sprintf
              "    acc%d_%d += a * llmopt_schema_q4_0_value(row_%d_%d + blk_byte_off, lane);"
              s j s j)
        |> String.concat "\n")
    |> String.concat "\n"
  in
  let simd_sums =
    List.init num_slots (fun s ->
        List.init 16 (fun j ->
            Printf.sprintf "  acc%d_%d = simd_sum(acc%d_%d);" s j s j)
        |> String.concat "\n")
    |> String.concat "\n\n"
  in
  let in_bufs =
    if num_aux = 0 then ""
    else
      all_aux_values
      |> List.mapi (fun k v ->
             let dtype_str =
               match Ir.Value.dtype v with
               | Ir.Dtype.Float16 -> "half"
               | _ -> "float"
             in
             Printf.sprintf
               "    device const %s* in_buf_%d = buf_%d + row * params.n;\n"
               dtype_str k k)
      |> String.concat ""
  in
  let value_var v =
    Printf.sprintf "in_buf_%d" (Ir.Value_id.to_int (Ir.Value.id v))
  in
  let export_writebacks =
    canonical_epilogues
    |> List.mapi (fun e canonical ->
        let out_row = Printf.sprintf "out_row_%d" e in
        List.init 16 (fun j ->
            let idx =
              if j = 0 then "base_col"
              else Printf.sprintf "(base_col + %du)" j
            in
            let expr =
              Applicative_expr.emit_msl_expr ~value_var
                ~slot_emitter:(fun s -> Printf.sprintf "acc%d_%d" s j)
                ~index_expr:idx canonical
            in
            if j = 0 then
              Printf.sprintf "    %s[base_col] = half(%s);" out_row expr
            else
              Printf.sprintf
                "    if (base_col + %du < params.n) %s[base_col + %du] = half(%s);"
                j out_row j expr)
        |> String.concat "\n")
    |> String.concat "\n\n"
  in
  let out_row_decls =
    List.init num_exports (fun e ->
        Printf.sprintf "    device half* out_row_%d = output_%d + row * params.n;" e e)
    |> String.concat "\n"
  in
  String.concat "\n"
    [ Printf.sprintf {|
#define Q4_0_OCT_ACCUMULATE(col_acc, row_ptr) \
  { \
    device const uchar* blk0 = (row_ptr) + blk_byte_off; \
    device const uchar* blk1 = blk0 + 36u; \
    device const uchar* blk2 = blk0 + 72u; \
    device const uchar* blk3 = blk0 + 108u; \
    const half s0 = *reinterpret_cast<device const half*>(blk0 + %du); \
    const half s1 = *reinterpret_cast<device const half*>(blk1 + %du); \
    const half s2 = *reinterpret_cast<device const half*>(blk2 + %du); \
    const half s3 = *reinterpret_cast<device const half*>(blk3 + %du); \
    const uchar p0 = blk0[%du + l]; \
    const uchar p1 = blk1[%du + l]; \
    const uchar p2 = blk2[%du + l]; \
    const uchar p3 = blk3[%du + l]; \
    const float q0_l = float(int(p0 & 15u) - %d); \
    const float q0_h = float(int(p0 >> 4u) - %d); \
    const float q1_l = float(int(p1 & 15u) - %d); \
    const float q1_h = float(int(p1 >> 4u) - %d); \
    const float q2_l = float(int(p2 & 15u) - %d); \
    const float q2_h = float(int(p2 >> 4u) - %d); \
    const float q3_l = float(int(p3 & 15u) - %d); \
    const float q3_h = float(int(p3 >> 4u) - %d); \
    (col_acc) += float(s0) * (a0_low * q0_l + a0_high * q0_h) \
               + float(s1) * (a1_low * q1_l + a1_high * q1_h) \
               + float(s2) * (a2_low * q2_l + a2_high * q2_h) \
               + float(s3) * (a3_low * q3_l + a3_high * q3_h); \
  }

#define Q4_0_PAIR_ACCUMULATE(col_acc, row_ptr) \
  { \
    device const uchar* blk = (row_ptr) + blk_byte_off; \
    const half s = *reinterpret_cast<device const half*>(blk + %du); \
    const uchar p = blk[%du + l]; \
    const float q_l = float(int(p & 15u) - %d); \
    const float q_h = float(int(p >> 4u) - %d); \
    (col_acc) += float(s) * (a_low * q_l + a_high * q_h); \
  }
|}
        scale_offset scale_offset scale_offset scale_offset
        quant_offset quant_offset quant_offset quant_offset
        zero zero zero zero zero zero zero zero
        scale_offset quant_offset zero zero;
      Printf.sprintf "kernel void %s(" name;
      "    device const half* input [[buffer(0)]],";
      weight_buffers;
      aux_buffers;
      output_buffers;
      Printf.sprintf "    constant QuantLinearParams& params [[buffer(%d)]]," buf_idx_params;
      "    uint gid [[thread_position_in_grid]],";
      "    uint tid [[thread_index_in_threadgroup]],";
      "    uint lane [[thread_index_in_simdgroup]]) {";
      "  const uint simd_idx = gid >> 5u;";
      "  const uint col_groups = (params.n + 15u) >> 4u;";
      "  const uint total_elements = params.m * col_groups;";
      "  if (simd_idx >= total_elements) return;";
      "";
      "  const uint row = simd_idx / col_groups;";
      "  const uint group_idx = simd_idx - row * col_groups;";
      "  const uint base_col = group_idx << 4u;";
      "";
      "  device const half* in_ptr = input + row * params.k;";
      "";
      Printf.sprintf "  const uint blocks = params.k / %du;" block_elements;
      Printf.sprintf "  const uint col_stride = blocks * %du;" block_bytes;
      weight_bases;
      "";
      accumulators;
      "";
      "  const uint oct_blocks = blocks >> 3u;";
      "  const uint l = lane & 15u;";
      "  const uint b_offset = lane >> 4u;";
      "";
      "  for (uint b = 0u; b < oct_blocks; ++b) {";
      "    const uint block_idx = (b << 3u) + b_offset;";
      Printf.sprintf "    const uint blk_byte_off = block_idx * %du;" block_bytes;
      "";
      Printf.sprintf "    const uint k_idx0 = block_idx * %du + l;" block_elements;
      "    const float a0_low = float(in_ptr[k_idx0]);";
      "    const float a0_high = float(in_ptr[k_idx0 + 16u]);";
      "    const float a1_low = float(in_ptr[k_idx0 + 64u]);";
      "    const float a1_high = float(in_ptr[k_idx0 + 80u]);";
      "    const float a2_low = float(in_ptr[k_idx0 + 128u]);";
      "    const float a2_high = float(in_ptr[k_idx0 + 144u]);";
      "    const float a3_low = float(in_ptr[k_idx0 + 192u]);";
      "    const float a3_high = float(in_ptr[k_idx0 + 208u]);";
      "";
      oct_accumulates;
      "  }";
      "";
      "  const uint rem_start = oct_blocks << 3u;";
      "  const uint rem_half = (blocks - rem_start) >> 1u;";
      "  for (uint b = 0u; b < rem_half; ++b) {";
      "    const uint block_idx = rem_start + (b << 1u) + b_offset;";
      Printf.sprintf "    const uint blk_byte_off = block_idx * %du;" block_bytes;
      Printf.sprintf "    const uint k_idx = block_idx * %du + l;" block_elements;
      "    const float a_low = float(in_ptr[k_idx]);";
      "    const float a_high = float(in_ptr[k_idx + 16u]);";
      rem_accumulates;
      "  }";
      "";
      "  if ((blocks & 1u) != 0u) {";
      "    const uint block_idx = blocks - 1u;";
      Printf.sprintf "    const uint blk_byte_off = block_idx * %du;" block_bytes;
      Printf.sprintf "    const uint k_idx = block_idx * %du + lane;" block_elements;
      "    const float a = float(in_ptr[k_idx]);";
      odd_accumulates;
      "  }";
      "";
      simd_sums;
      "";
      in_bufs;
      "  if (lane == 0u) {";
      out_row_decls;
      export_writebacks;
      "  }";
      "}";
      "#undef Q4_0_OCT_ACCUMULATE";
      "#undef Q4_0_PAIR_ACCUMULATE" ]
