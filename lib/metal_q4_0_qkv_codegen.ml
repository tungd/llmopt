let emit_q4_0_qkv_linear_f16_x8 ~scale_offset ~quant_offset ~zero ~block_elements
    ~block_bytes =
  Printf.sprintf {|
kernel void llmopt_q4_0_qkv_linear_f16_x8(
    device const half* input [[buffer(0)]],
    device const uchar* q_weight [[buffer(1)]],
    device const uchar* k_weight [[buffer(2)]],
    device const uchar* v_weight [[buffer(3)]],
    device half* q_output [[buffer(4)]],
    device half* k_output [[buffer(5)]],
    device half* v_output [[buffer(6)]],
    constant QuantQKVParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint q_groups = (params.n_q + 7u) >> 3u;
  const uint k_groups = (params.n_k + 7u) >> 3u;
  const uint v_groups = (params.n_v + 7u) >> 3u;
  const uint total_groups = q_groups + k_groups + v_groups;
  const uint total_elements = params.m * total_groups;
  if (simd_idx >= total_elements) return;

  const uint row = simd_idx / total_groups;
  const uint group_idx = simd_idx - row * total_groups;

  device const half* activation = input + row * params.k;

  uint base_col;
  device const uchar* weight_ptr;
  device half* output;
  uint stride;
  uint limit;
  if (group_idx < q_groups) {
    base_col = group_idx << 3u;
    weight_ptr = q_weight;
    output = q_output;
    stride = params.n_q;
    limit = params.n_q;
  } else if (group_idx < q_groups + k_groups) {
    base_col = (group_idx - q_groups) << 3u;
    weight_ptr = k_weight;
    output = k_output;
    stride = params.n_k;
    limit = params.n_k;
  } else {
    base_col = (group_idx - q_groups - k_groups) << 3u;
    weight_ptr = v_weight;
    output = v_output;
    stride = params.n_v;
    limit = params.n_v;
  }

  const uint blocks = params.k / %du;
  const uint col_stride = blocks * %du;
  device const uchar* row_blocks0 = weight_ptr + base_col * col_stride;
  device const uchar* row_blocks1 = weight_ptr + min(base_col + 1u, limit - 1u) * col_stride;
  device const uchar* row_blocks2 = weight_ptr + min(base_col + 2u, limit - 1u) * col_stride;
  device const uchar* row_blocks3 = weight_ptr + min(base_col + 3u, limit - 1u) * col_stride;
  device const uchar* row_blocks4 = weight_ptr + min(base_col + 4u, limit - 1u) * col_stride;
  device const uchar* row_blocks5 = weight_ptr + min(base_col + 5u, limit - 1u) * col_stride;
  device const uchar* row_blocks6 = weight_ptr + min(base_col + 6u, limit - 1u) * col_stride;
  device const uchar* row_blocks7 = weight_ptr + min(base_col + 7u, limit - 1u) * col_stride;

  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
  float accumulator2 = 0.0f;
  float accumulator3 = 0.0f;
  float accumulator4 = 0.0f;
  float accumulator5 = 0.0f;
  float accumulator6 = 0.0f;
  float accumulator7 = 0.0f;

  const uint half_blocks = blocks >> 1u;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;

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

    const uint8_t byte0 = (lane < 16u) ? blk0[%du + lane] : blk0[%du + lane - 16u];
    const uint8_t byte1 = (lane < 16u) ? blk1[%du + lane] : blk1[%du + lane - 16u];
    const uint8_t byte2 = (lane < 16u) ? blk2[%du + lane] : blk2[%du + lane - 16u];
    const uint8_t byte3 = (lane < 16u) ? blk3[%du + lane] : blk3[%du + lane - 16u];
    const uint8_t byte4 = (lane < 16u) ? blk4[%du + lane] : blk4[%du + lane - 16u];
    const uint8_t byte5 = (lane < 16u) ? blk5[%du + lane] : blk5[%du + lane - 16u];
    const uint8_t byte6 = (lane < 16u) ? blk6[%du + lane] : blk6[%du + lane - 16u];
    const uint8_t byte7 = (lane < 16u) ? blk7[%du + lane] : blk7[%du + lane - 16u];

    const int q0 = int((lane < 16u) ? (byte0 & 0x0Fu) : (byte0 >> 4)) - %d;
    const int q1 = int((lane < 16u) ? (byte1 & 0x0Fu) : (byte1 >> 4)) - %d;
    const int q2 = int((lane < 16u) ? (byte2 & 0x0Fu) : (byte2 >> 4)) - %d;
    const int q3 = int((lane < 16u) ? (byte3 & 0x0Fu) : (byte3 >> 4)) - %d;
    const int q4 = int((lane < 16u) ? (byte4 & 0x0Fu) : (byte4 >> 4)) - %d;
    const int q5 = int((lane < 16u) ? (byte5 & 0x0Fu) : (byte5 >> 4)) - %d;
    const int q6 = int((lane < 16u) ? (byte6 & 0x0Fu) : (byte6 >> 4)) - %d;
    const int q7 = int((lane < 16u) ? (byte7 & 0x0Fu) : (byte7 >> 4)) - %d;

    const float a = float(activation[block_idx * %du + lane]);
    accumulator0 += float(scale0) * float(q0) * a;
    accumulator1 += float(scale1) * float(q1) * a;
    accumulator2 += float(scale2) * float(q2) * a;
    accumulator3 += float(scale3) * float(q3) * a;
    accumulator4 += float(scale4) * float(q4) * a;
    accumulator5 += float(scale5) * float(q5) * a;
    accumulator6 += float(scale6) * float(q6) * a;
    accumulator7 += float(scale7) * float(q7) * a;
  }

  accumulator0 = simd_sum(accumulator0);
  accumulator1 = simd_sum(accumulator1);
  accumulator2 = simd_sum(accumulator2);
  accumulator3 = simd_sum(accumulator3);
  accumulator4 = simd_sum(accumulator4);
  accumulator5 = simd_sum(accumulator5);
  accumulator6 = simd_sum(accumulator6);
  accumulator7 = simd_sum(accumulator7);

  if (lane == 0u) {
    device half* out_row = output + row * stride;
    out_row[base_col] = half(accumulator0);
    if (base_col + 1u < limit) out_row[base_col + 1u] = half(accumulator1);
    if (base_col + 2u < limit) out_row[base_col + 2u] = half(accumulator2);
    if (base_col + 3u < limit) out_row[base_col + 3u] = half(accumulator3);
    if (base_col + 4u < limit) out_row[base_col + 4u] = half(accumulator4);
    if (base_col + 5u < limit) out_row[base_col + 5u] = half(accumulator5);
    if (base_col + 6u < limit) out_row[base_col + 6u] = half(accumulator6);
    if (base_col + 7u < limit) out_row[base_col + 7u] = half(accumulator7);
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
    block_bytes
    scale_offset scale_offset scale_offset scale_offset
    scale_offset scale_offset scale_offset scale_offset
    quant_offset quant_offset quant_offset quant_offset
    quant_offset quant_offset quant_offset quant_offset
    quant_offset quant_offset quant_offset quant_offset
    quant_offset quant_offset quant_offset quant_offset
    zero zero zero zero zero zero zero zero
    block_elements

let emit_q4_0_qkv_linear_f16_x16 ~scale_offset ~quant_offset ~zero ~block_elements
    ~block_bytes =
  let row_blocks_decls =
    List.init 16 (fun j ->
        if j = 0 then
          "  device const uchar* row_blocks0 = weight_ptr + base_col * col_stride;"
        else
          Printf.sprintf
            "  device const uchar* row_blocks%d = weight_ptr + min(base_col + %du, limit - 1u) * col_stride;"
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
  let stores =
    List.init 16 (fun j ->
        if j = 0 then "    out_row[base_col] = half(accumulator0);"
        else
          Printf.sprintf
            "    if (base_col + %du < limit) out_row[base_col + %du] = half(accumulator%d);"
            j j j)
    |> String.concat "\n"
  in
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

kernel void llmopt_q4_0_qkv_linear_f16_x16(
    device const half* input [[buffer(0)]],
    device const uchar* q_weight [[buffer(1)]],
    device const uchar* k_weight [[buffer(2)]],
    device const uchar* v_weight [[buffer(3)]],
    device half* q_output [[buffer(4)]],
    device half* k_output [[buffer(5)]],
    device half* v_output [[buffer(6)]],
    constant QuantQKVParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint q_groups = (params.n_q + 15u) >> 4u;
  const uint k_groups = (params.n_k + 15u) >> 4u;
  const uint v_groups = (params.n_v + 15u) >> 4u;
  const uint total_groups = q_groups + k_groups + v_groups;
  const uint total_elements = params.m * total_groups;
  if (simd_idx >= total_elements) return;

  const uint row = simd_idx / total_groups;
  const uint group_idx = simd_idx - row * total_groups;

  device const half* activation = input + row * params.k;

  uint base_col;
  device const uchar* weight_ptr;
  device half* output;
  uint stride;
  uint limit;
  if (group_idx < q_groups) {
    base_col = group_idx << 4u;
    weight_ptr = q_weight;
    output = q_output;
    stride = params.n_q;
    limit = params.n_q;
  } else if (group_idx < q_groups + k_groups) {
    base_col = (group_idx - q_groups) << 4u;
    weight_ptr = k_weight;
    output = k_output;
    stride = params.n_k;
    limit = params.n_k;
  } else {
    base_col = (group_idx - q_groups - k_groups) << 4u;
    weight_ptr = v_weight;
    output = v_output;
    stride = params.n_v;
    limit = params.n_v;
  }

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

  if (lane == 0u) {
    device half* out_row = output + row * stride;
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
    block_elements block_bytes
    row_blocks_decls accumulators_decls
    block_bytes block_elements oct_accumulates
    block_bytes block_elements rem_accumulates
    block_bytes block_elements odd_accumulates
    simd_sums stores

let emit_q4_0_qkv_linear_f16_m32 ~scale_offset ~quant_offset ~zero ~block_elements
    ~block_bytes =
  Printf.sprintf {|
kernel void llmopt_q4_0_qkv_linear_f16_m32(
    device const half* input [[buffer(0)]],
    device const uchar* q_weight [[buffer(1)]],
    device const uchar* k_weight [[buffer(2)]],
    device const uchar* v_weight [[buffer(3)]],
    device half* q_output [[buffer(4)]],
    device half* k_output [[buffer(5)]],
    device half* v_output [[buffer(6)]],
    constant QuantQKVParams& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint total_cols = params.n_q + params.n_k + params.n_v;
  const uint m_tiles = (params.m + 31u) >> 5u;
  const uint total_tiles = m_tiles * total_cols;
  const uint tile_idx = simd_idx >> 1u;
  const uint simd_in_tile = simd_idx & 1u;
  if (tile_idx >= total_tiles) return;
  const uint m_tile = tile_idx / total_cols;
  const uint col_idx = tile_idx - m_tile * total_cols;
  const uint row_base = m_tile << 5u;
  const uint sub_row_base = row_base + (simd_in_tile << 4u);

  uint col;
  device const uchar* weight_ptr;
  device half* out_ptr;
  uint out_stride;
  if (col_idx < params.n_q) {
    col = col_idx;
    out_stride = params.n_q;
    weight_ptr = q_weight;
    out_ptr = q_output;
  } else if (col_idx < params.n_q + params.n_k) {
    col = col_idx - params.n_q;
    out_stride = params.n_k;
    weight_ptr = k_weight;
    out_ptr = k_output;
  } else {
    col = col_idx - (params.n_q + params.n_k);
    out_stride = params.n_v;
    weight_ptr = v_weight;
    out_ptr = v_output;
  }

  const uint blocks = params.k / %du;
  device const uchar* row_blocks = weight_ptr + col * blocks * %du;

  device const half* in_ptrs[16];
  #pragma unroll
  for (uint r = 0; r < 16; ++r) {
    in_ptrs[r] = (sub_row_base + r < params.m) ? input + (sub_row_base + r) * params.k : input;
  }

  float accumulator[16] = {0.0f};

  const uint half_blocks = blocks >> 1u;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;

  for (uint b = 0u; b < half_blocks; ++b) {
    const uint block_idx = (b << 1u) + b_offset;
    device const uchar* blk = row_blocks + block_idx * %du;
    const half scale = *reinterpret_cast<device const half*>(blk + %du);
    const uchar packed = blk[%du + l];
    const float s = float(scale);
    const uint k = block_idx * %du + l;
    const float q_low = float(int(packed & 15u) - %d);
    const float q_high = float(int(packed >> 4u) - %d);
    const float w_low = s * q_low;
    const float w_high = s * q_high;

    #pragma unroll
    for (uint r = 0; r < 16; ++r) {
      accumulator[r] += float(in_ptrs[r][k]) * w_low + float(in_ptrs[r][k + 16u]) * w_high;
    }
  }

  if ((blocks & 1u) != 0u) {
    const uint block_idx = blocks - 1u;
    device const uchar* blk = row_blocks + block_idx * %du;
    const half scale = *reinterpret_cast<device const half*>(blk + %du);
    const uint8_t byte_val = (lane < 16u) ? blk[%du + lane] : blk[%du + lane - 16u];
    const uint8_t nib = (lane < 16u) ? (byte_val & 0x0Fu) : (byte_val >> 4u);
    const float decoded = float(scale) * float(int(nib) - %d);
    const uint k = block_idx * %du + lane;
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
      if (sub_row_base + r < params.m) {
        out_ptr[(sub_row_base + r) * out_stride + col] = half(accumulator[r]);
      }
    }
  }
}
|}
    block_elements block_bytes block_bytes scale_offset quant_offset
    block_elements zero zero
    block_bytes scale_offset quant_offset quant_offset
    zero block_elements
