let ( let* ) = Result.bind

let field_offset layout id =
  match Block_schema.Physical_layout.field layout id with
  | Some field -> Ok (Block_schema.Physical_layout.Field.offset field)
  | None -> Error "block-quant schema is missing a required physical field"

let offset_zero = function
  | Block_schema.Physical_layout.Signedness.Offset value -> Ok value
  | signedness ->
      Error
        (Printf.sprintf "constructive nibble decode does not support %s"
           (Block_schema.Physical_layout.Signedness.to_string signedness))

let require_block32 ~target physical label =
  let block_elements = Block_schema.Physical_layout.block_elements physical in
  let simd_lanes = target.Target_hardware.memory.simd_lanes in
  if block_elements = 32 && simd_lanes = 32 then Ok ()
  else
    Error
      (Printf.sprintf
         "constructive %s Metal lowering requires block=32 and SIMD=32 (got block=%d SIMD=%d)"
         label block_elements simd_lanes)

let emit_linear_pair ~name ~value_function ~block_elements ~block_bytes =
  Printf.sprintf {|
kernel void llmopt_%s_linear_f16(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint output_index = gid >> 5u;
  const uint output_elements = params.m * params.n;
  if (output_index >= output_elements) return;
  const uint row = output_index / params.n;
  const uint column = output_index - row * params.n;
  const uint blocks = params.k / %du;
  device const uchar* row_blocks = weight + column * blocks * %du;
  device const half* activation = input + row * params.k;
  float accumulator = 0.0f;
  for (uint block = 0; block < blocks; ++block) {
    const uint k = block * %du + lane;
    accumulator += float(activation[k])
        * %s(row_blocks + block * %du, lane);
  }
  accumulator = simd_sum(accumulator);
  if (lane == 0u) {
    if (params.has_bias != 0u) accumulator += float(bias[column]);
    output[output_index] = half(accumulator);
  }
}

kernel void llmopt_%s_linear_f16_m2(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint column = gid >> 5u;
  if (column >= params.n) return;
  const uint blocks = params.k / %du;
  device const uchar* row_blocks = weight + column * blocks * %du;
  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
  for (uint block = 0; block < blocks; ++block) {
    const uint k = block * %du + lane;
    const float decoded = %s(row_blocks + block * %du, lane);
    accumulator0 += float(input[k]) * decoded;
    accumulator1 += float(input[params.k + k]) * decoded;
  }
  accumulator0 = simd_sum(accumulator0);
  accumulator1 = simd_sum(accumulator1);
  if (lane == 0u) {
    const float bias_value =
        params.has_bias == 0u ? 0.0f : float(bias[column]);
    output[column] = half(accumulator0 + bias_value);
    output[params.n + column] = half(accumulator1 + bias_value);
  }
}
|}
    name block_elements block_bytes block_elements value_function block_bytes
    name block_elements block_bytes block_elements value_function block_bytes

let emit_superblock_linear_pair ~name ~value_function ~block_elements
    ~block_bytes ~subblock_elements =
  let subblocks = block_elements / subblock_elements in
  Printf.sprintf {|
kernel void llmopt_%s_linear_f16(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint output_index = gid >> 5u;
  const uint output_elements = params.m * params.n;
  if (output_index >= output_elements) return;
  const uint row = output_index / params.n;
  const uint column = output_index - row * params.n;
  const uint blocks = params.k / %du;
  device const uchar* row_blocks = weight + column * blocks * %du;
  device const half* activation = input + row * params.k;
  float accumulator = 0.0f;
  for (uint block = 0; block < blocks; ++block) {
    device const uchar* packed = row_blocks + block * %du;
    for (uint subblock = 0; subblock < %du; ++subblock) {
      const uint k = block * %du + subblock * %du + lane;
      accumulator += float(activation[k])
          * %s(packed, lane, subblock);
    }
  }
  accumulator = simd_sum(accumulator);
  if (lane == 0u) {
    if (params.has_bias != 0u) accumulator += float(bias[column]);
    output[output_index] = half(accumulator);
  }
}

kernel void llmopt_%s_linear_f16_m2(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint column = gid >> 5u;
  if (column >= params.n) return;
  const uint blocks = params.k / %du;
  device const uchar* row_blocks = weight + column * blocks * %du;
  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
  for (uint block = 0; block < blocks; ++block) {
    device const uchar* packed = row_blocks + block * %du;
    for (uint subblock = 0; subblock < %du; ++subblock) {
      const uint k = block * %du + subblock * %du + lane;
      const float decoded = %s(packed, lane, subblock);
      accumulator0 += float(input[k]) * decoded;
      accumulator1 += float(input[params.k + k]) * decoded;
    }
  }
  accumulator0 = simd_sum(accumulator0);
  accumulator1 = simd_sum(accumulator1);
  if (lane == 0u) {
    const float bias_value =
        params.has_bias == 0u ? 0.0f : float(bias[column]);
    output[column] = half(accumulator0 + bias_value);
    output[params.n + column] = half(accumulator1 + bias_value);
  }
}
|}
    name block_elements block_bytes block_bytes subblocks block_elements
    subblock_elements value_function name block_elements block_bytes block_bytes
    subblocks block_elements subblock_elements value_function

let require_superblock ~target physical logical label =
  let block_elements = Block_schema.Physical_layout.block_elements physical in
  let subblock_elements =
    Block_schema.Logical_decode.subblock_elements logical
  in
  let simd_lanes = target.Target_hardware.memory.simd_lanes in
  if block_elements = 256 && subblock_elements = 32 && simd_lanes = 32 then
    Ok ()
  else
    Error
      (Printf.sprintf
         "constructive %s Metal lowering requires block=256, subblock=32, SIMD=32 (got block=%d subblock=%d SIMD=%d)"
         label block_elements subblock_elements simd_lanes)

let emit_superblock schema ~name ~value_function value_source =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  value_source
  ^ emit_superblock_linear_pair ~name ~value_function
      ~block_elements:(Block_schema.Physical_layout.block_elements physical)
      ~block_bytes:(Block_schema.Physical_layout.block_bytes physical)
      ~subblock_elements:(Block_schema.Logical_decode.subblock_elements logical)

let q4_k_linear ~target schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* scale_offset = field_offset physical Block_schema.Physical_layout.Field.Block_scale in
  let* minimum_offset = field_offset physical Block_schema.Physical_layout.Field.Block_minimum in
  let* table_offset = field_offset physical Block_schema.Physical_layout.Field.Scale_minimum_table in
  let* quant_offset = field_offset physical Block_schema.Physical_layout.Field.Quantized in
  let* () = require_superblock ~target physical logical "Q4_K" in
  if Block_schema.Logical_decode.code_bits equation <> 4 then
    Error "constructive Q4_K emitter requires four-bit codes"
  else
    let value_function = "llmopt_schema_q4_k_value" in
    Ok
      (emit_superblock schema ~name:"q4_k" ~value_function
         (Printf.sprintf {|
inline float %s(device const uchar* block, uint lane, uint subblock) {
  const float d = float(*reinterpret_cast<device const half*>(block + %du));
  const float dmin = float(*reinterpret_cast<device const half*>(block + %du));
  device const uchar* table = block + %du;
  const uint j = subblock & 3u;
  const uint scale = subblock < 4u
      ? (table[j] & 63u)
      : ((table[j + 8u] & 15u) | ((table[j] >> 6u) << 4u));
  const uint minimum = subblock < 4u
      ? (table[j + 4u] & 63u)
      : ((table[j + 8u] >> 4u) | ((table[j + 4u] >> 6u) << 4u));
  const uchar packed = block[%du + (subblock >> 1u) * 32u + lane];
  const uint code = (subblock & 1u) == 0u ? (packed & 15u) : (packed >> 4u);
  return d * float(scale) * float(code) - dmin * float(minimum);
}
|} value_function scale_offset minimum_offset table_offset quant_offset))

let q5_k_linear ~target schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* scale_offset = field_offset physical Block_schema.Physical_layout.Field.Block_scale in
  let* minimum_offset = field_offset physical Block_schema.Physical_layout.Field.Block_minimum in
  let* table_offset = field_offset physical Block_schema.Physical_layout.Field.Scale_minimum_table in
  let* high_offset = field_offset physical Block_schema.Physical_layout.Field.Quantized_high in
  let* low_offset = field_offset physical Block_schema.Physical_layout.Field.Quantized_low in
  let* () = require_superblock ~target physical logical "Q5_K" in
  if Block_schema.Logical_decode.code_bits equation <> 5 then
    Error "constructive Q5_K emitter requires five-bit codes"
  else
    let value_function = "llmopt_schema_q5_k_value" in
    Ok
      (emit_superblock schema ~name:"q5_k" ~value_function
         (Printf.sprintf {|
inline float %s(device const uchar* block, uint lane, uint subblock) {
  const float d = float(*reinterpret_cast<device const half*>(block + %du));
  const float dmin = float(*reinterpret_cast<device const half*>(block + %du));
  device const uchar* table = block + %du;
  const uint j = subblock & 3u;
  const uint scale = subblock < 4u
      ? (table[j] & 63u)
      : ((table[j + 8u] & 15u) | ((table[j] >> 6u) << 4u));
  const uint minimum = subblock < 4u
      ? (table[j + 4u] & 63u)
      : ((table[j + 8u] >> 4u) | ((table[j + 4u] >> 6u) << 4u));
  const uchar packed = block[%du + (subblock >> 1u) * 32u + lane];
  const uint nibble = (subblock & 1u) == 0u ? (packed & 15u) : (packed >> 4u);
  const uint high = (block[%du + lane] >> subblock) & 1u;
  return d * float(scale) * float(nibble | (high << 4u))
      - dmin * float(minimum);
}
|} value_function scale_offset minimum_offset table_offset low_offset high_offset))

let q6_k_linear ~target schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* zero = offset_zero (Block_schema.Logical_decode.code_signedness equation) in
  let* low_offset = field_offset physical Block_schema.Physical_layout.Field.Quantized_low in
  let* high_offset = field_offset physical Block_schema.Physical_layout.Field.Quantized_high in
  let* table_offset = field_offset physical Block_schema.Physical_layout.Field.Scale_table in
  let* scale_offset = field_offset physical Block_schema.Physical_layout.Field.Block_scale in
  let* () = require_superblock ~target physical logical "Q6_K" in
  if Block_schema.Logical_decode.code_bits equation <> 6 then
    Error "constructive Q6_K emitter requires six-bit codes"
  else
    let value_function = "llmopt_schema_q6_k_value" in
    Ok
      (emit_superblock schema ~name:"q6_k" ~value_function
         (Printf.sprintf {|
inline float %s(device const uchar* block, uint lane, uint subblock) {
  const uint half_block = subblock >> 2u;
  const uint within = subblock & 3u;
  const uint low_index = half_block * 64u + (within & 1u) * 32u + lane;
  const uint low_shift = (within >> 1u) * 4u;
  const uint high_index = half_block * 32u + lane;
  const uint high_shift = within * 2u;
  const uint code = ((block[%du + low_index] >> low_shift) & 15u)
      | (((block[%du + high_index] >> high_shift) & 3u) << 4u);
  const uint scale_index = half_block * 8u + (lane >> 4u) + within * 2u;
  const char subscale = *reinterpret_cast<device const char*>(block + %du + scale_index);
  const float scale = float(*reinterpret_cast<device const half*>(block + %du));
  return scale * float(subscale) * float(int(code) - %d);
}
|} value_function low_offset high_offset table_offset scale_offset zero))

let iq4_xs_linear ~target schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* scale_offset = field_offset physical Block_schema.Physical_layout.Field.Block_scale in
  let* high_offset = field_offset physical Block_schema.Physical_layout.Field.Codebook_scale_high in
  let* low_offset = field_offset physical Block_schema.Physical_layout.Field.Codebook_scale_low in
  let* quant_offset = field_offset physical Block_schema.Physical_layout.Field.Quantized in
  let* () = require_superblock ~target physical logical "IQ4_XS" in
  match equation, Block_schema.Logical_decode.scale equation with
  | Block_schema.Logical_decode.Codebook { values; code_bits = 4; _ },
    Block_schema.Logical_decode.Product
      [ Block_schema.Logical_decode.Per_block _;
        Block_schema.Logical_decode.Packed_power_of_two { bias; denominator; _ } ] ->
      let values = values |> List.map string_of_int |> String.concat ", " in
      let value_function = "llmopt_schema_iq4_xs_value" in
      Ok
        (emit_superblock schema ~name:"iq4_xs" ~value_function
           (Printf.sprintf {|
constant int llmopt_schema_iq4_xs_values[16] = { %s };
inline float %s(device const uchar* block, uint lane, uint subblock) {
  const float d = float(*reinterpret_cast<device const half*>(block + %du));
  const ushort high = *reinterpret_cast<device const ushort*>(block + %du);
  const uchar low_byte = block[%du + (subblock >> 1u)];
  const uint low = (low_byte >> (4u * (subblock & 1u))) & 15u;
  const uint high_bits = (high >> (2u * subblock)) & 3u;
  const int exponent = int(low | (high_bits << 4u)) - %d;
  const uchar packed = block[%du + (subblock >> 1u) * 32u + lane];
  const uint code = (subblock & 1u) == 0u ? (packed & 15u) : (packed >> 4u);
  return d * exp2(float(exponent) / float(%d))
      * float(llmopt_schema_iq4_xs_values[code]);
}
|} values value_function scale_offset high_offset low_offset bias quant_offset
              denominator))
  | _ -> Error "constructive IQ4_XS emitter requires the codebook power-of-two schema"

let q8_0_linear ~target schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* scale_offset =
    field_offset physical Block_schema.Physical_layout.Field.Block_scale
  in
  let* quant_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized
  in
  let* () = require_block32 ~target physical "Q8_0" in
  if Block_schema.quant_type schema <> Some Ir.Dtype.Q8_0 then
    Error "constructive Q8_0 emitter received another storage schema"
  else if Block_schema.Logical_decode.code_bits equation <> 8
       || Block_schema.Logical_decode.code_signedness equation
          <> Block_schema.Physical_layout.Signedness.Signed
  then Error "constructive Q8_0 emitter requires signed eight-bit codes"
  else
    let value_function = "llmopt_schema_q8_0_value" in
    Ok
      (Printf.sprintf {|
inline float %s(device const uchar* block, uint lane) {
  const half scale = *reinterpret_cast<device const half*>(block + %du);
  const char code = *reinterpret_cast<device const char*>(block + %du + lane);
  return float(scale) * float(code);
}
|} value_function scale_offset quant_offset
      ^ emit_linear_pair ~name:"q8_0" ~value_function
          ~block_elements:(Block_schema.Physical_layout.block_elements physical)
          ~block_bytes:(Block_schema.Physical_layout.block_bytes physical))

let q5_0_linear ~target schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* zero = offset_zero (Block_schema.Logical_decode.code_signedness equation) in
  let* scale_offset =
    field_offset physical Block_schema.Physical_layout.Field.Block_scale
  in
  let* high_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized_high
  in
  let* low_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized_low
  in
  let* () = require_block32 ~target physical "Q5_0" in
  if Block_schema.quant_type schema <> Some Ir.Dtype.Q5_0 then
    Error "constructive Q5_0 emitter received another storage schema"
  else if Block_schema.Logical_decode.code_bits equation <> 5 then
    Error "constructive Q5_0 emitter requires five-bit codes"
  else
    let value_function = "llmopt_schema_q5_0_value" in
    Ok
      (Printf.sprintf {|
inline float %s(device const uchar* block, uint lane) {
  const half scale = *reinterpret_cast<device const half*>(block + %du);
  const uchar high_byte = block[%du + (lane >> 3u)];
  const uint high_bit = (high_byte >> (lane & 7u)) & 1u;
  const uchar low_byte = block[%du + (lane & 15u)];
  const uint low_nibble = lane < 16u ? (low_byte & 15u) : (low_byte >> 4u);
  return float(scale) * float(int(low_nibble | (high_bit << 4u)) - %d);
}
|} value_function scale_offset high_offset low_offset zero
      ^ emit_linear_pair ~name:"q5_0" ~value_function
          ~block_elements:(Block_schema.Physical_layout.block_elements physical)
          ~block_bytes:(Block_schema.Physical_layout.block_bytes physical))

let emit_q4_0_linear_f16_x4 ~scale_offset ~quant_offset ~zero ~block_elements
    ~block_bytes =
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
  const uint col_group = gid >> 5u;
  const uint base_col = col_group << 2u;
  const uint tg_base_col = ((gid - tid) >> 5u) << 2u;
  if (tg_base_col >= params.n) return;
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

  threadgroup half s_activation[4096];
  for (uint k_start = 0u; k_start < params.k; k_start += 4096u) {
    const uint chunk_len = min(4096u, params.k - k_start);
    for (uint i = tid; i < chunk_len; i += 256u) {
      s_activation[i] = activation[k_start + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint chunk_blocks = chunk_len / %du;
    const uint chunk_half_blocks = chunk_blocks >> 1u;
    const uint b_start = k_start / %du;

    for (uint b = 0u; b < chunk_half_blocks; ++b) {
      const uint block_idx = b_start + (b << 1u) + b_offset;
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
      accumulator0 += float(scale0) * (a_low * q0_low + a_high * q0_high);
      accumulator1 += float(scale1) * (a_low * q1_low + a_high * q1_high);
      accumulator2 += float(scale2) * (a_low * q2_low + a_high * q2_high);
      accumulator3 += float(scale3) * (a_low * q3_low + a_high * q3_high);
    }
    if ((chunk_blocks & 1u) != 0u) {
      const uint block_idx = b_start + chunk_blocks - 1u;
      const uint blk_byte_off = block_idx * %du;
      const uint s_k = (chunk_blocks - 1u) * %du + lane;
      const float a = float(s_activation[s_k]);
      accumulator0 += a * llmopt_schema_q4_0_value(row_blocks0 + blk_byte_off, lane);
      accumulator1 += a * llmopt_schema_q4_0_value(row_blocks1 + blk_byte_off, lane);
      accumulator2 += a * llmopt_schema_q4_0_value(row_blocks2 + blk_byte_off, lane);
      accumulator3 += a * llmopt_schema_q4_0_value(row_blocks3 + blk_byte_off, lane);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
    block_elements block_bytes block_elements block_elements block_bytes
    scale_offset scale_offset scale_offset scale_offset quant_offset
    quant_offset quant_offset quant_offset block_elements zero zero zero zero
    zero zero zero zero block_bytes block_elements

let emit_q4_0_linear_f16_x8 ~scale_offset ~quant_offset ~zero ~block_elements
    ~block_bytes =
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
  const uint col_group = gid >> 5u;
  const uint base_col = col_group << 3u;
  const uint tg_base_col = ((gid - tid) >> 5u) << 3u;
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
    const uint b_start = k_start / %du;

    for (uint b = 0u; b < chunk_half_blocks; ++b) {
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
    if ((chunk_blocks & 1u) != 0u) {
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
    block_elements block_bytes block_elements block_elements block_bytes
    scale_offset scale_offset scale_offset scale_offset
    scale_offset scale_offset scale_offset scale_offset
    quant_offset quant_offset quant_offset quant_offset
    quant_offset quant_offset quant_offset quant_offset
    block_elements
    zero zero zero zero zero zero zero zero
    zero zero zero zero zero zero zero zero
    block_bytes block_elements

let emit_q4_0_linear_f16_m16 ~scale_offset ~quant_offset ~zero ~block_elements
    ~block_bytes =
  Printf.sprintf {|
kernel void llmopt_q4_0_linear_f16_m16(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint m_tiles = (params.m + 15u) >> 4u;
  const uint total_tiles = m_tiles * params.n;
  if (simd_idx >= total_tiles) return;
  const uint m_tile = simd_idx / params.n;
  const uint column = simd_idx - m_tile * params.n;
  const uint row_base = m_tile << 4u;
  const uint blocks = params.k / %du;
  device const uchar* row_blocks = weight + column * blocks * %du;

  float acc[16] = {0.0f};

  const uint half_blocks = blocks >> 1u;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;

  device const half* in_ptrs[16];
  #pragma unroll
  for (uint r = 0; r < 16; ++r) {
    in_ptrs[r] = (row_base + r < params.m) ? input + (row_base + r) * params.k : input;
  }

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
      acc[r] += float(in_ptrs[r][k]) * w_low + float(in_ptrs[r][k + 16u]) * w_high;
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
      acc[r] += float(in_ptrs[r][k]) * decoded;
    }
  }

  #pragma unroll
  for (uint r = 0; r < 16; ++r) {
    acc[r] = simd_sum(acc[r]);
  }

  if (lane == 0u) {
    const float b_val = (params.has_bias != 0u) ? float(bias[column]) : 0.0f;
    #pragma unroll
    for (uint r = 0; r < 16; ++r) {
      if (row_base + r < params.m) {
        output[(row_base + r) * params.n + column] = half(acc[r] + b_val);
      }
    }
  }
}
|}
    block_elements block_bytes block_bytes scale_offset quant_offset
    block_elements zero zero block_bytes scale_offset quant_offset
    quant_offset zero block_elements

let emit_q4_0_linear_f16_m16_add ~scale_offset ~quant_offset ~zero ~block_elements
    ~block_bytes =
  Printf.sprintf {|
kernel void llmopt_q4_0_linear_f16_m16_add(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint m_tiles = (params.m + 15u) >> 4u;
  const uint total_tiles = m_tiles * params.n;
  if (simd_idx >= total_tiles) return;
  const uint m_tile = simd_idx / params.n;
  const uint column = simd_idx - m_tile * params.n;
  const uint row_base = m_tile << 4u;
  const uint blocks = params.k / %du;
  device const uchar* row_blocks = weight + column * blocks * %du;

  float acc[16] = {0.0f};

  const uint half_blocks = blocks >> 1u;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;

  device const half* in_ptrs[16];
  #pragma unroll
  for (uint r = 0; r < 16; ++r) {
    in_ptrs[r] = (row_base + r < params.m) ? input + (row_base + r) * params.k : input;
  }

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
      acc[r] += float(in_ptrs[r][k]) * w_low + float(in_ptrs[r][k + 16u]) * w_high;
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
      acc[r] += float(in_ptrs[r][k]) * decoded;
    }
  }

  #pragma unroll
  for (uint r = 0; r < 16; ++r) {
    acc[r] = simd_sum(acc[r]);
  }

  if (lane == 0u) {
    #pragma unroll
    for (uint r = 0; r < 16; ++r) {
      if (row_base + r < params.m) {
        const uint out_idx = (row_base + r) * params.n + column;
        output[out_idx] = half(half(acc[r]) + bias[out_idx]);
      }
    }
  }
}
|}
    block_elements block_bytes block_bytes scale_offset quant_offset
    block_elements zero zero block_bytes scale_offset quant_offset
    quant_offset zero block_elements

let emit_q4_0_linear_f16_m32 ~scale_offset ~quant_offset ~zero ~block_elements
    ~block_bytes =
  Printf.sprintf {|
kernel void llmopt_q4_0_linear_f16_m32(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint m_tiles = (params.m + 31u) >> 5u;
  const uint total_tiles = m_tiles * params.n;
  const uint tile_idx = simd_idx >> 1u;
  const uint simd_in_tile = simd_idx & 1u;
  if (tile_idx >= total_tiles) return;
  const uint m_tile = tile_idx / params.n;
  const uint column = tile_idx - m_tile * params.n;
  const uint row_base = m_tile << 5u;
  const uint sub_row_base = row_base + (simd_in_tile << 4u);
  const uint blocks = params.k / %du;
  device const uchar* row_blocks = weight + column * blocks * %du;

  float acc[16] = {0.0f};

  const uint half_blocks = blocks >> 1u;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;

  device const half* in_ptrs[16];
  #pragma unroll
  for (uint r = 0; r < 16; ++r) {
    in_ptrs[r] = (sub_row_base + r < params.m) ? input + (sub_row_base + r) * params.k : input;
  }

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
      acc[r] += float(in_ptrs[r][k]) * w_low + float(in_ptrs[r][k + 16u]) * w_high;
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
      acc[r] += float(in_ptrs[r][k]) * decoded;
    }
  }

  #pragma unroll
  for (uint r = 0; r < 16; ++r) {
    acc[r] = simd_sum(acc[r]);
  }

  if (lane == 0u) {
    const float b_val = (params.has_bias != 0u) ? float(bias[column]) : 0.0f;
    #pragma unroll
    for (uint r = 0; r < 16; ++r) {
      if (sub_row_base + r < params.m) {
        output[(sub_row_base + r) * params.n + column] = half(acc[r] + b_val);
      }
    }
  }
}
|}
    block_elements block_bytes block_bytes scale_offset quant_offset
    block_elements zero zero block_bytes scale_offset quant_offset
    quant_offset zero block_elements

let emit_q4_0_linear_f16_m32_add ~scale_offset ~quant_offset ~zero ~block_elements
    ~block_bytes =
  Printf.sprintf {|
kernel void llmopt_q4_0_linear_f16_m32_add(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint m_tiles = (params.m + 31u) >> 5u;
  const uint total_tiles = m_tiles * params.n;
  const uint tile_idx = simd_idx >> 1u;
  const uint simd_in_tile = simd_idx & 1u;
  if (tile_idx >= total_tiles) return;
  const uint m_tile = tile_idx / params.n;
  const uint column = tile_idx - m_tile * params.n;
  const uint row_base = m_tile << 5u;
  const uint sub_row_base = row_base + (simd_in_tile << 4u);
  const uint blocks = params.k / %du;
  device const uchar* row_blocks = weight + column * blocks * %du;

  float acc[16] = {0.0f};

  const uint half_blocks = blocks >> 1u;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;

  device const half* in_ptrs[16];
  #pragma unroll
  for (uint r = 0; r < 16; ++r) {
    in_ptrs[r] = (sub_row_base + r < params.m) ? input + (sub_row_base + r) * params.k : input;
  }

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
      acc[r] += float(in_ptrs[r][k]) * w_low + float(in_ptrs[r][k + 16u]) * w_high;
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
      acc[r] += float(in_ptrs[r][k]) * decoded;
    }
  }

  #pragma unroll
  for (uint r = 0; r < 16; ++r) {
    acc[r] = simd_sum(acc[r]);
  }

  if (lane == 0u) {
    #pragma unroll
    for (uint r = 0; r < 16; ++r) {
      if (sub_row_base + r < params.m) {
        const uint out_idx = (sub_row_base + r) * params.n + column;
        output[out_idx] = half(half(acc[r]) + bias[out_idx]);
      }
    }
  }
}
|}
    block_elements block_bytes block_bytes scale_offset quant_offset
    block_elements zero zero block_bytes scale_offset quant_offset
    quant_offset zero block_elements

let q4_0_linear ~target schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* zero = offset_zero (Block_schema.Logical_decode.code_signedness equation) in
  let* scale_offset =
    field_offset physical Block_schema.Physical_layout.Field.Block_scale
  in
  let* quant_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized
  in
  let block_elements = Block_schema.Physical_layout.block_elements physical in
  let block_bytes = Block_schema.Physical_layout.block_bytes physical in
  let simd_lanes = target.Target_hardware.memory.simd_lanes in
  if Block_schema.quant_type schema <> Some Ir.Dtype.Q4_0 then
    Error "constructive Q4_0 emitter received another storage schema"
  else if Block_schema.Logical_decode.code_bits equation <> 4 then
    Error "constructive Q4_0 emitter requires four-bit codes"
  else if block_elements <> 32 || simd_lanes <> 32 then
    Error
      (Printf.sprintf
         "constructive Q4_0 Metal lowering requires block=32 and SIMD=32 (got block=%d SIMD=%d)"
         block_elements simd_lanes)
  else
    Ok
      (Printf.sprintf {|
inline float llmopt_schema_q4_0_value(
    device const uchar* block,
    uint lane) {
  const half scale =
      *reinterpret_cast<device const half*>(block + %du);
  const uchar packed = block[%du + (lane & 15u)];
  const uint code = lane < 16u ? (packed & 15u) : (packed >> 4u);
  return float(scale) * float(int(code) - %d);
}

kernel void llmopt_q4_0_linear_f16(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint output_index = gid >> 5u;
  const uint output_elements = params.m * params.n;
  if (output_index >= output_elements) return;
  const uint row = output_index / params.n;
  const uint column = output_index - row * params.n;
  const uint blocks = params.k / %du;
  device const uchar* row_blocks = weight + column * blocks * %du;
  device const half* activation = input + row * params.k;
  float accumulator = 0.0f;
  const uint half_blocks = blocks >> 1u;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;
  for (uint b = 0u; b < half_blocks; ++b) {
    const uint block_idx = (b << 1u) + b_offset;
    device const uchar* blk = row_blocks + block_idx * %du;
    const half scale = *reinterpret_cast<device const half*>(blk + %du);
    const uchar packed = blk[%du + l];
    const uint k = block_idx * %du + l;
    const float a_low = float(activation[k]);
    const float a_high = float(activation[k + 16u]);
    const float q_low = float(int(packed & 15u) - %d);
    const float q_high = float(int(packed >> 4u) - %d);
    accumulator += float(scale) * (a_low * q_low + a_high * q_high);
  }
  if ((blocks & 1u) != 0u) {
    const uint block_idx = blocks - 1u;
    device const uchar* blk = row_blocks + block_idx * %du;
    const uint k = block_idx * %du + lane;
    accumulator += float(activation[k])
        * llmopt_schema_q4_0_value(blk, lane);
  }
  accumulator = simd_sum(accumulator);
  if (lane == 0u) {
    if (params.has_bias != 0u) accumulator += float(bias[column]);
    output[output_index] = half(accumulator);
  }
}

kernel void llmopt_q4_0_linear_f16_m2(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint column = gid >> 5u;
  if (column >= params.n) return;
  const uint blocks = params.k / %du;
  device const uchar* row_blocks = weight + column * blocks * %du;
  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
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
    const float in0_low = float(input[k]);
    const float in0_high = float(input[k + 16u]);
    const float in1_low = float(input[params.k + k]);
    const float in1_high = float(input[params.k + k + 16u]);
    const float q_low = float(int(packed & 15u) - %d);
    const float q_high = float(int(packed >> 4u) - %d);
    accumulator0 += s * (in0_low * q_low + in0_high * q_high);
    accumulator1 += s * (in1_low * q_low + in1_high * q_high);
  }
  if ((blocks & 1u) != 0u) {
    const uint block_idx = blocks - 1u;
    device const uchar* blk = row_blocks + block_idx * %du;
    const uint k = block_idx * %du + lane;
    const float decoded = llmopt_schema_q4_0_value(blk, lane);
    accumulator0 += float(input[k]) * decoded;
    accumulator1 += float(input[params.k + k]) * decoded;
  }
  accumulator0 = simd_sum(accumulator0);
  accumulator1 = simd_sum(accumulator1);
  if (lane == 0u) {
    const float bias_value =
        params.has_bias == 0u ? 0.0f : float(bias[column]);
    output[column] = half(accumulator0 + bias_value);
    output[params.n + column] = half(accumulator1 + bias_value);
  }
}

kernel void llmopt_q4_0_linear_f16_m4(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint m_tiles = (params.m + 3u) >> 2u;
  const uint total_tiles = m_tiles * params.n;
  if (simd_idx >= total_tiles) return;
  const uint m_tile = simd_idx / params.n;
  const uint column = simd_idx - m_tile * params.n;
  const uint row_base = m_tile << 2u;
  const uint blocks = params.k / %du;
  device const uchar* row_blocks = weight + column * blocks * %du;

  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
  float accumulator2 = 0.0f;
  float accumulator3 = 0.0f;

  const uint half_blocks = blocks >> 1u;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;

  device const half* in0 = input + row_base * params.k;
  device const half* in1 = (row_base + 1u < params.m) ? in0 + params.k : in0;
  device const half* in2 = (row_base + 2u < params.m) ? in0 + 2u * params.k : in0;
  device const half* in3 = (row_base + 3u < params.m) ? in0 + 3u * params.k : in0;

  for (uint b = 0u; b < half_blocks; ++b) {
    const uint block_idx = (b << 1u) + b_offset;
    device const uchar* blk = row_blocks + block_idx * %du;
    const half scale = *reinterpret_cast<device const half*>(blk + %du);
    const uchar packed = blk[%du + l];
    const float s = float(scale);
    const uint k = block_idx * %du + l;
    const float q_low = float(int(packed & 15u) - %d);
    const float q_high = float(int(packed >> 4u) - %d);

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
    device const uchar* blk = row_blocks + block_idx * %du;
    const uint k = block_idx * %du + lane;
    const float decoded = llmopt_schema_q4_0_value(blk, lane);
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
    const float b_val = (params.has_bias != 0u) ? float(bias[column]) : 0.0f;
    output[row_base * params.n + column] = half(accumulator0 + b_val);
    if (row_base + 1u < params.m)
      output[(row_base + 1u) * params.n + column] = half(accumulator1 + b_val);
    if (row_base + 2u < params.m)
      output[(row_base + 2u) * params.n + column] = half(accumulator2 + b_val);
    if (row_base + 3u < params.m)
      output[(row_base + 3u) * params.n + column] = half(accumulator3 + b_val);
  }
}

kernel void llmopt_q4_0_linear_f16_m4_add(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint m_tiles = (params.m + 3u) >> 2u;
  const uint total_tiles = m_tiles * params.n;
  if (simd_idx >= total_tiles) return;
  const uint m_tile = simd_idx / params.n;
  const uint column = simd_idx - m_tile * params.n;
  const uint row_base = m_tile << 2u;
  const uint blocks = params.k / %du;
  device const uchar* row_blocks = weight + column * blocks * %du;

  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
  float accumulator2 = 0.0f;
  float accumulator3 = 0.0f;

  const uint half_blocks = blocks >> 1u;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;

  device const half* in0 = input + row_base * params.k;
  device const half* in1 = (row_base + 1u < params.m) ? in0 + params.k : in0;
  device const half* in2 = (row_base + 2u < params.m) ? in0 + 2u * params.k : in0;
  device const half* in3 = (row_base + 3u < params.m) ? in0 + 3u * params.k : in0;

  for (uint b = 0u; b < half_blocks; ++b) {
    const uint block_idx = (b << 1u) + b_offset;
    device const uchar* blk = row_blocks + block_idx * %du;
    const half scale = *reinterpret_cast<device const half*>(blk + %du);
    const uchar packed = blk[%du + l];
    const float s = float(scale);
    const uint k = block_idx * %du + l;
    const float q_low = float(int(packed & 15u) - %d);
    const float q_high = float(int(packed >> 4u) - %d);

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
    device const uchar* blk = row_blocks + block_idx * %du;
    const uint k = block_idx * %du + lane;
    const float decoded = llmopt_schema_q4_0_value(blk, lane);
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
    const uint out_idx0 = row_base * params.n + column;
    output[out_idx0] = half(half(accumulator0) + bias[out_idx0]);
    if (row_base + 1u < params.m) {
      const uint out_idx1 = (row_base + 1u) * params.n + column;
      output[out_idx1] = half(half(accumulator1) + bias[out_idx1]);
    }
    if (row_base + 2u < params.m) {
      const uint out_idx2 = (row_base + 2u) * params.n + column;
      output[out_idx2] = half(half(accumulator2) + bias[out_idx2]);
    }
    if (row_base + 3u < params.m) {
      const uint out_idx3 = (row_base + 3u) * params.n + column;
      output[out_idx3] = half(half(accumulator3) + bias[out_idx3]);
    }
  }
}
|}
         scale_offset quant_offset zero
         block_elements block_bytes block_bytes scale_offset quant_offset
         block_elements zero zero block_bytes block_elements
         block_elements block_bytes block_bytes scale_offset quant_offset
         block_elements zero zero block_bytes block_elements
         block_elements block_bytes block_bytes scale_offset quant_offset
         block_elements zero zero block_bytes block_elements
         block_elements block_bytes block_bytes scale_offset quant_offset
         block_elements zero zero block_bytes block_elements
      ^ emit_q4_0_linear_f16_x4 ~scale_offset ~quant_offset ~zero ~block_elements
          ~block_bytes
      ^ emit_q4_0_linear_f16_x8 ~scale_offset ~quant_offset ~zero ~block_elements
          ~block_bytes
      ^ emit_q4_0_linear_f16_m16 ~scale_offset ~quant_offset ~zero ~block_elements
          ~block_bytes
      ^ emit_q4_0_linear_f16_m16_add ~scale_offset ~quant_offset ~zero ~block_elements
          ~block_bytes
      ^ emit_q4_0_linear_f16_m32 ~scale_offset ~quant_offset ~zero ~block_elements
          ~block_bytes
      ^ emit_q4_0_linear_f16_m32_add ~scale_offset ~quant_offset ~zero ~block_elements
          ~block_bytes)

let linear_kernels ~target schema =
  match Block_schema.quant_type schema with
  | Some Ir.Dtype.Q8_0 -> q8_0_linear ~target schema
  | Some Ir.Dtype.Q4_K -> q4_k_linear ~target schema
  | Some Ir.Dtype.Q5_K -> q5_k_linear ~target schema
  | Some Ir.Dtype.Q6_K -> q6_k_linear ~target schema
  | Some Ir.Dtype.Q5_0 -> q5_0_linear ~target schema
  | Some Ir.Dtype.Q4_0 -> q4_0_linear ~target schema
  | Some Ir.Dtype.IQ4_XS -> iq4_xs_linear ~target schema
  | None -> Error "constructive Metal Linear emission requires block-quant storage"
