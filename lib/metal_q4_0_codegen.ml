(* Internal module split (LOOP item 7): Q4_0 schema assembly, specialized
   epilogue/multi-linear wrappers, and add twins. *)

open Metal_q4_0_decode_codegen
open Metal_q4_0_prefill_codegen
open Metal_q4_0_qkv_codegen
open Metal_q4_0_epilogue_codegen
open Metal_block_codegen_common

let ( let* ) = Result.bind

let emit_q4_0_linear_f16_add ?(name = "llmopt_q4_0_linear_f16_add")
    ~scale_offset ~quant_offset ~zero ~block_elements ~block_bytes () =
  Printf.sprintf {|
kernel void %s(
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
    output[output_index] = half(half(accumulator) + bias[output_index]);
  }
}
|}
    name block_elements block_bytes block_bytes scale_offset quant_offset
    block_elements zero zero block_bytes block_elements

let emit_q4_0_linear_f16_m2_add ?(name = "llmopt_q4_0_linear_f16_m2_add")
    ~scale_offset ~quant_offset ~zero ~block_elements ~block_bytes () =
  Printf.sprintf {|
kernel void %s(
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
    output[column] = half(half(accumulator0) + bias[column]);
    output[params.n + column] = half(half(accumulator1) + bias[params.n + column]);
  }
}
|}
    name block_elements block_bytes block_bytes scale_offset quant_offset
    block_elements zero zero block_bytes block_elements

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
          ~block_bytes ()
      ^ emit_q4_0_linear_f16_x8 ~split_k:1 ~scale_offset ~quant_offset ~zero
          ~block_elements ~block_bytes ()
      ^ emit_q4_0_linear_f16_x16 ~split_k:1 ~scale_offset ~quant_offset ~zero
          ~block_elements ~block_bytes ()
      ^ emit_q4_0_linear_f16_m16 ~scale_offset ~quant_offset ~zero ~block_elements
          ~block_bytes ()
      ^ emit_q4_0_linear_f16_m16_add ~scale_offset ~quant_offset ~zero ~block_elements
          ~block_bytes
      ^ emit_q4_0_linear_f16_m32 ~scale_offset ~quant_offset ~zero ~block_elements
          ~block_bytes ()
      ^ emit_q4_0_linear_f16_m32_add ~scale_offset ~quant_offset ~zero ~block_elements
          ~block_bytes
      ^ emit_q4_0_gated_linear_f16_x8 ~scale_offset ~quant_offset ~zero ~block_elements
          ~block_bytes)

let q4_0_x16_epilogue_kernel ~name ~epilogue ~target schema =
  q4_0_decode_epilogue_kernel ~columns:16 ~label:"x16"
    ~emit_flat:(fun ~scale_offset ~quant_offset ~zero ~block_elements
                    ~block_bytes ->
      emit_q4_0_linear_f16_x16 ~split_k:1 ~scale_offset ~quant_offset ~zero
        ~block_elements ~block_bytes ())
    ~name ~epilogue ~target schema

let q4_0_x8_epilogue_kernel ~name ~epilogue ~target schema =
  q4_0_decode_epilogue_kernel ~columns:8 ~label:"x8"
    ~emit_flat:(fun ~scale_offset ~quant_offset ~zero ~block_elements
                    ~block_bytes ->
      emit_q4_0_linear_f16_x8 ~split_k:1 ~scale_offset ~quant_offset ~zero
        ~block_elements ~block_bytes ())
    ~name ~epilogue ~target schema

let q4_0_x4_epilogue_kernel ~name ~epilogue ~target schema =
  q4_0_decode_epilogue_kernel ~columns:4 ~label:"x4"
    ~emit_flat:(fun ~scale_offset ~quant_offset ~zero ~block_elements
                    ~block_bytes ->
      emit_q4_0_linear_f16_x4 ~scale_offset ~quant_offset ~zero ~block_elements
        ~block_bytes ())
    ~name ~epilogue ~target schema

let q4_0_m16_epilogue_kernel ~name ~epilogue ~target schema =
  q4_0_decode_epilogue_kernel ~rows:16 ~row_guard:"row_base" ~columns:16
    ~label:"m16"
    ~emit_flat:(fun ~scale_offset ~quant_offset ~zero ~block_elements
                    ~block_bytes ->
      emit_q4_0_linear_f16_m16 ~scale_offset ~quant_offset ~zero ~block_elements
        ~block_bytes ())
    ~name ~epilogue ~target schema

let q4_0_m32_epilogue_kernel ~name ~epilogue ~target schema =
  q4_0_decode_epilogue_kernel ~rows:16 ~row_guard:"sub_row_base" ~columns:16
    ~label:"m32"
    ~emit_flat:(fun ~scale_offset ~quant_offset ~zero ~block_elements
                    ~block_bytes ->
      emit_q4_0_linear_f16_m32 ~scale_offset ~quant_offset ~zero ~block_elements
        ~block_bytes ())
    ~name ~epilogue ~target schema

let q4_0_multi_linear_f16_x8_kernel ~name ~epilogue ?(num_slots = 2) ~target
    schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* zero =
    offset_zero (Block_schema.Logical_decode.code_signedness equation)
  in
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
    Error
      "constructive Q4_0 multi-linear x8 emitter received another storage schema"
  else if Block_schema.Logical_decode.code_bits equation <> 4 then
    Error "constructive Q4_0 multi-linear x8 emitter requires four-bit codes"
  else if block_elements <> 32 || simd_lanes <> 32 then
    Error
      (Printf.sprintf
         "constructive Q4_0 multi-linear x8 emission requires block=32 and \
          SIMD=32 (got block=%d SIMD=%d)"
         block_elements simd_lanes)
  else
    Ok
      (emit_q4_0_multi_linear_f16_x8 ~name ~epilogue ~num_slots ~scale_offset
         ~quant_offset ~zero ~block_elements ~block_bytes ())

let q4_0_multi_linear_f16_x16_kernel ~name ~epilogue ?(num_slots = 2) ~target
    schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* zero =
    offset_zero (Block_schema.Logical_decode.code_signedness equation)
  in
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
    Error
      "constructive Q4_0 multi-linear x16 emitter received another storage schema"
  else if Block_schema.Logical_decode.code_bits equation <> 4 then
    Error "constructive Q4_0 multi-linear x16 emitter requires four-bit codes"
  else if block_elements <> 32 || simd_lanes <> 32 then
    Error
      (Printf.sprintf
         "constructive Q4_0 multi-linear x16 emission requires block=32 and \
          SIMD=32 (got block=%d SIMD=%d)"
         block_elements simd_lanes)
  else
    Ok
      (emit_q4_0_multi_linear_f16_x16 ~name ~epilogue ~num_slots ~scale_offset
         ~quant_offset ~zero ~block_elements ~block_bytes ())

let q4_0_multi_export_linear_f16_x8_kernel ~name ~epilogues ?(num_slots = 2)
    ~target schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* zero =
    offset_zero (Block_schema.Logical_decode.code_signedness equation)
  in
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
    Error
      "constructive Q4_0 multi-export x8 emitter received another storage schema"
  else if Block_schema.Logical_decode.code_bits equation <> 4 then
    Error "constructive Q4_0 multi-export x8 emitter requires four-bit codes"
  else if block_elements <> 32 || simd_lanes <> 32 then
    Error
      (Printf.sprintf
         "constructive Q4_0 multi-export x8 emission requires block=32 and \
          SIMD=32 (got block=%d SIMD=%d)"
         block_elements simd_lanes)
  else
    Ok
      (emit_q4_0_multi_export_linear_f16_x8 ~name ~epilogues ~num_slots
         ~scale_offset ~quant_offset ~zero ~block_elements ~block_bytes ())

let q4_0_multi_export_linear_f16_x16_kernel ~name ~epilogues ?(num_slots = 2)
    ~target schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* zero =
    offset_zero (Block_schema.Logical_decode.code_signedness equation)
  in
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
    Error
      "constructive Q4_0 multi-export x16 emitter received another storage schema"
  else if Block_schema.Logical_decode.code_bits equation <> 4 then
    Error "constructive Q4_0 multi-export x16 emitter requires four-bit codes"
  else if block_elements <> 32 || simd_lanes <> 32 then
    Error
      (Printf.sprintf
         "constructive Q4_0 multi-export x16 emission requires block=32 and \
          SIMD=32 (got block=%d SIMD=%d)"
         block_elements simd_lanes)
  else
    Ok
      (emit_q4_0_multi_export_linear_f16_x16 ~name ~epilogues ~num_slots
         ~scale_offset ~quant_offset ~zero ~block_elements ~block_bytes ())

(* G4P12-04d-A2 schema-level admission for the generic and _m2 _add twins.
   The twins carry no epilogue parameter (they are Residual_add by
   construction with the additive tensor bound to the flat [bias] buffer(2)
   slot) and share the flat ABI, so these wrappers only validate the Q4_0
   block schema and emit the parametric-style twin template. *)
let q4_0_twin_schema_geometry ~label ~target schema =
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
    Error
      (Printf.sprintf
         "constructive Q4_0 %s twin emitter received another storage schema"
         label)
  else if Block_schema.Logical_decode.code_bits equation <> 4 then
    Error
      (Printf.sprintf
         "constructive Q4_0 %s twin emitter requires four-bit codes" label)
  else if block_elements <> 32 || simd_lanes <> 32 then
    Error
      (Printf.sprintf
         "constructive Q4_0 %s twin emission requires block=32 and SIMD=32 \
          (got block=%d SIMD=%d)"
         label block_elements simd_lanes)
  else
    Ok (scale_offset, quant_offset, zero, block_elements, block_bytes)

let q4_0_linear_add_kernel ~name ~target schema =
  let* scale_offset, quant_offset, zero, block_elements, block_bytes =
    q4_0_twin_schema_geometry ~label:"linear-add" ~target schema
  in
  Ok
    (emit_q4_0_linear_f16_add ~name ~scale_offset ~quant_offset ~zero
       ~block_elements ~block_bytes ())

let q4_0_m2_add_kernel ~name ~target schema =
  let* scale_offset, quant_offset, zero, block_elements, block_bytes =
    q4_0_twin_schema_geometry ~label:"m2-add" ~target schema
  in
  Ok
    (emit_q4_0_linear_f16_m2_add ~name ~scale_offset ~quant_offset ~zero
       ~block_elements ~block_bytes ())


let q4_0_qkv ~target schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* () = require_block32 ~target physical "Q4_0 QKV" in
  let* scale_offset =
    field_offset physical Block_schema.Physical_layout.Field.Block_scale
  in
  let* quant_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized
  in
  let* zero = offset_zero (Block_schema.Logical_decode.code_signedness equation) in
  let block_elements = Block_schema.Physical_layout.block_elements physical in
  let block_bytes = Block_schema.Physical_layout.block_bytes physical in
  Ok
    (emit_q4_0_qkv_linear_f16_x8 ~scale_offset ~quant_offset ~zero ~block_elements ~block_bytes
    ^ emit_q4_0_qkv_linear_f16_x16 ~scale_offset ~quant_offset ~zero ~block_elements ~block_bytes
    ^ emit_q4_0_qkv_linear_f16_m32 ~scale_offset ~quant_offset ~zero ~block_elements ~block_bytes)

