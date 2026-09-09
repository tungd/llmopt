(* Internal module split (LOOP item 7): leaf emitters and epilogue
   helpers now live in sibling modules opened here. *)

open Metal_q4_0_decode_codegen
open Metal_q4_0_prefill_codegen
open Metal_q4_0_qkv_codegen
open Metal_q4_0_epilogue_codegen
open Metal_block_codegen_common

let ( let* ) = Result.bind

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
      (emit_superblock ~m2_x4:true schema ~name:"q4_k" ~value_function
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
      (emit_superblock ~m2_x4:true schema ~name:"q5_k" ~value_function
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
      (emit_superblock ~m2_x4:true schema ~name:"q6_k" ~value_function
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
        (emit_superblock ~m2_x4:false schema ~name:"iq4_xs" ~value_function
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

(* Digest-keyed epilogue variants of the real Q4_0 decode kernels.          *)
(*                                                                          *)
(* [llmopt_q4_0_linear_f16_x8] and [llmopt_q4_0_linear_f16_x4] are the     *)
(* costed decode kernels for the real Gemma-4 decode shapes (m = 1, k a     *)
(* multiple of 32, n a multiple of 8 for the x8 kernel and of 4 otherwise): *)
(* 256-thread kernels in which each SIMDgroup reduces a whole column group  *)
(* and the [accumulator0..c-1] registers own one output column each.  An    *)
(* epilogue-tail variant of such a kernel rewrites the column-guarded       *)
(* store tail into [columns] stores that apply an arbitrary [Epilogue.t]    *)
(* readback and renames the kernel to the deterministic digest name derived *)
(* from the canonicalized epilogue (see [Epilogue.algebraic_kernel_name]).  *)
(*                                                                          *)
(* The rewritten kernel header drops the legacy [bias] slot (buffer(2);     *)
(* the store tail that read it is replaced wholesale) and instead declares  *)
(* the canonicalized epilogue's input buffers immediately after the weight  *)
(* parameter, before the output workspace:                                 *)
(*                                                                          *)
(*   kernel void <digest>(                                                  *)
(*       device const half* input [[buffer(0)]],                            *)
(*       device const uchar* weight [[buffer(1)]],                          *)
(*       device const half* buf_0 [[buffer(2)]],   (p canonical buffers)    *)
(*       ...                                                                 *)
(*       device half* output [[buffer(2 + p)]],                             *)
(*       constant QuantLinearParams& params [[buffer(3 + p)]],              *)
(*       uint gid ..., uint tid ..., uint lane ...)                         *)
(*                                                                          *)
(* The epilogue is canonicalized first so the emitted buffer names          *)
(* ([buf_0], [buf_1], ...) and the digest name agree with the stable        *)
(* first-occurrence order of [Epilogue.inputs].  The flat kernel text is    *)
(* left untouched, so the legacy flat decode path stays byte-identical.     *)
(* The single-row rewrite is shared by the x8 and x4 decode kernels, whose  *)
(* headers, thread-position attributes, and writeback guard markers are     *)
(* identical; only the column count per SIMDgroup and the label used in     *)
(* error messages differ.                                                   *)
let emit_q4_0_linear_f16_add ?(name = "llmopt_q4_0_linear_f16_add")
    ~scale_offset ~quant_offset ~zero ~block_elements ~block_bytes =
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
    ~scale_offset ~quant_offset ~zero ~block_elements ~block_bytes =
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
       ~block_elements ~block_bytes)

let q4_0_m2_add_kernel ~name ~target schema =
  let* scale_offset, quant_offset, zero, block_elements, block_bytes =
    q4_0_twin_schema_geometry ~label:"m2-add" ~target schema
  in
  Ok
    (emit_q4_0_linear_f16_m2_add ~name ~scale_offset ~quant_offset ~zero
       ~block_elements ~block_bytes)

(* G4P12-05-1 schema-level admission for the Q8_0 generic and _m2 _add twins
   (the block32 twin templates above, parameterized by the Q8_0 value
   function).  The geometry validation mirrors q8_0_linear's preamble; the
   offsets only prove the physical fields exist (the value function reads the
   block directly, so the twin body itself carries no offsets). *)
let q8_0_twin_schema_geometry ~label ~target schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* _scale_offset =
    field_offset physical Block_schema.Physical_layout.Field.Block_scale
  in
  let* _quant_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized
  in
  let* () = require_block32 ~target physical "Q8_0" in
  if Block_schema.quant_type schema <> Some Ir.Dtype.Q8_0 then
    Error
      (Printf.sprintf
         "constructive Q8_0 %s twin emitter received another storage schema"
         label)
  else if Block_schema.Logical_decode.code_bits equation <> 8
       || Block_schema.Logical_decode.code_signedness equation
          <> Block_schema.Physical_layout.Signedness.Signed
  then Error "constructive Q8_0 twin emitter requires signed eight-bit codes"
  else
    let block_elements = Block_schema.Physical_layout.block_elements physical in
    let block_bytes = Block_schema.Physical_layout.block_bytes physical in
    Ok (block_elements, block_bytes)

let q8_0_linear_add_kernel ~name ~target schema =
  let* block_elements, block_bytes =
    q8_0_twin_schema_geometry ~label:"linear-add" ~target schema
  in
  Ok
    (emit_block32_linear_add ~name ~value_function:"llmopt_schema_q8_0_value"
       ~block_elements ~block_bytes)

let q8_0_m2_add_kernel ~name ~target schema =
  let* block_elements, block_bytes =
    q8_0_twin_schema_geometry ~label:"m2-add" ~target schema
  in
  Ok
    (emit_block32_m2_add ~name ~value_function:"llmopt_schema_q8_0_value"
       ~block_elements ~block_bytes)

(* G4P12-05-1: same block32 twin admissions for Q5_0 (five-bit offset codes;
   the schema-value decode reads the scale/high/low offsets directly, so the
   twin body again carries no offsets of its own). *)
let q5_0_twin_schema_geometry ~label ~target schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* _scale_offset =
    field_offset physical Block_schema.Physical_layout.Field.Block_scale
  in
  let* _high_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized_high
  in
  let* _low_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized_low
  in
  let* () = require_block32 ~target physical "Q5_0" in
  if Block_schema.quant_type schema <> Some Ir.Dtype.Q5_0 then
    Error
      (Printf.sprintf
         "constructive Q5_0 %s twin emitter received another storage schema"
         label)
  else if Block_schema.Logical_decode.code_bits equation <> 5 then
    Error "constructive Q5_0 twin emitter requires five-bit codes"
  else
    let block_elements = Block_schema.Physical_layout.block_elements physical in
    let block_bytes = Block_schema.Physical_layout.block_bytes physical in
    Ok (block_elements, block_bytes)

let q5_0_linear_add_kernel ~name ~target schema =
  let* block_elements, block_bytes =
    q5_0_twin_schema_geometry ~label:"linear-add" ~target schema
  in
  Ok
    (emit_block32_linear_add ~name ~value_function:"llmopt_schema_q5_0_value"
       ~block_elements ~block_bytes)

let q5_0_m2_add_kernel ~name ~target schema =
  let* block_elements, block_bytes =
    q5_0_twin_schema_geometry ~label:"m2-add" ~target schema
  in
  Ok
    (emit_block32_m2_add ~name ~value_function:"llmopt_schema_q5_0_value"
       ~block_elements ~block_bytes)

(* G4P12-05-1b: superblock twin schema admissions for Q4_K/Q5_K/Q6_K/IQ4_XS.
   The family value function reads the block (scale/min/table offsets resolved
   inside the value function from the schema that built it), so the geometry
   checks the superblock shape, the quant type and the code width, plus
   presence of the distinguishing physical fields. *)
let superblock_twin_geometry ~label ~target ~expected_quant ~code_bits schema =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let equation = Block_schema.Logical_decode.equation logical in
  let* () = require_superblock ~target physical logical "twin" in
  if Block_schema.quant_type schema <> Some expected_quant then
    Error
      (Printf.sprintf
         "constructive superblock %s twin emitter received another storage schema"
         label)
  else if Block_schema.Logical_decode.code_bits equation <> code_bits then
    Error
      (Printf.sprintf
         "constructive %s twin emitter requires %d-bit codes" label code_bits)
  else
    let block_elements = Block_schema.Physical_layout.block_elements physical in
    let block_bytes = Block_schema.Physical_layout.block_bytes physical in
    let subblock_elements =
      Block_schema.Logical_decode.subblock_elements logical
    in
    Ok (block_elements, block_bytes, subblock_elements)

let q4_k_twin_schema_geometry ~label ~target schema =
  let physical = Block_schema.physical_layout schema in
  let* _quant_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized
  in
  let* _table_offset =
    field_offset physical Block_schema.Physical_layout.Field.Scale_minimum_table
  in
  superblock_twin_geometry ~label ~target ~expected_quant:Ir.Dtype.Q4_K
    ~code_bits:4 schema

let q5_k_twin_schema_geometry ~label ~target schema =
  let physical = Block_schema.physical_layout schema in
  let* _high_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized_high
  in
  let* _low_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized_low
  in
  superblock_twin_geometry ~label ~target ~expected_quant:Ir.Dtype.Q5_K
    ~code_bits:5 schema

let q6_k_twin_schema_geometry ~label ~target schema =
  let physical = Block_schema.physical_layout schema in
  let* _high_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized_high
  in
  let* _low_offset =
    field_offset physical Block_schema.Physical_layout.Field.Quantized_low
  in
  let* _scale_table =
    field_offset physical Block_schema.Physical_layout.Field.Scale_table
  in
  superblock_twin_geometry ~label ~target ~expected_quant:Ir.Dtype.Q6_K
    ~code_bits:6 schema

let iq4_xs_twin_schema_geometry ~label ~target schema =
  let physical = Block_schema.physical_layout schema in
  let* _high_offset =
    field_offset physical Block_schema.Physical_layout.Field.Codebook_scale_high
  in
  let* _low_offset =
    field_offset physical Block_schema.Physical_layout.Field.Codebook_scale_low
  in
  superblock_twin_geometry ~label ~target ~expected_quant:Ir.Dtype.IQ4_XS
    ~code_bits:4 schema

let superblock_linear_add ~name ~value_function ~geometry schema =
  let* block_elements, block_bytes, subblock_elements = geometry schema in
  Ok
    (emit_superblock_linear_add ~name ~value_function ~block_elements
       ~block_bytes ~subblock_elements)

let q4_k_linear_add_kernel ~name ~target schema =
  let* body =
    superblock_linear_add ~name ~value_function:"llmopt_schema_q4_k_value"
      ~geometry:(q4_k_twin_schema_geometry ~label:"linear-add" ~target) schema
  in
  Ok body

let q5_k_linear_add_kernel ~name ~target schema =
  let* body =
    superblock_linear_add ~name ~value_function:"llmopt_schema_q5_k_value"
      ~geometry:(q5_k_twin_schema_geometry ~label:"linear-add" ~target) schema
  in
  Ok body

let q6_k_linear_add_kernel ~name ~target schema =
  let* body =
    superblock_linear_add ~name ~value_function:"llmopt_schema_q6_k_value"
      ~geometry:(q6_k_twin_schema_geometry ~label:"linear-add" ~target) schema
  in
  Ok body

let iq4_xs_linear_add_kernel ~name ~target schema =
  let* body =
    superblock_linear_add ~name ~value_function:"llmopt_schema_iq4_xs_value"
      ~geometry:(iq4_xs_twin_schema_geometry ~label:"linear-add" ~target)
      schema
  in
  Ok body

let iq4_xs_m2_add_kernel ~name ~target schema =
  let* block_elements, block_bytes, subblock_elements =
    iq4_xs_twin_schema_geometry ~label:"m2-add" ~target schema
  in
  Ok
    (emit_superblock_m2_add ~name ~value_function:"llmopt_schema_iq4_xs_value"
       ~block_elements ~block_bytes ~subblock_elements)

(* G4P12-05-2: on-demand _add twins for the parametric m2_x4 tile.  The flat
   m2_x4 kernel ships unconditionally in the K-family emission (emit_superblock
   ~m2_x4:true); the fold appends the twin here when a Linear-add m=2 base
   selects the m2_x4 tile, replacing the llama-shim patch. *)
let k_m2_x4_add_kernel ~name ~target ~value_function schema geometry =
  let* block_elements, block_bytes, subblock_elements = geometry schema in
  Ok
    (emit_superblock_m2_x4_add ~name ~value_function ~block_elements
       ~block_bytes ~subblock_elements)

let q4_k_m2_x4_add_kernel ~name ~target schema =
  k_m2_x4_add_kernel ~name ~target ~value_function:"llmopt_schema_q4_k_value"
    schema (q4_k_twin_schema_geometry ~label:"m2_x4-add" ~target)

let q5_k_m2_x4_add_kernel ~name ~target schema =
  k_m2_x4_add_kernel ~name ~target ~value_function:"llmopt_schema_q5_k_value"
    schema (q5_k_twin_schema_geometry ~label:"m2_x4-add" ~target)

let q6_k_m2_x4_add_kernel ~name ~target schema =
  k_m2_x4_add_kernel ~name ~target ~value_function:"llmopt_schema_q6_k_value"
    schema (q6_k_twin_schema_geometry ~label:"m2_x4-add" ~target)

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
    ^ emit_q4_0_qkv_linear_f16_m32 ~scale_offset ~quant_offset ~zero ~block_elements ~block_bytes)

let qkv_kernels ~target schema =
  match Block_schema.quant_type schema with
  | Some Ir.Dtype.Q4_0 -> q4_0_qkv ~target schema
  | _ -> Error "constructive QKV emission only implemented for Q4_0"



