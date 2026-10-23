(** Shared schema offset/geometry checks and common block32 emission loops for
   block-quantized codegen (Tile32_Affine{4,8} and Q4_0). *)

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

let emit_block32_linear_add ~name ~value_function ~block_elements ~block_bytes =
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
  for (uint block = 0; block < blocks; ++block) {
    const uint k = block * %du + lane;
    accumulator += float(activation[k])
        * %s(row_blocks + block * %du, lane);
  }
  accumulator = simd_sum(accumulator);
  if (lane == 0u) {
    output[output_index] = half(half(accumulator) + bias[output_index]);
  }
}
|}
    name block_elements block_bytes block_elements value_function block_bytes

let emit_block32_m2_add ~name ~value_function ~block_elements ~block_bytes =
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
  for (uint block = 0; block < blocks; ++block) {
    const uint k = block * %du + lane;
    const float decoded = %s(row_blocks + block * %du, lane);
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
    name block_elements block_bytes block_elements value_function block_bytes
