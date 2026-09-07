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

(* G4P12-05-1: parametric-style _add twins for the generic and paired-row
   (_m2) tiles of the block32 families (Q8_0 first; Q5_0 reuses these once it
   leaves the legacy shim).  The decode bodies mirror emit_linear_pair
   byte-for-byte except the writeback tail: the additive tensor (the residual,
   bound to the flat [bias] buffer(2) slot) is applied element-wise with the
   double-rounding store half(half(acc) + bias[idx]) -- the legacy twin
   numerics contract, numerically distinct from a single-rounding digest tail.
   Same ABI as the flat kernels: input buffer(0), weight buffer(1), bias
   buffer(2), output buffer(3), params buffer(4).  Emitted on demand through
   the schema-level q8_0_linear_add_kernel/q8_0_m2_add_kernel admissions;
   the zoo text (emit_linear_pair output) is untouched. *)
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

(* G4P12-05-1b: the same twins for the superblock families (Q4_K/Q5_K/Q6_K/
   IQ4_XS).  The decode bodies mirror emit_superblock_linear_pair's generic
   and paired-row bodies (one schema-value call per 32-element subblock over
   256-element superblocks) with the twin writeback tails above; the schema
   value function is in scope because quant_source emits it before the twin
   text.  On-demand through the schema-level q{...}_linear_add_kernel /
   q{...}_m2_add_kernel admissions below. *)
let emit_superblock_linear_add ~name ~value_function ~block_elements
    ~block_bytes ~subblock_elements =
  let subblocks = block_elements / subblock_elements in
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
    device const uchar* packed = row_blocks + block * %du;
    for (uint subblock = 0; subblock < %du; ++subblock) {
      const uint k = block * %du + subblock * %du + lane;
      accumulator += float(activation[k])
          * %s(packed, lane, subblock);
    }
  }
  accumulator = simd_sum(accumulator);
  if (lane == 0u) {
    output[output_index] = half(half(accumulator) + bias[output_index]);
  }
}
|}
    name block_elements block_bytes block_bytes subblocks block_elements
    subblock_elements value_function

let emit_superblock_m2_add ~name ~value_function ~block_elements ~block_bytes
    ~subblock_elements =
  let subblocks = block_elements / subblock_elements in
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
    output[column] = half(half(accumulator0) + bias[column]);
    output[params.n + column] = half(half(accumulator1) + bias[params.n + column]);
  }
}
|}
    name block_elements block_bytes block_bytes subblocks block_elements
    subblock_elements value_function

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

(* G4P12-05-2: the m2_x4 four-column m=2 tile, parametrically.  The kernel
   covers four output columns per 32-lane simdgroup (column_in_group = lane
   >> 3) with eight local lanes per column walking the k dimension four
   elements at a time; the per-element decode is the family schema value
   function (decode-identical to the llama tile it replaces -- see exp-0167
   for the per-element audit), and the writeback tail follows the parametric
   flat/_add conventions.  Same ABI and lane semantics as the llama m2_x4,
   so the name-based runtime dispatch (columns = (n + 3) / 4, short-row grid)
   is unchanged; plain decode at m=2 becomes FP-consistent with the m >= 3
   parametric generic (option A: decode-consistent scalar switch). *)
let emit_superblock_m2_x4_flat ~name ~value_function ~block_elements
    ~block_bytes ~subblock_elements =
  let subblocks = block_elements / subblock_elements in
  Printf.sprintf {|
kernel void llmopt_%s_linear_f16_m2_x4(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint column_group = gid >> 5u;
  const uint column_in_group = lane >> 3u;
  const uint local_lane = lane & 7u;
  const uint column = (column_group << 2u) + column_in_group;
  if (column >= params.n) return;
  const uint blocks = params.k / %du;
  device const uchar* row_blocks = weight + column * blocks * %du;
  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
  for (uint block = 0; block < blocks; ++block) {
    device const uchar* packed = row_blocks + block * %du;
    for (uint subblock = 0; subblock < %du; ++subblock) {
      const uint k = block * %du + subblock * %du + local_lane * 4u;
      const float w0 = %s(packed, local_lane * 4u + 0u, subblock);
      const float w1 = %s(packed, local_lane * 4u + 1u, subblock);
      const float w2 = %s(packed, local_lane * 4u + 2u, subblock);
      const float w3 = %s(packed, local_lane * 4u + 3u, subblock);
      accumulator0 += float(input[k]) * w0 + float(input[k + 1u]) * w1
          + float(input[k + 2u]) * w2 + float(input[k + 3u]) * w3;
      accumulator1 += float(input[params.k + k]) * w0
          + float(input[params.k + k + 1u]) * w1
          + float(input[params.k + k + 2u]) * w2
          + float(input[params.k + k + 3u]) * w3;
    }
  }
  accumulator0 += simd_shuffle_down(accumulator0, 4);
  accumulator1 += simd_shuffle_down(accumulator1, 4);
  accumulator0 += simd_shuffle_down(accumulator0, 2);
  accumulator1 += simd_shuffle_down(accumulator1, 2);
  accumulator0 += simd_shuffle_down(accumulator0, 1);
  accumulator1 += simd_shuffle_down(accumulator1, 1);
  if (local_lane == 0u) {
    const float bias_value =
        params.has_bias == 0u ? 0.0f : float(bias[column]);
    output[column] = half(accumulator0 + bias_value);
    output[params.n + column] = half(accumulator1 + bias_value);
  }
}
|}
    name block_elements block_bytes block_bytes subblocks block_elements
    subblock_elements value_function value_function value_function value_function

let emit_superblock_m2_x4_add ~name ~value_function ~block_elements
    ~block_bytes ~subblock_elements =
  let subblocks = block_elements / subblock_elements in
  Printf.sprintf {|
kernel void %s(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint column_group = gid >> 5u;
  const uint column_in_group = lane >> 3u;
  const uint local_lane = lane & 7u;
  const uint column = (column_group << 2u) + column_in_group;
  if (column >= params.n) return;
  const uint blocks = params.k / %du;
  device const uchar* row_blocks = weight + column * blocks * %du;
  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
  for (uint block = 0; block < blocks; ++block) {
    device const uchar* packed = row_blocks + block * %du;
    for (uint subblock = 0; subblock < %du; ++subblock) {
      const uint k = block * %du + subblock * %du + local_lane * 4u;
      const float w0 = %s(packed, local_lane * 4u + 0u, subblock);
      const float w1 = %s(packed, local_lane * 4u + 1u, subblock);
      const float w2 = %s(packed, local_lane * 4u + 2u, subblock);
      const float w3 = %s(packed, local_lane * 4u + 3u, subblock);
      accumulator0 += float(input[k]) * w0 + float(input[k + 1u]) * w1
          + float(input[k + 2u]) * w2 + float(input[k + 3u]) * w3;
      accumulator1 += float(input[params.k + k]) * w0
          + float(input[params.k + k + 1u]) * w1
          + float(input[params.k + k + 2u]) * w2
          + float(input[params.k + k + 3u]) * w3;
    }
  }
  accumulator0 += simd_shuffle_down(accumulator0, 4);
  accumulator1 += simd_shuffle_down(accumulator1, 4);
  accumulator0 += simd_shuffle_down(accumulator0, 2);
  accumulator1 += simd_shuffle_down(accumulator1, 2);
  accumulator0 += simd_shuffle_down(accumulator0, 1);
  accumulator1 += simd_shuffle_down(accumulator1, 1);
  if (local_lane == 0u) {
    output[column] = half(half(accumulator0) + bias[column]);
    output[params.n + column] = half(half(accumulator1) + bias[params.n + column]);
  }
}
|}
    name block_elements block_bytes block_bytes subblocks block_elements
    subblock_elements value_function value_function value_function value_function

let emit_superblock ~m2_x4 schema ~name ~value_function value_source =
  let physical = Block_schema.physical_layout schema in
  let logical = Block_schema.logical_decode schema in
  let block_elements = Block_schema.Physical_layout.block_elements physical in
  let block_bytes = Block_schema.Physical_layout.block_bytes physical in
  let subblock_elements =
    Block_schema.Logical_decode.subblock_elements logical
  in
  value_source
  ^ emit_superblock_linear_pair ~name ~value_function ~block_elements
      ~block_bytes ~subblock_elements
  ^ if m2_x4 then
      emit_superblock_m2_x4_flat ~name ~value_function ~block_elements
        ~block_bytes ~subblock_elements
    else ""

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
let find_substring_from haystack needle start =
  let hlen = String.length haystack in
  let nlen = String.length needle in
  let rec go i =
    if i + nlen > hlen then None
    else if String.sub haystack i nlen = needle then Some i
    else go (i + 1)
  in
  if nlen = 0 then None else go (max start 0)

let q4_0_single_row_epilogue_store ~columns epilogue =
  List.init columns (fun j ->
      let acc = Printf.sprintf "accumulator%d" j in
      let idx =
        if j = 0 then "base_col" else Printf.sprintf "(base_col + %du)" j
      in
      let expr = Epilogue.emit_msl_writeback ~acc ~idx epilogue in
      if j = 0 then Printf.sprintf "    output[base_col] = half(%s);" expr
      else
        Printf.sprintf
          "    if (base_col + %du < params.n) output[base_col + %du] = half(%s);"
          j j expr)
  |> String.concat "\n"

let q4_0_single_row_digest_of_flat ~columns ~label ~name epilogue kernel =
  (* The tail of the plain kernel's argument list (the thread-position
     attributes through the closing ") {") is shared verbatim by its digest
     variant; everything before it is re-emitted with the canonicalized
     epilogue buffers inserted between the weight and the output workspace. *)
  let attr_marker = "    uint gid [[thread_position_in_grid]]," in
  match find_substring_from kernel attr_marker 0 with
  | None ->
      Error
        (Printf.sprintf "Q4_0 %s digest emission: decode kernel thread \
                         attributes are absent"
           label)
  | Some attr_start -> (
      match find_substring_from kernel ") {" attr_start with
      | None ->
          Error
            (Printf.sprintf "Q4_0 %s digest emission: decode kernel header is \
                             unterminated"
               label)
      | Some close ->
          let header_stop = close + 3 in
          let attributes =
            String.sub kernel attr_start (header_stop - attr_start)
          in
          let body =
            String.sub kernel header_stop (String.length kernel - header_stop)
          in
          let buffer_count = List.length (Epilogue.inputs epilogue) in
          let header =
            "kernel void " ^ name ^ "(\n"
            ^ "    device const half* input [[buffer(0)]],\n"
            ^ "    device const uchar* weight [[buffer(1)]]"
            ^ Epilogue.msl_buffer_decls ~start_index:2 epilogue
            ^ Printf.sprintf
                ",\n    device half* output [[buffer(%d)]],\n    constant \
                 QuantLinearParams& params [[buffer(%d)]],\n"
                (2 + buffer_count) (3 + buffer_count)
            ^ attributes
          in
          let tail_marker = "  if (lane == 0u && base_col < params.n) {" in
          let full = header ^ body in
          match find_substring_from full tail_marker 0 with
          | None ->
              Error
                (Printf.sprintf "Q4_0 %s digest emission: writeback tail is \
                                 absent from the decode kernel"
                   label)
          | Some tail_start ->
              let head = String.sub full 0 tail_start in
              Ok
                (head ^ tail_marker ^ "\n"
                ^ q4_0_single_row_epilogue_store ~columns epilogue
                ^ "\n  }\n}\n"))

let q4_0_x8_digest_of_flat ~name epilogue kernel =
  q4_0_single_row_digest_of_flat ~columns:8 ~label:"x8" ~name epilogue kernel

let q4_0_x4_digest_of_flat ~name epilogue kernel =
  q4_0_single_row_digest_of_flat ~columns:4 ~label:"x4" ~name epilogue kernel

(* Row-block digest variants (m16/m32): these kernels run one simdgroup per
   (row tile x column) and keep a [float acc[rows]] array, so the digest
   writeback replaces the flat store loop with one guarded store per
   accumulator row instead of a single-row column span.  The accumulator
   slot name is the flat kernel's array ([acc[r]]); the row-guard variable
   is the flat kernel's base-row register ([row_base] for m16,
   [sub_row_base] for m32).  The digest header layout matches the
   single-row variant exactly: epilogue buffers at buffer(2..2+p), the
   output workspace at buffer(2+p) and the params at buffer(3+p). *)
let q4_0_row_block_epilogue_stores ~rows ~row_guard epilogue =
  List.init rows (fun r ->
      let acc = Printf.sprintf "acc[%d]" r in
      let idx = Printf.sprintf "out_idx%d" r in
      let expr = Epilogue.emit_msl_writeback ~acc ~idx epilogue in
      Printf.sprintf
        "    if (%s + %du < params.m) {\n      const uint %s = (%s + %du) * params.n + column;\n      output[%s] = half(%s);\n    }"
        row_guard r idx row_guard r idx expr)
  |> String.concat "\n"

let q4_0_row_block_digest_of_flat ~rows ~row_guard ~label ~name epilogue
    kernel =
  let attr_marker = "    uint gid [[thread_position_in_grid]]," in
  match find_substring_from kernel attr_marker 0 with
  | None ->
      Error
        (Printf.sprintf "Q4_0 %s digest emission: decode kernel thread \
                         attributes are absent"
           label)
  | Some attr_start -> (
      match find_substring_from kernel ") {" attr_start with
      | None ->
          Error
            (Printf.sprintf "Q4_0 %s digest emission: decode kernel header is \
                             unterminated"
               label)
      | Some close ->
          let header_stop = close + 3 in
          let attributes =
            String.sub kernel attr_start (header_stop - attr_start)
          in
          let body =
            String.sub kernel header_stop (String.length kernel - header_stop)
          in
          let buffer_count = List.length (Epilogue.inputs epilogue) in
          let header =
            "kernel void " ^ name ^ "(\n"
            ^ "    device const half* input [[buffer(0)]],\n"
            ^ "    device const uchar* weight [[buffer(1)]]"
            ^ Epilogue.msl_buffer_decls ~start_index:2 epilogue
            ^ Printf.sprintf
                ",\n    device half* output [[buffer(%d)]],\n    constant \
                 QuantLinearParams& params [[buffer(%d)]],\n"
                (2 + buffer_count) (3 + buffer_count)
            ^ attributes
          in
          let tail_marker = "  if (lane == 0u) {" in
          let full = header ^ body in
          match find_substring_from full tail_marker 0 with
          | None ->
              Error
                (Printf.sprintf "Q4_0 %s digest emission: writeback tail is \
                                 absent from the decode kernel"
                   label)
          | Some tail_start ->
              let head = String.sub full 0 tail_start in
              Ok
                (head ^ tail_marker ^ "\n"
                ^ q4_0_row_block_epilogue_stores ~rows ~row_guard epilogue
                ^ "\n  }\n}\n"))

let q4_0_m16_digest_of_flat ~name epilogue kernel =
  q4_0_row_block_digest_of_flat ~rows:16 ~row_guard:"row_base" ~label:"m16"
    ~name epilogue kernel

(* The m32 tile runs two simdgroups per 32-row tile; each simdgroup
   [simd_in_tile] covers a 16-row half through [sub_row_base], so its
   writeback is structurally the same sixteen guarded stores as m16 but
   guarded against [sub_row_base]. *)
let q4_0_m32_digest_of_flat ~name epilogue kernel =
  q4_0_row_block_digest_of_flat ~rows:16 ~row_guard:"sub_row_base" ~label:"m32"
    ~name epilogue kernel

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

let emit_q4_0_linear_f16_m16 ?(name = "llmopt_q4_0_linear_f16_m16")
    ?(epilogue = Epilogue.none) ~scale_offset ~quant_offset ~zero ~block_elements
    ~block_bytes () =
  let flat = Printf.sprintf {|
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
  in
  if Epilogue.equal epilogue Epilogue.none then flat
  else
    match q4_0_m16_digest_of_flat ~name epilogue flat with
    | Ok text -> text
    | Error message ->
        invalid_arg
          (Printf.sprintf "Metal_block_codegen q4_0_linear_f16_m16: %s" message)

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

let emit_q4_0_linear_f16_m32 ?(name = "llmopt_q4_0_linear_f16_m32")
    ?(epilogue = Epilogue.none) ~scale_offset ~quant_offset ~zero ~block_elements
    ~block_bytes () =
  let flat = Printf.sprintf {|
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
  in
  if Epilogue.equal epilogue Epilogue.none then flat
  else
    match q4_0_m32_digest_of_flat ~name epilogue flat with
    | Ok text -> text
    | Error message ->
        invalid_arg
          (Printf.sprintf "Metal_block_codegen q4_0_linear_f16_m32: %s" message)

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

(* G4P12-04d-A2: parametric-style _add twins for the generic (arbitrary-row,
   llmopt_q4_0_linear_f16) and paired-row (_m2) tiles.  Metal's legacy
   quant_linear_add_source text-patcher used to produce these two twins from
   the llama-style legacy bodies; they are now emitted here in the same
   parametric style as the m4/m16/m32 zoo twins, keeping the legacy twin
   numerics contract: the additive tensor (the residual, bound to the [bias]
   buffer(2) slot as in the flat kernels) is applied element-wise with the
   double-rounding store half(half(acc) + bias[idx]) -- numerically distinct
   from the single-rounding digest Residual_add tail.  Same ABI as the flat
   kernels: input buffer(0), weight buffer(1), bias buffer(2), output
   buffer(3), params buffer(4); the decode order matches the parametric flat
   bodies already serving plain decode. *)
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

let q4_0_decode_epilogue_kernel ?(rows = 0) ?(row_guard = "") ~columns ~label
    ~emit_flat ~name ~epilogue ~target schema =
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
         "constructive Q4_0 %s digest emitter received another storage schema"
         label)
  else if Block_schema.Logical_decode.code_bits equation <> 4 then
    Error
      (Printf.sprintf
         "constructive Q4_0 %s digest emitter requires four-bit codes" label)
  else if block_elements <> 32 || simd_lanes <> 32 then
    Error
      (Printf.sprintf
         "constructive Q4_0 %s digest emission requires block=32 and SIMD=32 \
          (got block=%d SIMD=%d)"
         label block_elements simd_lanes)
  else
    let flat =
      emit_flat ~scale_offset ~quant_offset ~zero ~block_elements ~block_bytes
    in
    let digest_result =
      if rows = 0 then
        q4_0_single_row_digest_of_flat ~columns ~label ~name epilogue flat
      else
        q4_0_row_block_digest_of_flat ~rows ~row_guard ~label ~name epilogue
          flat
    in
    match digest_result with
    | Ok text -> Ok text
    | Error message ->
        Error
          (Printf.sprintf "constructive Q4_0 %s digest emission: %s" label message)

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

