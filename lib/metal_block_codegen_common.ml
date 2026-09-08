(** Shared schema offset/geometry checks and the common block32/superblock
   emission loops for block-quantized codegen.  Split out of
   metal_block_codegen.ml (LOOP item 7). *)

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
