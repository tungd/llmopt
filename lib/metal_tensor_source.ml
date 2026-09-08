(** Metal dense Linear shader catalog and shared text patchers
   (replace_first and the second lexical replace_all used by the
   normalization/tensor catalogs).  Split out of metal.ml (LOOP item 5). *)

open Metal_source_common

let linear_f16_source =
  "\nconstant uint LINEAR_SIMD_WIDTH = 32;\n"
  ^ "constant uint LINEAR_ROWS_PER_BLOCK = 8;\n\n"
  ^ "struct LinearF16Params { uint m; uint n; uint k; };\n\n"
  ^ "kernel void llmopt_linear_f16(\n"
  ^ "    device const half* input [[buffer(0)]],\n"
  ^ "    device const half* weight [[buffer(1)]],\n"
  ^ "    device half* output [[buffer(2)]],\n"
  ^ "    constant LinearF16Params& params [[buffer(3)]],\n"
  ^ "    uint gid [[thread_position_in_grid]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  const uint column = gid / LINEAR_SIMD_WIDTH;\n"
  ^ "  if (column >= params.n) return;\n"
  ^ "  const uint weight_base = column * params.k;\n"
  ^ "  for (uint row_base = 0; row_base < params.m;\n"
  ^ "       row_base += LINEAR_ROWS_PER_BLOCK) {\n"
  ^ "    float accumulators[8] = {};\n"
  ^ "    const uint rows = min(LINEAR_ROWS_PER_BLOCK, params.m - row_base);\n"
  ^ "    for (uint inner = lane; inner < params.k; inner += LINEAR_SIMD_WIDTH) {\n"
  ^ "      const float weight_value = float(weight[weight_base + inner]);\n"
  ^ "      for (uint row = 0; row < rows; ++row)\n"
  ^ "        accumulators[row] +=\n"
  ^ "            float(input[(row_base + row) * params.k + inner]) * weight_value;\n"
  ^ "    }\n"
  ^ "    for (uint row = 0; row < rows; ++row) {\n"
  ^ "      const float value = simd_sum(accumulators[row]);\n"
  ^ "      if (lane == 0) output[(row_base + row) * params.n + column] = half(value);\n"
  ^ "    }\n"
  ^ "  }\n"
  ^ "}\n\n"

let linear_f16_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_linear_f16" ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16 ]

let linear_f16_bf16_source =
  "\nconstant uint LINEAR_BF16_SIMD_WIDTH = 32;\n"
  ^ "constant uint LINEAR_BF16_ROWS_PER_BLOCK = 8;\n\n"
  ^ "struct LinearBf16Params { uint m; uint n; uint k; };\n\n"
  ^ "kernel void llmopt_linear_f16_bf16(\n"
  ^ "    device const half* input [[buffer(0)]],\n"
  ^ "    device const bfloat* weight [[buffer(1)]],\n"
  ^ "    device half* output [[buffer(2)]],\n"
  ^ "    constant LinearBf16Params& params [[buffer(3)]],\n"
  ^ "    uint gid [[thread_position_in_grid]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  const uint column = gid / LINEAR_BF16_SIMD_WIDTH;\n"
  ^ "  if (column >= params.n) return;\n"
  ^ "  const uint weight_base = column * params.k;\n"
  ^ "  for (uint row_base = 0; row_base < params.m;\n"
  ^ "       row_base += LINEAR_BF16_ROWS_PER_BLOCK) {\n"
  ^ "    float accumulators[8] = {};\n"
  ^ "    const uint rows = min(LINEAR_BF16_ROWS_PER_BLOCK, params.m - row_base);\n"
  ^ "    for (uint inner = lane; inner < params.k; inner += LINEAR_BF16_SIMD_WIDTH) {\n"
  ^ "      const float weight_value = float(weight[weight_base + inner]);\n"
  ^ "      for (uint row = 0; row < rows; ++row)\n"
  ^ "        accumulators[row] +=\n"
  ^ "            float(input[(row_base + row) * params.k + inner]) * weight_value;\n"
  ^ "    }\n"
  ^ "    for (uint row = 0; row < rows; ++row) {\n"
  ^ "      const float value = simd_sum(accumulators[row]);\n"
  ^ "      if (lane == 0) output[(row_base + row) * params.n + column] = half(value);\n"
  ^ "    }\n"
  ^ "  }\n"
  ^ "}\n\n"

let linear_f16_bf16_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_linear_f16_bf16" ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16 ]

let linear_f16_f32_source =
  "\nconstant uint LINEAR_F32_WEIGHT_SIMD_WIDTH = 32;\n"
  ^ "constant uint LINEAR_F32_WEIGHT_ROWS_PER_BLOCK = 8;\n\n"
  ^ "struct LinearF32WeightParams { uint m; uint n; uint k; };\n\n"
  ^ "kernel void llmopt_linear_f16_f32(\n"
  ^ "    device const half* input [[buffer(0)]],\n"
  ^ "    device const float* weight [[buffer(1)]],\n"
  ^ "    device half* output [[buffer(2)]],\n"
  ^ "    constant LinearF32WeightParams& params [[buffer(3)]],\n"
  ^ "    uint gid [[thread_position_in_grid]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  const uint column = gid / LINEAR_F32_WEIGHT_SIMD_WIDTH;\n"
  ^ "  if (column >= params.n) return;\n"
  ^ "  const uint weight_base = column * params.k;\n"
  ^ "  for (uint row_base = 0; row_base < params.m;\n"
  ^ "       row_base += LINEAR_F32_WEIGHT_ROWS_PER_BLOCK) {\n"
  ^ "    float accumulators[8] = {};\n"
  ^ "    const uint rows = min(LINEAR_F32_WEIGHT_ROWS_PER_BLOCK, params.m - row_base);\n"
  ^ "    for (uint inner = lane; inner < params.k; inner += LINEAR_F32_WEIGHT_SIMD_WIDTH) {\n"
  ^ "      const float weight_value = weight[weight_base + inner];\n"
  ^ "      for (uint row = 0; row < rows; ++row)\n"
  ^ "        accumulators[row] +=\n"
  ^ "            float(input[(row_base + row) * params.k + inner]) * weight_value;\n"
  ^ "    }\n"
  ^ "    for (uint row = 0; row < rows; ++row) {\n"
  ^ "      const float value = simd_sum(accumulators[row]);\n"
  ^ "      if (lane == 0) output[(row_base + row) * params.n + column] = half(value);\n"
  ^ "    }\n"
  ^ "  }\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_linear_f16_f32_m2_n2(\n"
  ^ "    device const half* input [[buffer(0)]],\n"
  ^ "    device const float* weight [[buffer(1)]],\n"
  ^ "    device half* output [[buffer(2)]],\n"
  ^ "    constant LinearF32WeightParams& params [[buffer(3)]],\n"
  ^ "    uint gid [[thread_position_in_grid]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  const uint first_column = (gid / LINEAR_F32_WEIGHT_SIMD_WIDTH) << 1;\n"
  ^ "  if (first_column >= params.n || params.m != 2u) return;\n"
  ^ "  const uint second_column = first_column + 1u;\n"
  ^ "  const uint first_weight_base = first_column * params.k;\n"
  ^ "  const uint second_weight_base = second_column * params.k;\n"
  ^ "  float first0 = 0.0f; float first1 = 0.0f;\n"
  ^ "  float second0 = 0.0f; float second1 = 0.0f;\n"
  ^ "  for (uint inner = lane; inner < params.k; inner += LINEAR_F32_WEIGHT_SIMD_WIDTH) {\n"
  ^ "    const float input0 = float(input[inner]);\n"
  ^ "    const float input1 = float(input[params.k + inner]);\n"
  ^ "    const float weight0 = weight[first_weight_base + inner];\n"
  ^ "    first0 += input0 * weight0; first1 += input1 * weight0;\n"
  ^ "    if (second_column < params.n) {\n"
  ^ "      const float weight1 = weight[second_weight_base + inner];\n"
  ^ "      second0 += input0 * weight1; second1 += input1 * weight1;\n"
  ^ "    }\n"
  ^ "  }\n"
  ^ "  first0 = simd_sum(first0); first1 = simd_sum(first1);\n"
  ^ "  second0 = simd_sum(second0); second1 = simd_sum(second1);\n"
  ^ "  if (lane == 0u) {\n"
  ^ "    output[first_column] = half(first0);\n"
  ^ "    output[params.n + first_column] = half(first1);\n"
  ^ "    if (second_column < params.n) {\n"
  ^ "      output[second_column] = half(second0);\n"
  ^ "      output[params.n + second_column] = half(second1);\n"
  ^ "    }\n"
  ^ "  }\n"
  ^ "}\n\n"

let linear_f16_f32_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_linear_f16_f32" ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_linear_f16_f32_m2_n2" ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16 ]

let replace_first ~needle ~replacement source =
  let needle_length = String.length needle in
  let source_length = String.length source in
  let rec find offset =
    if offset + needle_length > source_length then None
    else if String.sub source offset needle_length = needle then Some offset
    else find (offset + 1)
  in
  match find 0 with
  | None -> source
  | Some offset ->
      String.sub source 0 offset ^ replacement
      ^ String.sub source (offset + needle_length)
          (source_length - offset - needle_length)

let rec replace_all ~needle ~replacement source =
  let replaced = replace_first ~needle ~replacement source in
  if String.equal replaced source then source
  else replace_all ~needle ~replacement replaced

let embedding_source = {|
struct EmbeddingParams { uint tokens; uint vocabulary; uint width; };

struct embedding_block_q8_0 { half d; int8_t qs[32]; };
struct embedding_block_q5_0 { half d; uint8_t qh[4]; uint8_t qs[16]; };
struct embedding_block_q4_0 { half d; uint8_t qs[16]; };
struct embedding_block_q4_K {
  half d; half dmin; uint8_t scales[12]; uint8_t qs[128];
};
struct embedding_block_q5_K {
  half d; half dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128];
};
struct embedding_block_q6_K {
  uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d;
};

inline half llmopt_embedding_q4_k_value(
    device const embedding_block_q4_K& block, uint within) {
  const uint segment = within >> 5;
  const uint lane = within & 31u;
  const uint j = segment & 3u;
  const uint scale = segment < 4u
      ? uint(extract_bits(block.scales[j], 0u, 6u))
      : uint(extract_bits(block.scales[j + 8u], 0u, 4u) | (extract_bits(block.scales[j], 6u, 2u) << 4));
  const uint minimum = segment < 4u
      ? uint(extract_bits(block.scales[j + 4u], 0u, 6u))
      : uint(extract_bits(block.scales[j + 8u], 4u, 4u) | (extract_bits(block.scales[j + 4u], 6u, 2u) << 4));
  const uint packed = block.qs[(segment >> 1) * 32u + lane];
  const uint q = (segment & 1u) == 0u ? extract_bits(packed, 0u, 4u) : extract_bits(packed, 4u, 4u);
  return half(float(block.d) * float(scale) * float(q)
      - float(block.dmin) * float(minimum));
}

inline half llmopt_embedding_q5_k_value(
    device const embedding_block_q5_K& block, uint within) {
  const uint segment = within >> 5;
  const uint lane = within & 31u;
  const uint j = segment & 3u;
  const uint scale = segment < 4u
      ? uint(extract_bits(block.scales[j], 0u, 6u))
      : uint(extract_bits(block.scales[j + 8u], 0u, 4u) | (extract_bits(block.scales[j], 6u, 2u) << 4));
  const uint minimum = segment < 4u
      ? uint(extract_bits(block.scales[j + 4u], 0u, 6u))
      : uint(extract_bits(block.scales[j + 8u], 4u, 4u) | (extract_bits(block.scales[j + 4u], 6u, 2u) << 4));
  const uint packed = block.qs[(segment >> 1) * 32u + lane];
  const uint low = (segment & 1u) == 0u ? extract_bits(packed, 0u, 4u) : extract_bits(packed, 4u, 4u);
  const uint q = low | (extract_bits(uint(block.qh[lane]), segment, 1u) << 4);
  return half(float(block.d) * float(scale) * float(q)
      - float(block.dmin) * float(minimum));
}

inline half llmopt_embedding_q6_k_value(
    device const embedding_block_q6_K& block, uint within) {
  const uint segment = within >> 5;
  const uint lane = within & 31u;
  const uint half_index = segment >> 2;
  const uint local = segment & 3u;
  const uint ql_base = half_index * 64u + ((local & 1u) * 32u);
  const uint qh_base = half_index * 32u;
  const uint low = (local < 2u)
      ? extract_bits(uint(block.ql[ql_base + lane]), 0u, 4u)
      : extract_bits(uint(block.ql[ql_base + lane]), 4u, 4u);
  const uint high = extract_bits(uint(block.qh[qh_base + lane]), local * 2u, 2u);
  const int q = int(low | (high << 4)) - 32;
  const uint scale_index = half_index * 8u + (lane >> 4) + (local * 2u);
  return half(float(block.d) * float(block.scales[scale_index]) * float(q));
}

kernel void llmopt_embedding_f16(
    device const long* indices [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant EmbeddingParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint count = params.tokens * params.width;
  if (gid >= count) return;
  const uint token = gid / params.width;
  const uint dimension = gid % params.width;
  const long index = indices[token];
  output[gid] = (index >= 0 && index < long(params.vocabulary))
      ? weight[ulong(index) * params.width + dimension]
      : half(0.0h);
}

kernel void llmopt_embedding_q8_0(
    device const long* indices [[buffer(0)]],
    device const embedding_block_q8_0* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant EmbeddingParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint count = params.tokens * params.width;
  if (gid >= count) return;
  const uint token = gid / params.width;
  const uint dimension = gid % params.width;
  const long index = indices[token];
  if (index < 0 || index >= long(params.vocabulary)) { output[gid] = half(0.0h); return; }
  const ulong element = ulong(index) * params.width + dimension;
  device const embedding_block_q8_0& block = weight[element >> 5];
  output[gid] = half(float(block.d) * float(block.qs[element & 31u]));
}

kernel void llmopt_embedding_q5_0(
    device const long* indices [[buffer(0)]],
    device const embedding_block_q5_0* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant EmbeddingParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint count = params.tokens * params.width;
  if (gid >= count) return;
  const uint token = gid / params.width;
  const uint dimension = gid % params.width;
  const long index = indices[token];
  if (index < 0 || index >= long(params.vocabulary)) { output[gid] = half(0.0h); return; }
  const ulong element = ulong(index) * params.width + dimension;
  const uint within = uint(element & 31u);
  device const embedding_block_q5_0& block = weight[element >> 5];
  const uint high = (block.qh[within >> 3] >> (within & 7u)) & 1u;
  const uint packed = block.qs[within & 15u];
  const uint low = within < 16u ? packed & 15u : packed >> 4;
  output[gid] = half(float(block.d) * float(int(low | (high << 4)) - 16));
}

kernel void llmopt_embedding_q4_0(
    device const long* indices [[buffer(0)]],
    device const embedding_block_q4_0* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant EmbeddingParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint count = params.tokens * params.width;
  if (gid >= count) return;
  const uint token = gid / params.width;
  const uint dimension = gid % params.width;
  const long index = indices[token];
  if (index < 0 || index >= long(params.vocabulary)) { output[gid] = half(0.0h); return; }
  const ulong element = ulong(index) * params.width + dimension;
  const uint within = uint(element & 31u);
  device const embedding_block_q4_0& block = weight[element >> 5];
  const uint packed = block.qs[within & 15u];
  const uint q = within < 16u ? packed & 15u : packed >> 4;
  output[gid] = half(float(block.d) * float(int(q) - 8));
}

kernel void llmopt_embedding_q4_k(
    device const long* indices [[buffer(0)]],
    device const embedding_block_q4_K* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant EmbeddingParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint count = params.tokens * params.width;
  if (gid >= count) return;
  const uint token = gid / params.width;
  const uint dimension = gid % params.width;
  const long index = indices[token];
  if (index < 0 || index >= long(params.vocabulary)) { output[gid] = half(0.0h); return; }
  const ulong element = ulong(index) * params.width + dimension;
  output[gid] = llmopt_embedding_q4_k_value(weight[element >> 8], uint(element & 255u));
}

kernel void llmopt_embedding_q5_k(
    device const long* indices [[buffer(0)]],
    device const embedding_block_q5_K* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant EmbeddingParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint count = params.tokens * params.width;
  if (gid >= count) return;
  const uint token = gid / params.width;
  const uint dimension = gid % params.width;
  const long index = indices[token];
  if (index < 0 || index >= long(params.vocabulary)) { output[gid] = half(0.0h); return; }
  const ulong element = ulong(index) * params.width + dimension;
  output[gid] = llmopt_embedding_q5_k_value(weight[element >> 8], uint(element & 255u));
}

kernel void llmopt_embedding_q6_k(
    device const long* indices [[buffer(0)]],
    device const embedding_block_q6_K* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant EmbeddingParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint count = params.tokens * params.width;
  if (gid >= count) return;
  const uint token = gid / params.width;
  const uint dimension = gid % params.width;
  const long index = indices[token];
  if (index < 0 || index >= long(params.vocabulary)) { output[gid] = half(0.0h); return; }
  const ulong element = ulong(index) * params.width + dimension;
  output[gid] = llmopt_embedding_q6_k_value(weight[element >> 8], uint(element & 255u));
}

kernel void llmopt_chain_token_id(
    device const uint* argmax_in [[buffer(0)]],
    device long* token_out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid == 0) token_out[0] = (long)(argmax_in[0]);
}
|}

let embedding_entries =
  ([ "llmopt_embedding_f16";
     "llmopt_embedding_q8_0";
     "llmopt_embedding_q5_0";
     "llmopt_embedding_q4_0";
     "llmopt_embedding_q4_k";
     "llmopt_embedding_q5_k";
     "llmopt_embedding_q6_k" ]
  |> List.map (fun name ->
         kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1) ~name
           ~operation:Kernel_abi.Operation.Embedding
           ~input_dtype:Ir.Dtype.Int64 ~output_dtype:Ir.Dtype.Float16))
  @ [ kernel_entry_with_threadgroup ~threadgroup:(32, 1, 1)
        ~name:"llmopt_chain_token_id" ~operation:Kernel_abi.Operation.Embedding
        ~input_dtype:Ir.Dtype.Int64 ~output_dtype:Ir.Dtype.Int64 ]

let arange_source =
  "\nstruct ArangeParams { uint count; long start; long step; };\n\n"
  ^ "kernel void llmopt_arange_i64(\n"
  ^ "    device long* output [[buffer(0)]],\n"
  ^ "    constant ArangeParams& params [[buffer(1)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  if (gid < params.count)\n"
  ^ "    output[gid] = params.start + long(gid) * params.step;\n"
  ^ "}\n"

let arange_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_arange_i64" ~operation:Kernel_abi.Operation.Arange
      ~input_dtype:Ir.Dtype.Int64 ~output_dtype:Ir.Dtype.Int64 ]

let diff_source =
  "\nstruct DiffParams {\n"
  ^ "  uint outer; uint source_width; uint prepend_width; uint inner;\n"
  ^ "};\n\n"
  ^ "kernel void llmopt_diff_i64(\n"
  ^ "    device const long* source [[buffer(0)]],\n"
  ^ "    device const long* prepend [[buffer(1)]],\n"
  ^ "    device long* output [[buffer(2)]],\n"
  ^ "    constant DiffParams& params [[buffer(3)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  const uint output_width = params.source_width + params.prepend_width - 1;\n"
  ^ "  const uint count = params.outer * output_width * params.inner;\n"
  ^ "  if (gid >= count) return;\n"
  ^ "  const uint inner_index = gid % params.inner;\n"
  ^ "  const uint axis_index = (gid / params.inner) % output_width;\n"
  ^ "  const uint outer_index = gid / (params.inner * output_width);\n"
  ^ "  const uint left_axis = axis_index;\n"
  ^ "  const uint right_axis = axis_index + 1;\n"
  ^ "  const long left = left_axis < params.prepend_width\n"
  ^ "      ? prepend[(outer_index * params.prepend_width + left_axis)\n"
  ^ "          * params.inner + inner_index]\n"
  ^ "      : source[(outer_index * params.source_width\n"
  ^ "          + left_axis - params.prepend_width) * params.inner + inner_index];\n"
  ^ "  const long right = right_axis < params.prepend_width\n"
  ^ "      ? prepend[(outer_index * params.prepend_width + right_axis)\n"
  ^ "          * params.inner + inner_index]\n"
  ^ "      : source[(outer_index * params.source_width\n"
  ^ "          + right_axis - params.prepend_width) * params.inner + inner_index];\n"
  ^ "  output[gid] = right - left;\n"
  ^ "}\n"

let diff_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_diff_i64" ~operation:Kernel_abi.Operation.Diff
      ~input_dtype:Ir.Dtype.Int64 ~output_dtype:Ir.Dtype.Int64 ]

let cumsum_source =
  "\nstruct CumsumParams { uint outer; uint width; uint inner; };\n\n"
  ^ "kernel void llmopt_cumsum_bool_i64(\n"
  ^ "    device const uchar* input [[buffer(0)]],\n"
  ^ "    device long* output [[buffer(1)]],\n"
  ^ "    constant CumsumParams& params [[buffer(2)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  const uint lanes = params.outer * params.inner;\n"
  ^ "  if (gid >= lanes) return;\n"
  ^ "  const uint outer_index = gid / params.inner;\n"
  ^ "  const uint inner_index = gid % params.inner;\n"
  ^ "  long accumulator = 0;\n"
  ^ "  for (uint axis_index = 0; axis_index < params.width; ++axis_index) {\n"
  ^ "    const uint index = ((outer_index * params.width + axis_index)\n"
  ^ "        * params.inner) + inner_index;\n"
  ^ "    accumulator += input[index] != 0 ? 1 : 0;\n"
  ^ "    output[index] = accumulator;\n"
  ^ "  }\n"
  ^ "}\n"
  ^ "\nkernel void llmopt_cumsum_f32(\n"
  ^ "    device const float* input [[buffer(0)]],\n"
  ^ "    device float* output [[buffer(1)]],\n"
  ^ "    constant CumsumParams& params [[buffer(2)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  const uint lanes = params.outer * params.inner;\n"
  ^ "  if (gid >= lanes) return;\n"
  ^ "  const uint outer_index = gid / params.inner;\n"
  ^ "  const uint inner_index = gid % params.inner;\n"
  ^ "  float accumulator = 0.0f;\n"
  ^ "  for (uint axis_index = 0; axis_index < params.width; ++axis_index) {\n"
  ^ "    const uint index = ((outer_index * params.width + axis_index)\n"
  ^ "        * params.inner) + inner_index;\n"
  ^ "    accumulator += input[index];\n"
  ^ "    output[index] = accumulator;\n"
  ^ "  }\n"
  ^ "}\n"

let cumsum_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_cumsum_bool_i64" ~operation:Kernel_abi.Operation.Cumsum
      ~input_dtype:Ir.Dtype.Bool ~output_dtype:Ir.Dtype.Int64;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_cumsum_f32" ~operation:Kernel_abi.Operation.Cumsum
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32 ]

let tensor_primitive_source = {|
struct PadRightZeroParams {
  uint count; uint input_axis; uint output_axis; uint inner;
};

kernel void llmopt_pad_right_zero_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant PadRightZeroParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  const uint inner_index = gid % params.inner;
  const uint axis_index = (gid / params.inner) % params.output_axis;
  const uint outer_index = gid / (params.inner * params.output_axis);
  if (axis_index < params.input_axis) {
    const uint input_index = ((outer_index * params.input_axis + axis_index)
        * params.inner) + inner_index;
    output[gid] = input[input_index];
  } else {
    output[gid] = 0.0f;
  }
}

struct TriangularParams {
  uint count; uint rows; uint cols; uint upper; int diagonal;
};

kernel void llmopt_triangular_bool(
    device const uchar* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant TriangularParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  const int row = int((gid / params.cols) % params.rows);
  const int col = int(gid % params.cols);
  const bool keep = params.upper != 0
      ? col - row >= params.diagonal : col - row <= params.diagonal;
  output[gid] = keep ? input[gid] : 0;
}

kernel void llmopt_triangular_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant TriangularParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  const int row = int((gid / params.cols) % params.rows);
  const int col = int(gid % params.cols);
  const bool keep = params.upper != 0
      ? col - row >= params.diagonal : col - row <= params.diagonal;
  output[gid] = keep ? input[gid] : 0.0f;
}

struct MaskedFillParams {
  uint count; uint rank; uint left_scalar; uint right_scalar;
  uint output_shape[8]; uint input_shape[8]; uint mask_shape[8];
  long left_i64; long right_i64; float left_f32; float fill_value;
};

kernel void llmopt_masked_fill_f32(
    device const float* input [[buffer(0)]],
    device const uchar* mask [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant MaskedFillParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.count) return;
  uint remaining = gid;
  uint mask_offset = 0;
  uint mask_stride = 1;
  for (int axis = int(params.rank) - 1; axis >= 0; --axis) {
    const uint coordinate = remaining % params.output_shape[axis];
    remaining /= params.output_shape[axis];
    const uint dimension = params.mask_shape[axis];
    mask_offset += (dimension == 1 ? 0 : coordinate) * mask_stride;
    mask_stride *= dimension;
  }
  output[gid] = mask[mask_offset] != 0 ? params.fill_value : input[gid];
}

struct EyeParams { uint count; uint rows; uint cols; };

kernel void llmopt_eye_f32(
    device float* output [[buffer(0)]],
    constant EyeParams& params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid < params.count)
    output[gid] = gid / params.cols == gid % params.cols ? 1.0f : 0.0f;
}

struct BatchedMatmulParams {
  uint count; uint m; uint n; uint k; uint rank;
  uint batch_shape[3]; uint lhs_batch_shape[3]; uint rhs_batch_shape[3];
};

kernel void llmopt_batched_matmul_f32_tiled(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant BatchedMatmulParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
  const uint col = gid.x;
  const uint row = gid.y;
  uint batch_index = gid.z;
  uint coordinates[3] = {};
  for (int axis = int(params.rank) - 1; axis >= 0; --axis) {
    coordinates[axis] = batch_index % params.batch_shape[axis];
    batch_index /= params.batch_shape[axis];
  }
  uint lhs_batch = 0;
  uint rhs_batch = 0;
  for (uint axis = 0; axis < params.rank; ++axis) {
    lhs_batch = lhs_batch * params.lhs_batch_shape[axis]
        + (params.lhs_batch_shape[axis] == 1 ? 0 : coordinates[axis]);
    rhs_batch = rhs_batch * params.rhs_batch_shape[axis]
        + (params.rhs_batch_shape[axis] == 1 ? 0 : coordinates[axis]);
  }
  threadgroup float lhs_tile[16][16];
  threadgroup float rhs_tile[16][16];
  float accumulator = 0.0f;
  for (uint base = 0; base < params.k; base += 16) {
    const uint lhs_inner = base + tid.x;
    const uint rhs_inner = base + tid.y;
    lhs_tile[tid.y][tid.x] = row < params.m && lhs_inner < params.k
        ? lhs[(lhs_batch * params.m + row) * params.k + lhs_inner]
        : 0.0f;
    rhs_tile[tid.y][tid.x] = rhs_inner < params.k && col < params.n
        ? rhs[(rhs_batch * params.k + rhs_inner) * params.n + col]
        : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint inner = 0; inner < 16 && base + inner < params.k; ++inner)
      accumulator += lhs_tile[tid.y][inner] * rhs_tile[inner][tid.x];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < params.m && col < params.n)
    output[(gid.z * params.m + row) * params.n + col] = accumulator;
}

kernel void llmopt_batched_matmul_f32_simd8(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant BatchedMatmulParams& params [[buffer(3)]],
    uint3 group_position [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row_base = group_position.y * 8;
  const uint col_base = group_position.x * 8;
  uint batch_index = group_position.z;
  uint coordinates[3] = {};
  for (int axis = int(params.rank) - 1; axis >= 0; --axis) {
    coordinates[axis] = batch_index % params.batch_shape[axis];
    batch_index /= params.batch_shape[axis];
  }
  uint lhs_batch = 0;
  uint rhs_batch = 0;
  for (uint axis = 0; axis < params.rank; ++axis) {
    lhs_batch = lhs_batch * params.lhs_batch_shape[axis]
        + (params.lhs_batch_shape[axis] == 1 ? 0 : coordinates[axis]);
    rhs_batch = rhs_batch * params.rhs_batch_shape[axis]
        + (params.rhs_batch_shape[axis] == 1 ? 0 : coordinates[axis]);
  }
  threadgroup float lhs_tile[8][8];
  threadgroup float rhs_tile[8][8];
  simdgroup_float8x8 accumulator =
      make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  const uint element0 = lane * 2;
  const uint element1 = element0 + 1;
  const uint row0 = element0 / 8;
  const uint col0 = element0 % 8;
  const uint row1 = element1 / 8;
  const uint col1 = element1 % 8;
  for (uint inner_base = 0; inner_base < params.k; inner_base += 8) {
    const uint lhs_row0 = row_base + row0;
    const uint lhs_inner0 = inner_base + col0;
    lhs_tile[row0][col0] = lhs_row0 < params.m && lhs_inner0 < params.k
        ? lhs[(lhs_batch * params.m + lhs_row0) * params.k + lhs_inner0]
        : 0.0f;
    const uint lhs_row1 = row_base + row1;
    const uint lhs_inner1 = inner_base + col1;
    lhs_tile[row1][col1] = lhs_row1 < params.m && lhs_inner1 < params.k
        ? lhs[(lhs_batch * params.m + lhs_row1) * params.k + lhs_inner1]
        : 0.0f;
    const uint rhs_inner0 = inner_base + row0;
    const uint rhs_col0 = col_base + col0;
    rhs_tile[row0][col0] = rhs_inner0 < params.k && rhs_col0 < params.n
        ? rhs[(rhs_batch * params.k + rhs_inner0) * params.n + rhs_col0]
        : 0.0f;
    const uint rhs_inner1 = inner_base + row1;
    const uint rhs_col1 = col_base + col1;
    rhs_tile[row1][col1] = rhs_inner1 < params.k && rhs_col1 < params.n
        ? rhs[(rhs_batch * params.k + rhs_inner1) * params.n + rhs_col1]
        : 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 lhs_matrix;
    simdgroup_float8x8 rhs_matrix;
    simdgroup_load(lhs_matrix, &lhs_tile[0][0], 8);
    simdgroup_load(rhs_matrix, &rhs_tile[0][0], 8);
    simdgroup_multiply_accumulate(
        accumulator, lhs_matrix, rhs_matrix, accumulator);
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  threadgroup float output_tile[8][8];
  simdgroup_store(accumulator, &output_tile[0][0], 8);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const uint output_row0 = row_base + row0;
  const uint output_col0 = col_base + col0;
  if (output_row0 < params.m && output_col0 < params.n)
    output[(group_position.z * params.m + output_row0) * params.n + output_col0]
        = output_tile[row0][col0];
  const uint output_row1 = row_base + row1;
  const uint output_col1 = col_base + col1;
  if (output_row1 < params.m && output_col1 < params.n)
    output[(group_position.z * params.m + output_row1) * params.n + output_col1]
        = output_tile[row1][col1];
}
|}

let tensor_primitive_entries =
  let entry name operation input_dtype output_dtype =
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1) ~name ~operation
      ~input_dtype ~output_dtype
  in
  [ entry "llmopt_pad_right_zero_f32" Kernel_abi.Operation.Movement
      Ir.Dtype.Float32 Ir.Dtype.Float32;
    entry "llmopt_triangular_bool" Kernel_abi.Operation.Pointwise
      Ir.Dtype.Bool Ir.Dtype.Bool;
    entry "llmopt_triangular_f32" Kernel_abi.Operation.Pointwise
      Ir.Dtype.Float32 Ir.Dtype.Float32;
    entry "llmopt_masked_fill_f32" Kernel_abi.Operation.Pointwise
      Ir.Dtype.Float32 Ir.Dtype.Float32;
    entry "llmopt_eye_f32" Kernel_abi.Operation.Eye Ir.Dtype.Float32
      Ir.Dtype.Float32;
    kernel_entry_with_threadgroup ~threadgroup:(16, 16, 1)
      ~name:"llmopt_batched_matmul_f32_tiled"
      ~operation:Kernel_abi.Operation.Batched_matmul
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32;
    kernel_entry_with_threadgroup ~threadgroup:(32, 1, 1)
      ~name:"llmopt_batched_matmul_f32_simd8"
      ~operation:Kernel_abi.Operation.Batched_matmul
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32 ]

let fill_source =
  "\nstruct FillBoolParams { uint count; uint value; };\n\n"
  ^ "kernel void llmopt_fill_bool(\n"
  ^ "    device uchar* output [[buffer(0)]],\n"
  ^ "    constant FillBoolParams& params [[buffer(1)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  if (gid < params.count) output[gid] = params.value != 0 ? 1 : 0;\n"
  ^ "}\n\n"
  ^ "struct FillFloatParams { uint count; float value; };\n\n"
  ^ "kernel void llmopt_fill_f16(\n"
  ^ "    device half* output [[buffer(0)]],\n"
  ^ "    constant FillFloatParams& params [[buffer(1)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  if (gid < params.count) output[gid] = half(params.value);\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_fill_f32(\n"
  ^ "    device float* output [[buffer(0)]],\n"
  ^ "    constant FillFloatParams& params [[buffer(1)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  if (gid < params.count) output[gid] = params.value;\n"
  ^ "}\n"

let fill_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_fill_bool" ~operation:Kernel_abi.Operation.Fill
      ~input_dtype:Ir.Dtype.Bool ~output_dtype:Ir.Dtype.Bool;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_fill_f16" ~operation:Kernel_abi.Operation.Fill
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_fill_f32" ~operation:Kernel_abi.Operation.Fill
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32 ]

let gather2_source =
  "\nstruct Gather2Params {\n"
  ^ "  uint count; uint rows; uint cols;\n"
  ^ "  uint output0; uint output1; uint output2; uint output3;\n"
  ^ "  uint first0; uint first1; uint first2; uint first3;\n"
  ^ "  uint second0; uint second1; uint second2; uint second3;\n"
  ^ "};\n\n"
  ^ "kernel void llmopt_gather2_i64(\n"
  ^ "    device const long* source [[buffer(0)]],\n"
  ^ "    device const long* first_index [[buffer(1)]],\n"
  ^ "    device const long* second_index [[buffer(2)]],\n"
  ^ "    device long* output [[buffer(3)]],\n"
  ^ "    constant Gather2Params& params [[buffer(4)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  if (gid >= params.count) return;\n"
  ^ "  uint remaining = gid;\n"
  ^ "  const uint coordinate3 = remaining % params.output3;\n"
  ^ "  remaining /= params.output3;\n"
  ^ "  const uint coordinate2 = remaining % params.output2;\n"
  ^ "  remaining /= params.output2;\n"
  ^ "  const uint coordinate1 = remaining % params.output1;\n"
  ^ "  const uint coordinate0 = remaining / params.output1;\n"
  ^ "  const uint first_offset = (((\n"
  ^ "      (params.first0 == 1 ? 0 : coordinate0) * params.first1\n"
  ^ "      + (params.first1 == 1 ? 0 : coordinate1)) * params.first2\n"
  ^ "      + (params.first2 == 1 ? 0 : coordinate2)) * params.first3\n"
  ^ "      + (params.first3 == 1 ? 0 : coordinate3));\n"
  ^ "  const uint second_offset = (((\n"
  ^ "      (params.second0 == 1 ? 0 : coordinate0) * params.second1\n"
  ^ "      + (params.second1 == 1 ? 0 : coordinate1)) * params.second2\n"
  ^ "      + (params.second2 == 1 ? 0 : coordinate2)) * params.second3\n"
  ^ "      + (params.second3 == 1 ? 0 : coordinate3));\n"
  ^ "  const long row = first_index[first_offset];\n"
  ^ "  const long col = second_index[second_offset];\n"
  ^ "  output[gid] = row >= 0 && row < long(params.rows)\n"
  ^ "      && col >= 0 && col < long(params.cols)\n"
  ^ "      ? source[ulong(row) * params.cols + ulong(col)] : 0;\n"
  ^ "}\n"

let gather2_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_gather2_i64" ~operation:Kernel_abi.Operation.Gather2
      ~input_dtype:Ir.Dtype.Int64 ~output_dtype:Ir.Dtype.Int64 ]

let cast_source =
  "\nstruct CastParams { uint count; };\n\n"
  ^ "kernel void llmopt_cast_f16_f32(\n"
  ^ "    device const half* input [[buffer(0)]],\n"
  ^ "    device float* output [[buffer(1)]],\n"
  ^ "    constant CastParams& params [[buffer(2)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  if (gid < params.count) output[gid] = float(input[gid]);\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_cast_f32_f16(\n"
  ^ "    device const float* input [[buffer(0)]],\n"
  ^ "    device half* output [[buffer(1)]],\n"
  ^ "    constant CastParams& params [[buffer(2)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  if (gid < params.count) output[gid] = half(input[gid]);\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_cast_i64_f32(\n"
  ^ "    device const long* input [[buffer(0)]],\n"
  ^ "    device float* output [[buffer(1)]],\n"
  ^ "    constant CastParams& params [[buffer(2)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  if (gid < params.count) output[gid] = float(input[gid]);\n"
  ^ "}\n"

let cast_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_cast_f16_f32" ~operation:Kernel_abi.Operation.Cast
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_cast_f32_f16" ~operation:Kernel_abi.Operation.Cast
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_cast_i64_f32" ~operation:Kernel_abi.Operation.Cast
      ~input_dtype:Ir.Dtype.Int64 ~output_dtype:Ir.Dtype.Float32 ]

let pointwise_binary_kernel ~name ~input_type ~output_type ~scalar_field
    ~scalar_cast ~expression =
  "kernel void " ^ name ^ "(\n"
  ^ "    device const " ^ input_type ^ "* left [[buffer(0)]],\n"
  ^ "    device const " ^ input_type ^ "* right [[buffer(1)]],\n"
  ^ "    device " ^ output_type ^ "* output [[buffer(2)]],\n"
  ^ "    constant PointwiseParams& params [[buffer(3)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  if (gid >= params.count) return;\n"
  ^ "  const " ^ input_type ^ " left_value = params.left_scalar != 0\n"
  ^ "      ? " ^ scalar_cast ^ "(params.left_" ^ scalar_field ^ ")\n"
  ^ "      : left[llmopt_pointwise_offset(gid, params, true)];\n"
  ^ "  const " ^ input_type ^ " right_value = params.right_scalar != 0\n"
  ^ "      ? " ^ scalar_cast ^ "(params.right_" ^ scalar_field ^ ")\n"
  ^ "      : right[llmopt_pointwise_offset(gid, params, false)];\n"
  ^ "  output[gid] = " ^ expression ^ ";\n"
  ^ "}\n\n"

let pointwise_unary_kernel ~name ~input_type ~output_type ~expression =
  "kernel void " ^ name ^ "(\n"
  ^ "    device const " ^ input_type ^ "* input [[buffer(0)]],\n"
  ^ "    device " ^ output_type ^ "* output [[buffer(1)]],\n"
  ^ "    constant UnaryParams& params [[buffer(2)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  if (gid < params.count) {\n"
  ^ "    const " ^ input_type ^ " value = input[gid];\n"
  ^ "    output[gid] = " ^ expression ^ ";\n"
  ^ "  }\n"
  ^ "}\n\n"

let pointwise_mixed_mul_f16_f32 =
  "kernel void llmopt_mul_f16_f32(\n"
  ^ "    device const half* left [[buffer(0)]],\n"
  ^ "    device const float* right [[buffer(1)]],\n"
  ^ "    device float* output [[buffer(2)]],\n"
  ^ "    constant PointwiseParams& params [[buffer(3)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  if (gid >= params.count) return;\n"
  ^ "  const uint left_offset = llmopt_pointwise_offset(gid, params, true);\n"
  ^ "  const uint right_offset = llmopt_pointwise_offset(gid, params, false);\n"
  ^ "  output[gid] = float(left[left_offset]) * right[right_offset];\n"
  ^ "}\n\n"

let pointwise_mixed_mul_f16_f32_f16 =
  "kernel void llmopt_mul_f16_f32_f16(\n"
  ^ "    device const half* left [[buffer(0)]],\n"
  ^ "    device const float* right [[buffer(1)]],\n"
  ^ "    device half* output [[buffer(2)]],\n"
  ^ "    constant PointwiseParams& params [[buffer(3)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  if (gid >= params.count) return;\n"
  ^ "  const uint left_offset = llmopt_pointwise_offset(gid, params, true);\n"
  ^ "  const uint right_offset = llmopt_pointwise_offset(gid, params, false);\n"
  ^ "  output[gid] = half(float(left[left_offset]) * right[right_offset]);\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_mul_f32_f16_f16(\n"
  ^ "    device const float* left [[buffer(0)]],\n"
  ^ "    device const half* right [[buffer(1)]],\n"
  ^ "    device half* output [[buffer(2)]],\n"
  ^ "    constant PointwiseParams& params [[buffer(3)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  if (gid >= params.count) return;\n"
  ^ "  const uint left_offset = llmopt_pointwise_offset(gid, params, true);\n"
  ^ "  const uint right_offset = llmopt_pointwise_offset(gid, params, false);\n"
  ^ "  output[gid] = half(left[left_offset] * float(right[right_offset]));\n"
  ^ "}\n\n"

let pointwise_source =
  "\nstruct PointwiseParams {\n"
  ^ "  uint count; uint rank; uint left_scalar; uint right_scalar;\n"
  ^ "  uint output_shape[8]; uint left_shape[8]; uint right_shape[8];\n"
  ^ "  long left_i64; long right_i64; float left_f32; float right_f32;\n"
  ^ "};\n\n"
  ^ "struct UnaryParams { uint count; };\n\n"
  ^ "inline uint llmopt_pointwise_offset(uint gid,\n"
  ^ "    constant PointwiseParams& params, bool is_left) {\n"
  ^ "  uint remaining = gid;\n"
  ^ "  uint offset = 0;\n"
  ^ "  uint stride = 1;\n"
  ^ "  for (int axis = int(params.rank) - 1; axis >= 0; --axis) {\n"
  ^ "    const uint output_dimension = params.output_shape[axis];\n"
  ^ "    const uint coordinate = remaining % output_dimension;\n"
  ^ "    remaining /= output_dimension;\n"
  ^ "    const uint input_dimension = is_left\n"
  ^ "        ? params.left_shape[axis] : params.right_shape[axis];\n"
  ^ "    offset += (input_dimension == 1 ? 0 : coordinate) * stride;\n"
  ^ "    stride *= input_dimension;\n"
  ^ "  }\n"
  ^ "  return offset;\n"
  ^ "}\n\n"
  ^ pointwise_binary_kernel ~name:"llmopt_add_f16" ~input_type:"half"
      ~output_type:"half" ~scalar_field:"f32" ~scalar_cast:"half"
      ~expression:"left_value + right_value"
  ^ pointwise_binary_kernel ~name:"llmopt_add_f32" ~input_type:"float"
      ~output_type:"float" ~scalar_field:"f32" ~scalar_cast:"float"
      ~expression:"left_value + right_value"
  ^ pointwise_binary_kernel ~name:"llmopt_add_i64" ~input_type:"long"
      ~output_type:"long" ~scalar_field:"i64" ~scalar_cast:"long"
      ~expression:"left_value + right_value"
  ^ pointwise_binary_kernel ~name:"llmopt_sub_i64" ~input_type:"long"
      ~output_type:"long" ~scalar_field:"i64" ~scalar_cast:"long"
      ~expression:"left_value - right_value"
  ^ pointwise_binary_kernel ~name:"llmopt_sub_f32" ~input_type:"float"
      ~output_type:"float" ~scalar_field:"f32" ~scalar_cast:"float"
      ~expression:"left_value - right_value"
  ^ pointwise_binary_kernel ~name:"llmopt_div_f16" ~input_type:"half"
      ~output_type:"half" ~scalar_field:"f32" ~scalar_cast:"half"
      ~expression:"left_value / right_value"
  ^ pointwise_binary_kernel ~name:"llmopt_mul_f16" ~input_type:"half"
      ~output_type:"half" ~scalar_field:"f32" ~scalar_cast:"half"
      ~expression:"left_value * right_value"
  ^ pointwise_binary_kernel ~name:"llmopt_mul_f32" ~input_type:"float"
      ~output_type:"float" ~scalar_field:"f32" ~scalar_cast:"float"
      ~expression:"left_value * right_value"
  ^ pointwise_binary_kernel ~name:"llmopt_silu_mul_f16" ~input_type:"half"
      ~output_type:"half" ~scalar_field:"f32" ~scalar_cast:"half"
      ~expression:
        "half(float(left_value) / (1.0f + exp(-float(left_value)))) * right_value"
  ^ pointwise_binary_kernel ~name:"llmopt_gelu_mul_f16" ~input_type:"half"
      ~output_type:"half" ~scalar_field:"f32" ~scalar_cast:"half"
      ~expression:
        "half(0.5f * float(left_value) * (1.0f + tanh(clamp(0.7978845608f * (float(left_value) + 0.044715f * float(left_value) * float(left_value) * float(left_value)), -10.0f, 10.0f)))) * right_value"
  ^ pointwise_binary_kernel ~name:"llmopt_sigmoid_mul_f16" ~input_type:"half"
      ~output_type:"half" ~scalar_field:"f32" ~scalar_cast:"half"
      ~expression:
        "half(1.0f / (1.0f + exp(-float(left_value)))) * right_value"
  ^ pointwise_mixed_mul_f16_f32
  ^ pointwise_mixed_mul_f16_f32_f16
  ^ pointwise_binary_kernel ~name:"llmopt_le_i64" ~input_type:"long"
      ~output_type:"uchar" ~scalar_field:"i64" ~scalar_cast:"long"
      ~expression:"left_value <= right_value ? 1 : 0"
  ^ pointwise_binary_kernel ~name:"llmopt_gt_i64" ~input_type:"long"
      ~output_type:"uchar" ~scalar_field:"i64" ~scalar_cast:"long"
      ~expression:"left_value > right_value ? 1 : 0"
  ^ pointwise_binary_kernel ~name:"llmopt_eq_i64" ~input_type:"long"
      ~output_type:"uchar" ~scalar_field:"i64" ~scalar_cast:"long"
      ~expression:"left_value == right_value ? 1 : 0"
  ^ pointwise_binary_kernel ~name:"llmopt_ne_i64" ~input_type:"long"
      ~output_type:"uchar" ~scalar_field:"i64" ~scalar_cast:"long"
      ~expression:"left_value != right_value ? 1 : 0"
  ^ pointwise_binary_kernel ~name:"llmopt_and_bool" ~input_type:"uchar"
      ~output_type:"uchar" ~scalar_field:"i64" ~scalar_cast:"uchar"
      ~expression:"(left_value != 0 && right_value != 0) ? 1 : 0"
  ^ pointwise_unary_kernel ~name:"llmopt_neg_f16" ~input_type:"half"
      ~output_type:"half" ~expression:"-value"
  ^ pointwise_unary_kernel ~name:"llmopt_neg_f32" ~input_type:"float"
      ~output_type:"float" ~expression:"-value"
  ^ pointwise_unary_kernel ~name:"llmopt_silu_f16" ~input_type:"half"
      ~output_type:"half"
      ~expression:"half(float(value) / (1.0f + exp(-float(value))))"
  ^ pointwise_unary_kernel ~name:"llmopt_silu_f32" ~input_type:"float"
      ~output_type:"float" ~expression:"value / (1.0f + exp(-value))"
  ^ pointwise_unary_kernel ~name:"llmopt_cos_f32" ~input_type:"float"
      ~output_type:"float" ~expression:"cos(value)"
  ^ pointwise_unary_kernel ~name:"llmopt_sin_f32" ~input_type:"float"
      ~output_type:"float" ~expression:"sin(value)"
  ^ pointwise_unary_kernel ~name:"llmopt_tanh_f16" ~input_type:"half"
      ~output_type:"half" ~expression:"half(tanh(float(value)))"
  ^ pointwise_unary_kernel ~name:"llmopt_exp_f32" ~input_type:"float"
      ~output_type:"float" ~expression:"exp(value)"
  ^ pointwise_unary_kernel ~name:"llmopt_sigmoid_f16" ~input_type:"half"
      ~output_type:"half"
      ~expression:"half(1.0f / (1.0f + exp(-float(value))))"
  ^ pointwise_unary_kernel ~name:"llmopt_softplus_f32" ~input_type:"float"
      ~output_type:"float"
      ~expression:"value > 20.0f ? value : log(1.0f + exp(value))"
  ^ pointwise_unary_kernel ~name:"llmopt_rsqrt_f16" ~input_type:"half"
      ~output_type:"half" ~expression:"half(rsqrt(float(value)))"
  ^ pointwise_unary_kernel ~name:"llmopt_rsqrt_f32" ~input_type:"float"
      ~output_type:"float" ~expression:"rsqrt(value)"
  ^ "kernel void llmopt_pow_f32(\n"
  ^ "    device const float* input [[buffer(0)]],\n"
  ^ "    device float* output [[buffer(1)]],\n"
  ^ "    constant PointwiseParams& params [[buffer(2)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  if (gid < params.count) output[gid] = pow(input[gid], params.left_f32);\n"
  ^ "}\n\n"
  ^ pointwise_unary_kernel ~name:"llmopt_gelu_f16" ~input_type:"half"
      ~output_type:"half"
      ~expression:
        "half(0.5f * float(value) * (1.0f + tanh(clamp(0.7978845608f * (float(value) + 0.044715f * float(value) * float(value) * float(value)), -10.0f, 10.0f))))"

let pointwise_entries =
  let entry name input_dtype output_dtype =
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1) ~name
      ~operation:Kernel_abi.Operation.Pointwise ~input_dtype ~output_dtype
  in
  [ entry "llmopt_add_f16" Ir.Dtype.Float16 Ir.Dtype.Float16;
    entry "llmopt_add_f32" Ir.Dtype.Float32 Ir.Dtype.Float32;
    entry "llmopt_add_i64" Ir.Dtype.Int64 Ir.Dtype.Int64;
    entry "llmopt_sub_i64" Ir.Dtype.Int64 Ir.Dtype.Int64;
    entry "llmopt_sub_f32" Ir.Dtype.Float32 Ir.Dtype.Float32;
    entry "llmopt_div_f16" Ir.Dtype.Float16 Ir.Dtype.Float16;
    entry "llmopt_mul_f16" Ir.Dtype.Float16 Ir.Dtype.Float16;
    entry "llmopt_mul_f32" Ir.Dtype.Float32 Ir.Dtype.Float32;
    entry "llmopt_silu_mul_f16" Ir.Dtype.Float16 Ir.Dtype.Float16;
    entry "llmopt_gelu_mul_f16" Ir.Dtype.Float16 Ir.Dtype.Float16;
    entry "llmopt_sigmoid_mul_f16" Ir.Dtype.Float16 Ir.Dtype.Float16;
    entry "llmopt_mul_f16_f32" Ir.Dtype.Float16 Ir.Dtype.Float32;
    entry "llmopt_mul_f16_f32_f16" Ir.Dtype.Float16 Ir.Dtype.Float16;
    entry "llmopt_mul_f32_f16_f16" Ir.Dtype.Float32 Ir.Dtype.Float16;
    entry "llmopt_le_i64" Ir.Dtype.Int64 Ir.Dtype.Bool;
    entry "llmopt_gt_i64" Ir.Dtype.Int64 Ir.Dtype.Bool;
    entry "llmopt_eq_i64" Ir.Dtype.Int64 Ir.Dtype.Bool;
    entry "llmopt_ne_i64" Ir.Dtype.Int64 Ir.Dtype.Bool;
    entry "llmopt_tanh_f16" Ir.Dtype.Float16 Ir.Dtype.Float16;
    entry "llmopt_exp_f32" Ir.Dtype.Float32 Ir.Dtype.Float32;
    entry "llmopt_sigmoid_f16" Ir.Dtype.Float16 Ir.Dtype.Float16;
    entry "llmopt_softplus_f32" Ir.Dtype.Float32 Ir.Dtype.Float32;
    entry "llmopt_pow_f32" Ir.Dtype.Float32 Ir.Dtype.Float32;
    entry "llmopt_rsqrt_f16" Ir.Dtype.Float16 Ir.Dtype.Float16;
    entry "llmopt_rsqrt_f32" Ir.Dtype.Float32 Ir.Dtype.Float32;
    entry "llmopt_gelu_f16" Ir.Dtype.Float16 Ir.Dtype.Float16;
    entry "llmopt_and_bool" Ir.Dtype.Bool Ir.Dtype.Bool;
    entry "llmopt_neg_f16" Ir.Dtype.Float16 Ir.Dtype.Float16;
    entry "llmopt_neg_f32" Ir.Dtype.Float32 Ir.Dtype.Float32;
    entry "llmopt_silu_f16" Ir.Dtype.Float16 Ir.Dtype.Float16;
    entry "llmopt_silu_f32" Ir.Dtype.Float32 Ir.Dtype.Float32;
    entry "llmopt_cos_f32" Ir.Dtype.Float32 Ir.Dtype.Float32;
    entry "llmopt_sin_f32" Ir.Dtype.Float32 Ir.Dtype.Float32 ]

let movement_unary_kernel ~name ~value_type ~parameters ~body =
  "kernel void " ^ name ^ "(\n"
  ^ "    device const " ^ value_type ^ "* input [[buffer(0)]],\n"
  ^ "    device " ^ value_type ^ "* output [[buffer(1)]],\n"
  ^ "    constant " ^ parameters ^ "& params [[buffer(2)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ body ^ "}\n\n"

let movement_transpose_kernel ~name ~value_type =
  movement_unary_kernel ~name ~value_type ~parameters:"MovementParams"
    ~body:
      ("  if (gid >= params.count) return;\n"
      ^ "  uint coordinates[8] = {};\n"
      ^ "  uint remaining = gid;\n"
      ^ "  for (int axis = int(params.rank) - 1; axis >= 0; --axis) {\n"
      ^ "    coordinates[axis] = remaining % params.output_shape[axis];\n"
      ^ "    remaining /= params.output_shape[axis];\n"
      ^ "  }\n"
      ^ "  uint offset = 0; uint stride = 1;\n"
      ^ "  for (int axis = int(params.rank) - 1; axis >= 0; --axis) {\n"
      ^ "    uint coordinate = coordinates[axis];\n"
      ^ "    if (uint(axis) == params.axis0) coordinate = coordinates[params.axis1];\n"
      ^ "    else if (uint(axis) == params.axis1) coordinate = coordinates[params.axis0];\n"
      ^ "    offset += coordinate * stride;\n"
      ^ "    stride *= params.input_shape[axis];\n"
      ^ "  }\n"
      ^ "  output[gid] = input[offset];\n")

let movement_expand_kernel ~name ~value_type =
  movement_unary_kernel ~name ~value_type ~parameters:"MovementParams"
    ~body:
      ("  if (gid >= params.count) return;\n"
      ^ "  uint remaining = gid; uint offset = 0; uint stride = 1;\n"
      ^ "  for (int axis = int(params.rank) - 1; axis >= 0; --axis) {\n"
      ^ "    const uint coordinate = remaining % params.output_shape[axis];\n"
      ^ "    remaining /= params.output_shape[axis];\n"
      ^ "    const uint input_dimension = params.input_shape[axis];\n"
      ^ "    offset += (input_dimension == 1 ? 0 : coordinate) * stride;\n"
      ^ "    stride *= input_dimension;\n"
      ^ "  }\n"
      ^ "  output[gid] = input[offset];\n")

let movement_index_kernel ~name ~value_type =
  movement_unary_kernel ~name ~value_type ~parameters:"IndexParams"
    ~body:
      ("  if (gid >= params.count) return;\n"
      ^ "  uint output_coordinates[8] = {};\n"
      ^ "  uint remaining = gid;\n"
      ^ "  for (int axis = int(params.output_rank) - 1; axis >= 0; --axis) {\n"
      ^ "    output_coordinates[axis] = remaining % params.output_shape[axis];\n"
      ^ "    remaining /= params.output_shape[axis];\n"
      ^ "  }\n"
      ^ "  long input_coordinates[8] = {};\n"
      ^ "  uint input_axis = 0; uint output_axis = 0;\n"
      ^ "  for (uint selector = 0; selector < params.selector_count; ++selector) {\n"
      ^ "    const uint kind = params.selector_kind[selector];\n"
      ^ "    if (kind == 0) {\n"
      ^ "      input_coordinates[input_axis++] = params.starts[selector];\n"
      ^ "    } else if (kind == 1) {\n"
      ^ "      input_coordinates[input_axis++] = params.starts[selector]\n"
      ^ "          + long(output_coordinates[output_axis++]) * params.steps[selector];\n"
      ^ "    } else {\n"
      ^ "      ++output_axis;\n"
      ^ "    }\n"
      ^ "  }\n"
      ^ "  uint offset = 0; uint stride = 1;\n"
      ^ "  for (int axis = int(params.input_rank) - 1; axis >= 0; --axis) {\n"
      ^ "    offset += uint(input_coordinates[axis]) * stride;\n"
      ^ "    stride *= params.input_shape[axis];\n"
      ^ "  }\n"
      ^ "  output[gid] = input[offset];\n")

let movement_roll_kernel ~name ~value_type =
  movement_unary_kernel ~name ~value_type ~parameters:"RollParams"
    ~body:
      ("  if (gid >= params.count) return;\n"
      ^ "  uint coordinates[8] = {};\n"
      ^ "  uint remaining = gid;\n"
      ^ "  for (int axis = int(params.rank) - 1; axis >= 0; --axis) {\n"
      ^ "    coordinates[axis] = remaining % params.shape[axis];\n"
      ^ "    remaining /= params.shape[axis];\n"
      ^ "  }\n"
      ^ "  const uint width = params.shape[params.axis];\n"
      ^ "  int shifted = (int(coordinates[params.axis]) - params.shift) % int(width);\n"
      ^ "  if (shifted < 0) shifted += int(width);\n"
      ^ "  coordinates[params.axis] = uint(shifted);\n"
      ^ "  uint offset = 0; uint stride = 1;\n"
      ^ "  for (int axis = int(params.rank) - 1; axis >= 0; --axis) {\n"
      ^ "    offset += coordinates[axis] * stride;\n"
      ^ "    stride *= params.shape[axis];\n"
      ^ "  }\n"
      ^ "  output[gid] = input[offset];\n")

let movement_concat_kernel ~name ~value_type =
  "kernel void " ^ name ^ "(\n"
  ^ "    device const " ^ value_type ^ "* left [[buffer(0)]],\n"
  ^ "    device const " ^ value_type ^ "* right [[buffer(1)]],\n"
  ^ "    device " ^ value_type ^ "* output [[buffer(2)]],\n"
  ^ "    constant ConcatParams& params [[buffer(3)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  if (gid >= params.count) return;\n"
  ^ "  uint coordinates[8] = {};\n"
  ^ "  uint remaining = gid;\n"
  ^ "  for (int axis = int(params.rank) - 1; axis >= 0; --axis) {\n"
  ^ "    coordinates[axis] = remaining % params.output_shape[axis];\n"
  ^ "    remaining /= params.output_shape[axis];\n"
  ^ "  }\n"
  ^ "  const bool use_left = coordinates[params.axis] < params.left_axis;\n"
  ^ "  if (!use_left) coordinates[params.axis] -= params.left_axis;\n"
  ^ "  uint offset = 0; uint stride = 1;\n"
  ^ "  for (int axis = int(params.rank) - 1; axis >= 0; --axis) {\n"
  ^ "    const uint dimension = uint(axis) == params.axis\n"
  ^ "        ? (use_left ? params.left_axis\n"
  ^ "                    : params.output_shape[axis] - params.left_axis)\n"
  ^ "        : params.output_shape[axis];\n"
  ^ "    offset += coordinates[axis] * stride;\n"
  ^ "    stride *= dimension;\n"
  ^ "  }\n"
  ^ "  output[gid] = use_left ? left[offset] : right[offset];\n"
  ^ "}\n\n"

let movement_parameters_source =
  "\nstruct MovementParams {\n"
  ^ "  uint count; uint rank; uint axis0; uint axis1;\n"
  ^ "  uint input_shape[8]; uint output_shape[8];\n"
  ^ "};\n\n"
  ^ "struct IndexParams {\n"
  ^ "  uint count; uint input_rank; uint output_rank; uint selector_count;\n"
  ^ "  uint input_shape[8]; uint output_shape[8]; uint selector_kind[8];\n"
  ^ "  long starts[8]; long steps[8];\n"
  ^ "};\n\n"
  ^ "struct ConcatParams {\n"
  ^ "  uint count; uint rank; uint axis; uint left_axis;\n"
  ^ "  uint output_shape[8];\n"
  ^ "};\n\n"
  ^ "struct RollParams {\n"
  ^ "  uint count; uint rank; uint axis; int shift; uint shape[8];\n"
  ^ "};\n\n"

let movement_source =
  movement_transpose_kernel ~name:"llmopt_transpose_f16" ~value_type:"half"
  ^ movement_transpose_kernel ~name:"llmopt_transpose_f32" ~value_type:"float"
  ^ movement_index_kernel ~name:"llmopt_index_f16" ~value_type:"half"
  ^ movement_index_kernel ~name:"llmopt_index_f32" ~value_type:"float"
  ^ movement_index_kernel ~name:"llmopt_index_i64" ~value_type:"long"
  ^ movement_expand_kernel ~name:"llmopt_expand_f16" ~value_type:"half"
  ^ movement_expand_kernel ~name:"llmopt_expand_f32" ~value_type:"float"
  ^ movement_expand_kernel ~name:"llmopt_expand_bool" ~value_type:"uchar"
  ^ movement_expand_kernel ~name:"llmopt_expand_i64" ~value_type:"long"
  ^ movement_concat_kernel ~name:"llmopt_concat_f16" ~value_type:"half"
  ^ movement_concat_kernel ~name:"llmopt_concat_f32" ~value_type:"float"
  ^ movement_roll_kernel ~name:"llmopt_roll_f16" ~value_type:"half"

let movement_entries =
  let entry name dtype =
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1) ~name
      ~operation:Kernel_abi.Operation.Movement ~input_dtype:dtype
      ~output_dtype:dtype
  in
  [ entry "llmopt_transpose_f16" Ir.Dtype.Float16;
    entry "llmopt_transpose_f32" Ir.Dtype.Float32;
    entry "llmopt_index_f16" Ir.Dtype.Float16;
    entry "llmopt_index_f32" Ir.Dtype.Float32;
    entry "llmopt_index_i64" Ir.Dtype.Int64;
    entry "llmopt_expand_f16" Ir.Dtype.Float16;
    entry "llmopt_expand_f32" Ir.Dtype.Float32;
    entry "llmopt_expand_bool" Ir.Dtype.Bool;
    entry "llmopt_expand_i64" Ir.Dtype.Int64;
    entry "llmopt_concat_f16" Ir.Dtype.Float16;
    entry "llmopt_concat_f32" Ir.Dtype.Float32;
    entry "llmopt_roll_f16" Ir.Dtype.Float16 ]

let reduction_source =
  "\nstruct ReductionParams { uint outer; uint width; uint inner; };\n\n"
  ^ "kernel void llmopt_sum_f16(\n"
  ^ "    device const half* input [[buffer(0)]],\n"
  ^ "    device half* output [[buffer(1)]],\n"
  ^ "    constant ReductionParams& params [[buffer(2)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  const uint count = params.outer * params.inner;\n"
  ^ "  if (gid >= count) return;\n"
  ^ "  const uint outer_index = gid / params.inner;\n"
  ^ "  const uint inner_index = gid % params.inner;\n"
  ^ "  float accumulator = 0.0f;\n"
  ^ "  for (uint axis = 0; axis < params.width; ++axis) {\n"
  ^ "    const uint offset = ((outer_index * params.width + axis)\n"
  ^ "        * params.inner) + inner_index;\n"
  ^ "    accumulator += float(input[offset]);\n"
  ^ "  }\n"
  ^ "  output[gid] = half(accumulator);\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_sum_f32(\n"
  ^ "    device const float* input [[buffer(0)]],\n"
  ^ "    device float* output [[buffer(1)]],\n"
  ^ "    constant ReductionParams& params [[buffer(2)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  const uint count = params.outer * params.inner;\n"
  ^ "  if (gid >= count) return;\n"
  ^ "  const uint outer_index = gid / params.inner;\n"
  ^ "  const uint inner_index = gid % params.inner;\n"
  ^ "  float accumulator = 0.0f;\n"
  ^ "  for (uint axis = 0; axis < params.width; ++axis) {\n"
  ^ "    const uint offset = ((outer_index * params.width + axis)\n"
  ^ "        * params.inner) + inner_index;\n"
  ^ "    accumulator += input[offset];\n"
  ^ "  }\n"
  ^ "  output[gid] = accumulator;\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_mean_f32(\n"
  ^ "    device const float* input [[buffer(0)]],\n"
  ^ "    device float* output [[buffer(1)]],\n"
  ^ "    constant ReductionParams& params [[buffer(2)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  const uint count = params.outer * params.inner;\n"
  ^ "  if (gid >= count) return;\n"
  ^ "  const uint outer_index = gid / params.inner;\n"
  ^ "  const uint inner_index = gid % params.inner;\n"
  ^ "  float accumulator = 0.0f;\n"
  ^ "  for (uint axis = 0; axis < params.width; ++axis) {\n"
  ^ "    const uint offset = ((outer_index * params.width + axis)\n"
  ^ "        * params.inner) + inner_index;\n"
  ^ "    accumulator += input[offset];\n"
  ^ "  }\n"
  ^ "  output[gid] = accumulator / float(params.width);\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_argmax_f16(\n"
  ^ "    device const half* input [[buffer(0)]],\n"
  ^ "    device int* output [[buffer(1)]],\n"
  ^ "    constant ReductionParams& params [[buffer(2)]],\n"
  ^ "    uint tid [[thread_index_in_threadgroup]],\n"
  ^ "    uint group [[threadgroup_position_in_grid]]) {\n"
  ^ "  const uint count = params.outer * params.inner;\n"
  ^ "  if (group >= count) return;\n"
  ^ "  const uint outer_index = group / params.inner;\n"
  ^ "  const uint inner_index = group % params.inner;\n"
  ^ "  float best = -3.402823466e+38f;\n"
  ^ "  uint best_index = 0;\n"
  ^ "  for (uint axis = tid; axis < params.width; axis += 256) {\n"
  ^ "    const uint offset = ((outer_index * params.width + axis) * params.inner) + inner_index;\n"
  ^ "    const float candidate = float(input[offset]);\n"
  ^ "    if (candidate > best || (candidate == best && axis < best_index)) { best = candidate; best_index = axis; }\n"
  ^ "  }\n"
  ^ "  threadgroup float values[256];\n"
  ^ "  threadgroup uint indices[256];\n"
  ^ "  values[tid] = best; indices[tid] = best_index;\n"
  ^ "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  for (uint width = 128; width > 0; width >>= 1) {\n"
  ^ "    if (tid < width) {\n"
  ^ "      const float candidate = values[tid + width];\n"
  ^ "      const uint index = indices[tid + width];\n"
  ^ "      if (candidate > values[tid] || (candidate == values[tid] && index < indices[tid])) { values[tid] = candidate; indices[tid] = index; }\n"
  ^ "    }\n"
  ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  }\n"
  ^ "  if (tid == 0) output[group] = int(indices[0]);\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_argmax_f32(\n"
  ^ "    device const float* input [[buffer(0)]],\n"
  ^ "    device int* output [[buffer(1)]],\n"
  ^ "    constant ReductionParams& params [[buffer(2)]],\n"
  ^ "    uint tid [[thread_index_in_threadgroup]],\n"
  ^ "    uint group [[threadgroup_position_in_grid]]) {\n"
  ^ "  const uint count = params.outer * params.inner;\n"
  ^ "  if (group >= count) return;\n"
  ^ "  const uint outer_index = group / params.inner;\n"
  ^ "  const uint inner_index = group % params.inner;\n"
  ^ "  float best = -3.402823466e+38f;\n"
  ^ "  uint best_index = 0;\n"
  ^ "  for (uint axis = tid; axis < params.width; axis += 256) {\n"
  ^ "    const uint offset = ((outer_index * params.width + axis) * params.inner) + inner_index;\n"
  ^ "    const float candidate = input[offset];\n"
  ^ "    if (candidate > best || (candidate == best && axis < best_index)) { best = candidate; best_index = axis; }\n"
  ^ "  }\n"
  ^ "  threadgroup float values[256];\n"
  ^ "  threadgroup uint indices[256];\n"
  ^ "  values[tid] = best; indices[tid] = best_index;\n"
  ^ "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  for (uint width = 128; width > 0; width >>= 1) {\n"
  ^ "    if (tid < width) {\n"
  ^ "      const float candidate = values[tid + width];\n"
  ^ "      const uint index = indices[tid + width];\n"
  ^ "      if (candidate > values[tid] || (candidate == values[tid] && index < indices[tid])) { values[tid] = candidate; indices[tid] = index; }\n"
  ^ "    }\n"
  ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  }\n"
  ^ "  if (tid == 0) output[group] = int(indices[0]);\n"
  ^ "}\n\n"

let reduction_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_sum_f16" ~operation:Kernel_abi.Operation.Reduction
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_sum_f32" ~operation:Kernel_abi.Operation.Reduction
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_mean_f32" ~operation:Kernel_abi.Operation.Reduction
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_argmax_f16" ~operation:Kernel_abi.Operation.Reduction
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Int32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_argmax_f32" ~operation:Kernel_abi.Operation.Reduction
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Int32 ]

let update_slice_kernel_source ~name ~value_type =
  "kernel void " ^ name ^ "(\n"
  ^ "    device const " ^ value_type ^ "* destination [[buffer(0)]],\n"
  ^ "    device const " ^ value_type ^ "* source [[buffer(1)]],\n"
  ^ "    device " ^ value_type ^ "* output [[buffer(2)]],\n"
  ^ "    constant IndexParams& params [[buffer(3)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  if (gid >= params.count) return;\n"
  ^ "  output[gid] = destination[gid];\n"
  ^ "  uint destination_coordinates[8] = {};\n"
  ^ "  uint remaining = gid;\n"
  ^ "  for (int axis = int(params.input_rank) - 1; axis >= 0; --axis) {\n"
  ^ "    destination_coordinates[axis] = remaining % params.input_shape[axis];\n"
  ^ "    remaining /= params.input_shape[axis];\n"
  ^ "  }\n"
  ^ "  uint source_coordinates[8] = {};\n"
  ^ "  uint destination_axis = 0; uint source_axis = 0;\n"
  ^ "  bool selected = true;\n"
  ^ "  for (uint selector = 0; selector < params.selector_count; ++selector) {\n"
  ^ "    const uint kind = params.selector_kind[selector];\n"
  ^ "    if (kind == 0) {\n"
  ^ "      selected = selected\n"
  ^ "          && long(destination_coordinates[destination_axis]) == params.starts[selector];\n"
  ^ "      ++destination_axis;\n"
  ^ "    } else if (kind == 1) {\n"
  ^ "      const long delta = long(destination_coordinates[destination_axis])\n"
  ^ "          - params.starts[selector];\n"
  ^ "      const long step = params.steps[selector];\n"
  ^ "      if (step == 0 || delta % step != 0) {\n"
  ^ "        selected = false;\n"
  ^ "      } else {\n"
  ^ "        const long coordinate = delta / step;\n"
  ^ "        selected = selected && coordinate >= 0\n"
  ^ "            && coordinate < long(params.output_shape[source_axis]);\n"
  ^ "        source_coordinates[source_axis] = coordinate < 0 ? 0 : uint(coordinate);\n"
  ^ "      }\n"
  ^ "      ++destination_axis; ++source_axis;\n"
  ^ "    } else {\n"
  ^ "      source_coordinates[source_axis++] = 0;\n"
  ^ "    }\n"
  ^ "  }\n"
  ^ "  if (!selected) return;\n"
  ^ "  uint source_offset = 0; uint stride = 1;\n"
  ^ "  for (int axis = int(params.output_rank) - 1; axis >= 0; --axis) {\n"
  ^ "    source_offset += source_coordinates[axis] * stride;\n"
  ^ "    stride *= params.output_shape[axis];\n"
  ^ "  }\n"
  ^ "  output[gid] = source[source_offset];\n"
  ^ "}\n\n"

let update_slice_source =
  update_slice_kernel_source ~name:"llmopt_update_slice_f16" ~value_type:"half"
  ^ update_slice_kernel_source ~name:"llmopt_update_slice_f32" ~value_type:"float"

let update_slice_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_update_slice_f16"
      ~operation:Kernel_abi.Operation.Update_slice
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_update_slice_f32"
      ~operation:Kernel_abi.Operation.Update_slice
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32 ]
