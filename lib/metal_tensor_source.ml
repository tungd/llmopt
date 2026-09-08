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
