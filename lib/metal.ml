type kernel =
  | Matmul of int * int * int * Ir.Value.t * Ir.Value.t * Ir.Value.t
  | Fused of int * int * int * Ir.Value.t * Ir.Value.t * Ir.Value.t * Ir.Value.t
  | Linear of int * int * int * bool * Ir.Value.t * Ir.Value.t * Ir.Value.t option * Ir.Value.t

module Program = struct
  type t = { source : string; kernels : Kernel_abi.Entry.t list }

  let make ~source ~kernels = { source; kernels }
  let source program = program.source
  let kernels program = program.kernels
end

let kernel_entry_with_threadgroup ~threadgroup ~name ~operation ~input_dtype
    ~output_dtype =
  match
    Kernel_abi.Entry.create ~name ~operation ~input_dtype ~output_dtype
      ~threadgroup
  with
  | Ok entry -> entry
  | Error message -> invalid_arg ("invalid built-in Metal kernel: " ^ message)

let kernel_entry ~name ~operation ~input_dtype ~output_dtype =
  kernel_entry_with_threadgroup ~threadgroup:(16, 16, 1) ~name ~operation
    ~input_dtype ~output_dtype

let kernel_entry_with_threadgroup_and_tile ~tile ~threadgroup ~name ~operation
    ~input_dtype ~output_dtype =
  match
    Kernel_abi.Entry.create_with_tile ~tile ~name ~operation ~input_dtype
      ~output_dtype ~threadgroup
  with
  | Ok entry -> entry
  | Error message -> invalid_arg ("invalid parameterized Metal kernel: " ^ message)

let kernel_entry_with_tile ~tile ~name ~operation ~input_dtype ~output_dtype =
  kernel_entry_with_threadgroup_and_tile ~tile ~threadgroup:(16, 16, 1) ~name
    ~operation ~input_dtype ~output_dtype

let find_input_name graph value =
  Ir.Graph.nodes graph
  |> List.find_map (fun node ->
         match Ir.node_op node, Ir.node_output node with
         | Ir.Op.Input { name; source = _ }, Some output
           when Ir.Value.equal output value -> Some name
         | _ -> None)

let input_name graph value =
  match find_input_name graph value with
  | Some name -> name
  | None ->
      Printf.sprintf "value-%d" (Ir.Value_id.to_int (Ir.Value.id value))

let w4a16_source =
  "\nstruct W4A16Params { uint m; uint n; uint k; uint has_bias; };\n\n"
  ^ "kernel void llmopt_w4a16_linear_f16_g64(\n"
  ^ "    device const half* input [[buffer(0)]],\n"
  ^ "    device const uchar* packed_weight [[buffer(1)]],\n"
  ^ "    device const half* scale [[buffer(2)]],\n"
  ^ "    device const half* bias_or_scale [[buffer(3)]],\n"
  ^ "    device half* output [[buffer(4)]],\n"
  ^ "    constant W4A16Params& params [[buffer(5)]],\n"
  ^ "    uint gid [[thread_position_in_grid]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  const uint simd_idx = gid >> 5;\n"
  ^ "  const uint total_elements = params.m * params.n;\n"
  ^ "  if (simd_idx >= total_elements) return;\n"
  ^ "  const uint row = simd_idx / params.n;\n"
  ^ "  const uint col = simd_idx - row * params.n;\n"
  ^ "  const uint groups = params.k >> 6;\n"
  ^ "  const uint packed_base = col * (params.k >> 1);\n"
  ^ "  const uint scale_base = col * groups;\n"
  ^ "  const uint input_base = row * params.k;\n"
  ^ "  float acc = 0.0f;\n"
  ^ "  for (uint g = 0; g < groups; ++g) {\n"
  ^ "    const float s = float(scale[scale_base + g]);\n"
  ^ "    const uchar packed = packed_weight[packed_base + (g << 5) + lane];\n"
  ^ "    const int q0 = int(packed & 15u) - ((packed & 8u) ? 16 : 0);\n"
  ^ "    const int q1 = int(packed >> 4) - ((packed & 128u) ? 16 : 0);\n"
  ^ "    const uint in_idx = input_base + (g << 6) + (lane << 1);\n"
  ^ "    const float in0 = float(input[in_idx]);\n"
  ^ "    const float in1 = float(input[in_idx + 1]);\n"
  ^ "    acc += in0 * (float(q0) * s) + in1 * (float(q1) * s);\n"
  ^ "  }\n"
  ^ "  acc = simd_sum(acc);\n"
  ^ "  if (lane == 0) {\n"
  ^ "    if (params.has_bias != 0u) acc += float(bias_or_scale[col]);\n"
  ^ "    output[row * params.n + col] = half(acc);\n"
  ^ "  }\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_w4a16_linear_f16_g64_m4(\n"
  ^ "    device const half* input [[buffer(0)]],\n"
  ^ "    device const uchar* packed_weight [[buffer(1)]],\n"
  ^ "    device const half* scale [[buffer(2)]],\n"
  ^ "    device const half* bias_or_scale [[buffer(3)]],\n"
  ^ "    device half* output [[buffer(4)]],\n"
  ^ "    constant W4A16Params& params [[buffer(5)]],\n"
  ^ "    uint gid [[thread_position_in_grid]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  const uint simd_idx = gid >> 5;\n"
  ^ "  const uint m_blocks = (params.m + 3u) >> 2;\n"
  ^ "  const uint total_simd = m_blocks * params.n;\n"
  ^ "  if (simd_idx >= total_simd) return;\n"
  ^ "  const uint block_row = simd_idx / params.n;\n"
  ^ "  const uint col = simd_idx - block_row * params.n;\n"
  ^ "  const uint row0 = block_row << 2;\n"
  ^ "  const uint row1 = row0 + 1u;\n"
  ^ "  const uint row2 = row0 + 2u;\n"
  ^ "  const uint row3 = row0 + 3u;\n"
  ^ "  const uint groups = params.k >> 6;\n"
  ^ "  const uint packed_base = col * (params.k >> 1);\n"
  ^ "  const uint scale_base = col * groups;\n"
  ^ "  const uint in_base0 = row0 * params.k;\n"
  ^ "  const uint in_base1 = row1 * params.k;\n"
  ^ "  const uint in_base2 = row2 * params.k;\n"
  ^ "  const uint in_base3 = row3 * params.k;\n"
  ^ "  const bool valid1 = row1 < params.m;\n"
  ^ "  const bool valid2 = row2 < params.m;\n"
  ^ "  const bool valid3 = row3 < params.m;\n"
  ^ "  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;\n"
  ^ "  if (valid3) {\n"
  ^ "    for (uint g = 0; g < groups; ++g) {\n"
  ^ "      const float s = float(scale[scale_base + g]);\n"
  ^ "      const uchar packed = packed_weight[packed_base + (g << 5) + lane];\n"
  ^ "      const float q0 = float(int(packed & 15u) - ((packed & 8u) ? 16 : 0)) * s;\n"
  ^ "      const float q1 = float(int(packed >> 4) - ((packed & 128u) ? 16 : 0)) * s;\n"
  ^ "      const uint offset = (g << 6) + (lane << 1);\n"
  ^ "      const half2 x0 = *reinterpret_cast<device const half2*>(input + in_base0 + offset);\n"
  ^ "      const half2 x1 = *reinterpret_cast<device const half2*>(input + in_base1 + offset);\n"
  ^ "      const half2 x2 = *reinterpret_cast<device const half2*>(input + in_base2 + offset);\n"
  ^ "      const half2 x3 = *reinterpret_cast<device const half2*>(input + in_base3 + offset);\n"
  ^ "      acc0 += float(x0.x) * q0 + float(x0.y) * q1;\n"
  ^ "      acc1 += float(x1.x) * q0 + float(x1.y) * q1;\n"
  ^ "      acc2 += float(x2.x) * q0 + float(x2.y) * q1;\n"
  ^ "      acc3 += float(x3.x) * q0 + float(x3.y) * q1;\n"
  ^ "    }\n"
  ^ "  } else {\n"
  ^ "    for (uint g = 0; g < groups; ++g) {\n"
  ^ "      const float s = float(scale[scale_base + g]);\n"
  ^ "      const uchar packed = packed_weight[packed_base + (g << 5) + lane];\n"
  ^ "      const float q0 = float(int(packed & 15u) - ((packed & 8u) ? 16 : 0)) * s;\n"
  ^ "      const float q1 = float(int(packed >> 4) - ((packed & 128u) ? 16 : 0)) * s;\n"
  ^ "      const uint offset = (g << 6) + (lane << 1);\n"
  ^ "      const half2 x0 = *reinterpret_cast<device const half2*>(input + in_base0 + offset);\n"
  ^ "      acc0 += float(x0.x) * q0 + float(x0.y) * q1;\n"
  ^ "      if (valid1) { const half2 x = *reinterpret_cast<device const half2*>(input + in_base1 + offset); acc1 += float(x.x) * q0 + float(x.y) * q1; }\n"
  ^ "      if (valid2) { const half2 x = *reinterpret_cast<device const half2*>(input + in_base2 + offset); acc2 += float(x.x) * q0 + float(x.y) * q1; }\n"
  ^ "    }\n"
  ^ "  }\n"
  ^ "  acc0 = simd_sum(acc0);\n"
  ^ "  if (valid1) acc1 = simd_sum(acc1);\n"
  ^ "  if (valid2) acc2 = simd_sum(acc2);\n"
  ^ "  if (valid3) acc3 = simd_sum(acc3);\n"
  ^ "  if (lane == 0) {\n"
  ^ "    const bool has_bias = (params.has_bias != 0u);\n"
  ^ "    if (has_bias) acc0 += float(bias_or_scale[col]);\n"
  ^ "    output[row0 * params.n + col] = half(acc0);\n"
  ^ "    if (valid1) {\n"
  ^ "      if (has_bias) acc1 += float(bias_or_scale[col]);\n"
  ^ "      output[row1 * params.n + col] = half(acc1);\n"
  ^ "    }\n"
  ^ "    if (valid2) {\n"
  ^ "      if (has_bias) acc2 += float(bias_or_scale[col]);\n"
  ^ "      output[row2 * params.n + col] = half(acc2);\n"
  ^ "    }\n"
  ^ "    if (valid3) {\n"
  ^ "      if (has_bias) acc3 += float(bias_or_scale[col]);\n"
  ^ "      output[row3 * params.n + col] = half(acc3);\n"
  ^ "    }\n"
  ^ "  }\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_w4a16_linear_f16_g64_m8(\n"
  ^ "    device const half* input [[buffer(0)]],\n"
  ^ "    device const uchar* packed_weight [[buffer(1)]],\n"
  ^ "    device const half* scale [[buffer(2)]],\n"
  ^ "    device const half* bias_or_scale [[buffer(3)]],\n"
  ^ "    device half* output [[buffer(4)]],\n"
  ^ "    constant W4A16Params& params [[buffer(5)]],\n"
  ^ "    uint gid [[thread_position_in_grid]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  const uint simd_idx = gid >> 5;\n"
  ^ "  const uint m_blocks = (params.m + 7u) >> 3;\n"
  ^ "  const uint total_simd = m_blocks * params.n;\n"
  ^ "  if (simd_idx >= total_simd) return;\n"
  ^ "  const uint block_row = simd_idx / params.n;\n"
  ^ "  const uint col = simd_idx - block_row * params.n;\n"
  ^ "  const uint row0 = block_row << 3;\n"
  ^ "  const uint groups = params.k >> 6;\n"
  ^ "  const uint packed_base = col * (params.k >> 1);\n"
  ^ "  const uint scale_base = col * groups;\n"
  ^ "  const uint in_base0 = row0 * params.k;\n"
  ^ "  const uint in_base1 = (row0 + 1u) * params.k;\n"
  ^ "  const uint in_base2 = (row0 + 2u) * params.k;\n"
  ^ "  const uint in_base3 = (row0 + 3u) * params.k;\n"
  ^ "  const uint in_base4 = (row0 + 4u) * params.k;\n"
  ^ "  const uint in_base5 = (row0 + 5u) * params.k;\n"
  ^ "  const uint in_base6 = (row0 + 6u) * params.k;\n"
  ^ "  const uint in_base7 = (row0 + 7u) * params.k;\n"
  ^ "  const bool valid1 = (row0 + 1u) < params.m;\n"
  ^ "  const bool valid2 = (row0 + 2u) < params.m;\n"
  ^ "  const bool valid3 = (row0 + 3u) < params.m;\n"
  ^ "  const bool valid4 = (row0 + 4u) < params.m;\n"
  ^ "  const bool valid5 = (row0 + 5u) < params.m;\n"
  ^ "  const bool valid6 = (row0 + 6u) < params.m;\n"
  ^ "  const bool valid7 = (row0 + 7u) < params.m;\n"
  ^ "  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;\n"
  ^ "  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;\n"
  ^ "  for (uint g = 0; g < groups; ++g) {\n"
  ^ "    const float s = float(scale[scale_base + g]);\n"
  ^ "    const uchar packed = packed_weight[packed_base + (g << 5) + lane];\n"
  ^ "    const float q0 = float(int(packed & 15u) - ((packed & 8u) ? 16 : 0)) * s;\n"
  ^ "    const float q1 = float(int(packed >> 4) - ((packed & 128u) ? 16 : 0)) * s;\n"
  ^ "    const uint offset = (g << 6) + (lane << 1);\n"
  ^ "    acc0 += float(input[in_base0 + offset]) * q0 + float(input[in_base0 + offset + 1]) * q1;\n"
  ^ "    if (valid1) { acc1 += float(input[in_base1 + offset]) * q0 + float(input[in_base1 + offset + 1]) * q1; }\n"
  ^ "    if (valid2) { acc2 += float(input[in_base2 + offset]) * q0 + float(input[in_base2 + offset + 1]) * q1; }\n"
  ^ "    if (valid3) { acc3 += float(input[in_base3 + offset]) * q0 + float(input[in_base3 + offset + 1]) * q1; }\n"
  ^ "    if (valid4) { acc4 += float(input[in_base4 + offset]) * q0 + float(input[in_base4 + offset + 1]) * q1; }\n"
  ^ "    if (valid5) { acc5 += float(input[in_base5 + offset]) * q0 + float(input[in_base5 + offset + 1]) * q1; }\n"
  ^ "    if (valid6) { acc6 += float(input[in_base6 + offset]) * q0 + float(input[in_base6 + offset + 1]) * q1; }\n"
  ^ "    if (valid7) { acc7 += float(input[in_base7 + offset]) * q0 + float(input[in_base7 + offset + 1]) * q1; }\n"
  ^ "  }\n"
  ^ "  acc0 = simd_sum(acc0);\n"
  ^ "  if (valid1) acc1 = simd_sum(acc1);\n"
  ^ "  if (valid2) acc2 = simd_sum(acc2);\n"
  ^ "  if (valid3) acc3 = simd_sum(acc3);\n"
  ^ "  if (valid4) acc4 = simd_sum(acc4);\n"
  ^ "  if (valid5) acc5 = simd_sum(acc5);\n"
  ^ "  if (valid6) acc6 = simd_sum(acc6);\n"
  ^ "  if (valid7) acc7 = simd_sum(acc7);\n"
  ^ "  if (lane == 0) {\n"
  ^ "    const bool has_bias = (params.has_bias != 0u);\n"
  ^ "    if (has_bias) acc0 += float(bias_or_scale[col]);\n"
  ^ "    output[row0 * params.n + col] = half(acc0);\n"
  ^ "    if (valid1) { const uint r = row0 + 1u; if (has_bias) acc1 += float(bias_or_scale[col]); output[r * params.n + col] = half(acc1); }\n"
  ^ "    if (valid2) { const uint r = row0 + 2u; if (has_bias) acc2 += float(bias_or_scale[col]); output[r * params.n + col] = half(acc2); }\n"
  ^ "    if (valid3) { const uint r = row0 + 3u; if (has_bias) acc3 += float(bias_or_scale[col]); output[r * params.n + col] = half(acc3); }\n"
  ^ "    if (valid4) { const uint r = row0 + 4u; if (has_bias) acc4 += float(bias_or_scale[col]); output[r * params.n + col] = half(acc4); }\n"
  ^ "    if (valid5) { const uint r = row0 + 5u; if (has_bias) acc5 += float(bias_or_scale[col]); output[r * params.n + col] = half(acc5); }\n"
  ^ "    if (valid6) { const uint r = row0 + 6u; if (has_bias) acc6 += float(bias_or_scale[col]); output[r * params.n + col] = half(acc6); }\n"
  ^ "    if (valid7) { const uint r = row0 + 7u; if (has_bias) acc7 += float(bias_or_scale[col]); output[r * params.n + col] = half(acc7); }\n"
  ^ "  }\n"
  ^ "}\n"

let w4a16_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_w4a16_linear_f16_g64"
      ~operation:Kernel_abi.Operation.W4a16_linear
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_w4a16_linear_f16_g64_m4"
      ~operation:Kernel_abi.Operation.W4a16_linear
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_w4a16_linear_f16_g64_m8"
      ~operation:Kernel_abi.Operation.W4a16_linear
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16 ]

let w4a16_qkv_linear_source =
  "\nstruct W4A16QKVParams {\n"
  ^ "  uint m;\n"
  ^ "  uint k;\n"
  ^ "  uint n_q;\n"
  ^ "  uint n_k;\n"
  ^ "  uint n_v;\n"
  ^ "};\n\n"
  ^ "kernel void llmopt_w4a16_qkv_linear_f16_g64(\n"
  ^ "    device const half* input [[buffer(0)]],\n"
  ^ "    device const uchar* q_weight [[buffer(1)]],\n"
  ^ "    device const half* q_scale [[buffer(2)]],\n"
  ^ "    device const uchar* k_weight [[buffer(3)]],\n"
  ^ "    device const half* k_scale [[buffer(4)]],\n"
  ^ "    device const uchar* v_weight [[buffer(5)]],\n"
  ^ "    device const half* v_scale [[buffer(6)]],\n"
  ^ "    device half* q_output [[buffer(7)]],\n"
  ^ "    device half* k_output [[buffer(8)]],\n"
  ^ "    device half* v_output [[buffer(9)]],\n"
  ^ "    constant W4A16QKVParams& params [[buffer(10)]],\n"
  ^ "    uint gid [[thread_position_in_grid]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  const uint simd_idx = gid >> 5;\n"
  ^ "  const uint total_cols = params.n_q + params.n_k + params.n_v;\n"
  ^ "  const uint total_elements = params.m * total_cols;\n"
  ^ "  if (simd_idx >= total_elements) return;\n"
  ^ "  const uint row = simd_idx / total_cols;\n"
  ^ "  const uint col_idx = simd_idx - row * total_cols;\n"
  ^ "  device const uchar* weight_ptr;\n"
  ^ "  device const half* scale_ptr;\n"
  ^ "  device half* out_ptr;\n"
  ^ "  uint col;\n"
  ^ "  uint out_stride;\n"
  ^ "  if (col_idx < params.n_q) {\n"
  ^ "    col = col_idx;\n"
  ^ "    out_stride = params.n_q;\n"
  ^ "    weight_ptr = q_weight;\n"
  ^ "    scale_ptr = q_scale;\n"
  ^ "    out_ptr = q_output;\n"
  ^ "  } else if (col_idx < params.n_q + params.n_k) {\n"
  ^ "    col = col_idx - params.n_q;\n"
  ^ "    out_stride = params.n_k;\n"
  ^ "    weight_ptr = k_weight;\n"
  ^ "    scale_ptr = k_scale;\n"
  ^ "    out_ptr = k_output;\n"
  ^ "  } else {\n"
  ^ "    col = col_idx - (params.n_q + params.n_k);\n"
  ^ "    out_stride = params.n_v;\n"
  ^ "    weight_ptr = v_weight;\n"
  ^ "    scale_ptr = v_scale;\n"
  ^ "    out_ptr = v_output;\n"
  ^ "  }\n"
  ^ "  const uint groups = params.k >> 6;\n"
  ^ "  const uint packed_base = col * (params.k >> 1);\n"
  ^ "  const uint scale_base = col * groups;\n"
  ^ "  const uint input_base = row * params.k;\n"
  ^ "  float acc = 0.0f;\n"
  ^ "  for (uint g = 0; g < groups; ++g) {\n"
  ^ "    const float s = float(scale_ptr[scale_base + g]);\n"
  ^ "    const uchar packed = weight_ptr[packed_base + (g << 5) + lane];\n"
  ^ "    const int q0 = int(packed & 15u) - ((packed & 8u) ? 16 : 0);\n"
  ^ "    const int q1 = int(packed >> 4) - ((packed & 128u) ? 16 : 0);\n"
  ^ "    const uint in_idx = input_base + (g << 6) + (lane << 1);\n"
  ^ "    const float in0 = float(input[in_idx]);\n"
  ^ "    const float in1 = float(input[in_idx + 1]);\n"
  ^ "    acc += in0 * (float(q0) * s) + in1 * (float(q1) * s);\n"
  ^ "  }\n"
  ^ "  acc = simd_sum(acc);\n"
  ^ "  if (lane == 0) {\n"
  ^ "    out_ptr[row * out_stride + col] = half(acc);\n"
  ^ "  }\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_w4a16_qkv_linear_f16_g64_m4(\n"
  ^ "    device const half* input [[buffer(0)]],\n"
  ^ "    device const uchar* q_weight [[buffer(1)]],\n"
  ^ "    device const half* q_scale [[buffer(2)]],\n"
  ^ "    device const uchar* k_weight [[buffer(3)]],\n"
  ^ "    device const half* k_scale [[buffer(4)]],\n"
  ^ "    device const uchar* v_weight [[buffer(5)]],\n"
  ^ "    device const half* v_scale [[buffer(6)]],\n"
  ^ "    device half* q_output [[buffer(7)]],\n"
  ^ "    device half* k_output [[buffer(8)]],\n"
  ^ "    device half* v_output [[buffer(9)]],\n"
  ^ "    constant W4A16QKVParams& params [[buffer(10)]],\n"
  ^ "    uint gid [[thread_position_in_grid]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  const uint simd_idx = gid >> 5;\n"
  ^ "  const uint total_cols = params.n_q + params.n_k + params.n_v;\n"
  ^ "  const uint m_blocks = (params.m + 3u) >> 2;\n"
  ^ "  const uint total_simd = m_blocks * total_cols;\n"
  ^ "  if (simd_idx >= total_simd) return;\n"
  ^ "  const uint block_row = simd_idx / total_cols;\n"
  ^ "  const uint col_idx = simd_idx - block_row * total_cols;\n"
  ^ "  const uint row0 = block_row << 2;\n"
  ^ "  const uint row1 = row0 + 1u;\n"
  ^ "  const uint row2 = row0 + 2u;\n"
  ^ "  const uint row3 = row0 + 3u;\n"
  ^ "  const bool valid1 = row1 < params.m;\n"
  ^ "  const bool valid2 = row2 < params.m;\n"
  ^ "  const bool valid3 = row3 < params.m;\n"
  ^ "  uint col;\n"
  ^ "  device const uchar* weight_ptr;\n"
  ^ "  device const half* scale_ptr;\n"
  ^ "  device half* out_ptr;\n"
  ^ "  uint out_stride;\n"
  ^ "  if (col_idx < params.n_q) {\n"
  ^ "    col = col_idx;\n"
  ^ "    out_stride = params.n_q;\n"
  ^ "    weight_ptr = q_weight;\n"
  ^ "    scale_ptr = q_scale;\n"
  ^ "    out_ptr = q_output;\n"
  ^ "  } else if (col_idx < params.n_q + params.n_k) {\n"
  ^ "    col = col_idx - params.n_q;\n"
  ^ "    out_stride = params.n_k;\n"
  ^ "    weight_ptr = k_weight;\n"
  ^ "    scale_ptr = k_scale;\n"
  ^ "    out_ptr = k_output;\n"
  ^ "  } else {\n"
  ^ "    col = col_idx - (params.n_q + params.n_k);\n"
  ^ "    out_stride = params.n_v;\n"
  ^ "    weight_ptr = v_weight;\n"
  ^ "    scale_ptr = v_scale;\n"
  ^ "    out_ptr = v_output;\n"
  ^ "  }\n"
  ^ "  const uint groups = params.k >> 6;\n"
  ^ "  const uint packed_base = col * (params.k >> 1);\n"
  ^ "  const uint scale_base = col * groups;\n"
  ^ "  const uint in_base0 = row0 * params.k;\n"
  ^ "  const uint in_base1 = row1 * params.k;\n"
  ^ "  const uint in_base2 = row2 * params.k;\n"
  ^ "  const uint in_base3 = row3 * params.k;\n"
  ^ "  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;\n"
  ^ "  if (valid3) {\n"
  ^ "    for (uint g = 0; g < groups; ++g) {\n"
  ^ "      const float s = float(scale_ptr[scale_base + g]);\n"
  ^ "      const uchar packed = weight_ptr[packed_base + (g << 5) + lane];\n"
  ^ "      const float q0 = float(int(packed & 15u) - ((packed & 8u) ? 16 : 0)) * s;\n"
  ^ "      const float q1 = float(int(packed >> 4) - ((packed & 128u) ? 16 : 0)) * s;\n"
  ^ "      const uint offset = (g << 6) + (lane << 1);\n"
  ^ "      const half2 x0 = *reinterpret_cast<device const half2*>(input + in_base0 + offset);\n"
  ^ "      const half2 x1 = *reinterpret_cast<device const half2*>(input + in_base1 + offset);\n"
  ^ "      const half2 x2 = *reinterpret_cast<device const half2*>(input + in_base2 + offset);\n"
  ^ "      const half2 x3 = *reinterpret_cast<device const half2*>(input + in_base3 + offset);\n"
  ^ "      acc0 += float(x0.x) * q0 + float(x0.y) * q1;\n"
  ^ "      acc1 += float(x1.x) * q0 + float(x1.y) * q1;\n"
  ^ "      acc2 += float(x2.x) * q0 + float(x2.y) * q1;\n"
  ^ "      acc3 += float(x3.x) * q0 + float(x3.y) * q1;\n"
  ^ "    }\n"
  ^ "  } else {\n"
  ^ "    for (uint g = 0; g < groups; ++g) {\n"
  ^ "      const float s = float(scale_ptr[scale_base + g]);\n"
  ^ "      const uchar packed = weight_ptr[packed_base + (g << 5) + lane];\n"
  ^ "      const float q0 = float(int(packed & 15u) - ((packed & 8u) ? 16 : 0)) * s;\n"
  ^ "      const float q1 = float(int(packed >> 4) - ((packed & 128u) ? 16 : 0)) * s;\n"
  ^ "      const uint offset = (g << 6) + (lane << 1);\n"
  ^ "      const half2 x0 = *reinterpret_cast<device const half2*>(input + in_base0 + offset);\n"
  ^ "      acc0 += float(x0.x) * q0 + float(x0.y) * q1;\n"
  ^ "      if (valid1) { const half2 x = *reinterpret_cast<device const half2*>(input + in_base1 + offset); acc1 += float(x.x) * q0 + float(x.y) * q1; }\n"
  ^ "      if (valid2) { const half2 x = *reinterpret_cast<device const half2*>(input + in_base2 + offset); acc2 += float(x.x) * q0 + float(x.y) * q1; }\n"
  ^ "    }\n"
  ^ "  }\n"
  ^ "  acc0 = simd_sum(acc0);\n"
  ^ "  if (valid1) acc1 = simd_sum(acc1);\n"
  ^ "  if (valid2) acc2 = simd_sum(acc2);\n"
  ^ "  if (valid3) acc3 = simd_sum(acc3);\n"
  ^ "  if (lane == 0) {\n"
  ^ "    out_ptr[row0 * out_stride + col] = half(acc0);\n"
  ^ "    if (valid1) out_ptr[row1 * out_stride + col] = half(acc1);\n"
  ^ "    if (valid2) out_ptr[row2 * out_stride + col] = half(acc2);\n"
  ^ "    if (valid3) out_ptr[row3 * out_stride + col] = half(acc3);\n"
  ^ "  }\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_w4a16_qkv_linear_f16_g64_m8(\n"
  ^ "    device const half* input [[buffer(0)]],\n"
  ^ "    device const uchar* q_weight [[buffer(1)]],\n"
  ^ "    device const half* q_scale [[buffer(2)]],\n"
  ^ "    device const uchar* k_weight [[buffer(3)]],\n"
  ^ "    device const half* k_scale [[buffer(4)]],\n"
  ^ "    device const uchar* v_weight [[buffer(5)]],\n"
  ^ "    device const half* v_scale [[buffer(6)]],\n"
  ^ "    device half* q_output [[buffer(7)]],\n"
  ^ "    device half* k_output [[buffer(8)]],\n"
  ^ "    device half* v_output [[buffer(9)]],\n"
  ^ "    constant W4A16QKVParams& params [[buffer(10)]],\n"
  ^ "    uint gid [[thread_position_in_grid]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  const uint simd_idx = gid >> 5;\n"
  ^ "  const uint total_n = params.n_q + params.n_k + params.n_v;\n"
  ^ "  const uint m_blocks = (params.m + 7u) >> 3;\n"
  ^ "  const uint total_simd = m_blocks * total_n;\n"
  ^ "  if (simd_idx >= total_simd) return;\n"
  ^ "  const uint block_row = simd_idx / total_n;\n"
  ^ "  const uint col_idx = simd_idx - block_row * total_n;\n"
  ^ "  const uint row0 = block_row << 3;\n"
  ^ "  const bool valid1 = (row0 + 1u) < params.m;\n"
  ^ "  const bool valid2 = (row0 + 2u) < params.m;\n"
  ^ "  const bool valid3 = (row0 + 3u) < params.m;\n"
  ^ "  const bool valid4 = (row0 + 4u) < params.m;\n"
  ^ "  const bool valid5 = (row0 + 5u) < params.m;\n"
  ^ "  const bool valid6 = (row0 + 6u) < params.m;\n"
  ^ "  const bool valid7 = (row0 + 7u) < params.m;\n"
  ^ "  uint col;\n"
  ^ "  device const uchar* weight_ptr;\n"
  ^ "  device const half* scale_ptr;\n"
  ^ "  device half* out_ptr;\n"
  ^ "  uint out_stride;\n"
  ^ "  if (col_idx < params.n_q) {\n"
  ^ "    col = col_idx;\n"
  ^ "    out_stride = params.n_q;\n"
  ^ "    weight_ptr = q_weight;\n"
  ^ "    scale_ptr = q_scale;\n"
  ^ "    out_ptr = q_output;\n"
  ^ "  } else if (col_idx < params.n_q + params.n_k) {\n"
  ^ "    col = col_idx - params.n_q;\n"
  ^ "    out_stride = params.n_k;\n"
  ^ "    weight_ptr = k_weight;\n"
  ^ "    scale_ptr = k_scale;\n"
  ^ "    out_ptr = k_output;\n"
  ^ "  } else {\n"
  ^ "    col = col_idx - (params.n_q + params.n_k);\n"
  ^ "    out_stride = params.n_v;\n"
  ^ "    weight_ptr = v_weight;\n"
  ^ "    scale_ptr = v_scale;\n"
  ^ "    out_ptr = v_output;\n"
  ^ "  }\n"
  ^ "  const uint groups = params.k >> 6;\n"
  ^ "  const uint packed_base = col * (params.k >> 1);\n"
  ^ "  const uint scale_base = col * groups;\n"
  ^ "  const uint in_base0 = row0 * params.k;\n"
  ^ "  const uint in_base1 = (row0 + 1u) * params.k;\n"
  ^ "  const uint in_base2 = (row0 + 2u) * params.k;\n"
  ^ "  const uint in_base3 = (row0 + 3u) * params.k;\n"
  ^ "  const uint in_base4 = (row0 + 4u) * params.k;\n"
  ^ "  const uint in_base5 = (row0 + 5u) * params.k;\n"
  ^ "  const uint in_base6 = (row0 + 6u) * params.k;\n"
  ^ "  const uint in_base7 = (row0 + 7u) * params.k;\n"
  ^ "  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;\n"
  ^ "  float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;\n"
  ^ "  for (uint g = 0; g < groups; ++g) {\n"
  ^ "    const float s = float(scale_ptr[scale_base + g]);\n"
  ^ "    const uchar packed = weight_ptr[packed_base + (g << 5) + lane];\n"
  ^ "    const float q0 = float(int(packed & 15u) - ((packed & 8u) ? 16 : 0)) * s;\n"
  ^ "    const float q1 = float(int(packed >> 4) - ((packed & 128u) ? 16 : 0)) * s;\n"
  ^ "    const uint offset = (g << 6) + (lane << 1);\n"
  ^ "    acc0 += float(input[in_base0 + offset]) * q0 + float(input[in_base0 + offset + 1]) * q1;\n"
  ^ "    if (valid1) acc1 += float(input[in_base1 + offset]) * q0 + float(input[in_base1 + offset + 1]) * q1;\n"
  ^ "    if (valid2) acc2 += float(input[in_base2 + offset]) * q0 + float(input[in_base2 + offset + 1]) * q1;\n"
  ^ "    if (valid3) acc3 += float(input[in_base3 + offset]) * q0 + float(input[in_base3 + offset + 1]) * q1;\n"
  ^ "    if (valid4) acc4 += float(input[in_base4 + offset]) * q0 + float(input[in_base4 + offset + 1]) * q1;\n"
  ^ "    if (valid5) acc5 += float(input[in_base5 + offset]) * q0 + float(input[in_base5 + offset + 1]) * q1;\n"
  ^ "    if (valid6) acc6 += float(input[in_base6 + offset]) * q0 + float(input[in_base6 + offset + 1]) * q1;\n"
  ^ "    if (valid7) acc7 += float(input[in_base7 + offset]) * q0 + float(input[in_base7 + offset + 1]) * q1;\n"
  ^ "  }\n"
  ^ "  acc0 = simd_sum(acc0);\n"
  ^ "  if (valid1) acc1 = simd_sum(acc1);\n"
  ^ "  if (valid2) acc2 = simd_sum(acc2);\n"
  ^ "  if (valid3) acc3 = simd_sum(acc3);\n"
  ^ "  if (valid4) acc4 = simd_sum(acc4);\n"
  ^ "  if (valid5) acc5 = simd_sum(acc5);\n"
  ^ "  if (valid6) acc6 = simd_sum(acc6);\n"
  ^ "  if (valid7) acc7 = simd_sum(acc7);\n"
  ^ "  if (lane == 0) {\n"
  ^ "    out_ptr[row0 * out_stride + col] = half(acc0);\n"
  ^ "    if (valid1) out_ptr[(row0 + 1u) * out_stride + col] = half(acc1);\n"
  ^ "    if (valid2) out_ptr[(row0 + 2u) * out_stride + col] = half(acc2);\n"
  ^ "    if (valid3) out_ptr[(row0 + 3u) * out_stride + col] = half(acc3);\n"
  ^ "    if (valid4) out_ptr[(row0 + 4u) * out_stride + col] = half(acc4);\n"
  ^ "    if (valid5) out_ptr[(row0 + 5u) * out_stride + col] = half(acc5);\n"
  ^ "    if (valid6) out_ptr[(row0 + 6u) * out_stride + col] = half(acc6);\n"
  ^ "    if (valid7) out_ptr[(row0 + 7u) * out_stride + col] = half(acc7);\n"
  ^ "  }\n"
  ^ "}\n"

let w4a16_qkv_linear_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_w4a16_qkv_linear_f16_g64"
      ~operation:Kernel_abi.Operation.W4a16_qkv_linear
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_w4a16_qkv_linear_f16_g64_m4"
      ~operation:Kernel_abi.Operation.W4a16_qkv_linear
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_w4a16_qkv_linear_f16_g64_m8"
      ~operation:Kernel_abi.Operation.W4a16_qkv_linear
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16 ]

let w4a16_swiglu_rms_kernel ~name ~input_type =
  "kernel void " ^ name ^ "(\n"
  ^ "    device const " ^ input_type ^ "* input [[buffer(0)]],\n"
  ^ "    device const half* norm_weight [[buffer(1)]],\n"
  ^ "    device half* normalized [[buffer(2)]],\n"
  ^ "    constant W4A16SwiGLUFFNParams& params [[buffer(3)]],\n"
  ^ "    uint tid [[thread_index_in_threadgroup]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]],\n"
  ^ "    uint simdgroup [[simdgroup_index_in_threadgroup]],\n"
  ^ "    uint row [[threadgroup_position_in_grid]]) {\n"
  ^ "  if (row >= params.m) return;\n"
  ^ "  const uint base = row * params.k;\n"
  ^ "  float sum = 0.0f;\n"
  ^ "  for (uint i = tid; i < params.k; i += 256) {\n"
  ^ "    const float value = float(input[base + i]);\n"
  ^ "    sum += value * value;\n"
  ^ "  }\n"
  ^ "  sum = simd_sum(sum);\n"
  ^ "  threadgroup float partial[8];\n"
  ^ "  if (lane == 0) partial[simdgroup] = sum;\n"
  ^ "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  if (tid == 0) {\n"
  ^ "    float total = 0.0f;\n"
  ^ "    for (uint group = 0; group < 8; ++group) total += partial[group];\n"
  ^ "    partial[0] = rsqrt(total / float(params.k) + params.epsilon);\n"
  ^ "  }\n"
  ^ "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  const float inverse = partial[0];\n"
  ^ "  for (uint i = tid; i < params.k; i += 256)\n"
  ^ "    normalized[base + i] = half(float(input[base + i]) * float(norm_weight[i]) * inverse);\n"
  ^ "}\n"

let w4a16_swiglu_parallel_source =
  w4a16_swiglu_rms_kernel ~name:"llmopt_w4a16_swiglu_rms_f16_g64"
    ~input_type:"half"
  ^ w4a16_swiglu_rms_kernel ~name:"llmopt_w4a16_swiglu_rms_f32_g64"
      ~input_type:"float"
  ^ "kernel void llmopt_w4a16_dual_swiglu_f16_g64(\n"
  ^ "    device const half* normalized [[buffer(0)]],\n"
  ^ "    device const uchar* gate_weight [[buffer(1)]],\n"
  ^ "    device const half* gate_scale [[buffer(2)]],\n"
  ^ "    device const uchar* up_weight [[buffer(3)]],\n"
  ^ "    device const half* up_scale [[buffer(4)]],\n"
  ^ "    device half* product [[buffer(5)]],\n"
  ^ "    constant W4A16SwiGLUFFNParams& params [[buffer(6)]],\n"
  ^ "    uint gid [[thread_position_in_grid]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  const uint simd_idx = gid >> 5;\n"
  ^ "  const uint total_elements = params.m * params.n;\n"
  ^ "  if (simd_idx >= total_elements) return;\n"
  ^ "  const uint row = simd_idx / params.n;\n"
  ^ "  const uint col = simd_idx - row * params.n;\n"
  ^ "  const uint groups = params.k >> 6;\n"
  ^ "  const uint packed_base = col * (params.k >> 1);\n"
  ^ "  const uint scale_base = col * groups;\n"
  ^ "  const uint input_base = row * params.k;\n"
  ^ "  float gate_acc = 0.0f;\n"
  ^ "  float up_acc = 0.0f;\n"
  ^ "  for (uint g = 0; g < groups; ++g) {\n"
  ^ "    const float gate_s = float(gate_scale[scale_base + g]);\n"
  ^ "    const float up_s = float(up_scale[scale_base + g]);\n"
  ^ "    const uint p_offset = packed_base + (g << 5) + lane;\n"
  ^ "    const uchar g_packed = gate_weight[p_offset];\n"
  ^ "    const uchar u_packed = up_weight[p_offset];\n"
  ^ "    const int g_lo = int(g_packed & 15u) - ((g_packed & 8u) ? 16 : 0);\n"
  ^ "    const int g_hi = int(g_packed >> 4) - ((g_packed & 128u) ? 16 : 0);\n"
  ^ "    const int u_lo = int(u_packed & 15u) - ((u_packed & 8u) ? 16 : 0);\n"
  ^ "    const int u_hi = int(u_packed >> 4) - ((u_packed & 128u) ? 16 : 0);\n"
  ^ "    const uint in_idx = input_base + (g << 6) + (lane << 1);\n"
  ^ "    const float x0 = float(normalized[in_idx]);\n"
  ^ "    const float x1 = float(normalized[in_idx + 1]);\n"
  ^ "    gate_acc += x0 * (float(g_lo) * gate_s) + x1 * (float(g_hi) * gate_s);\n"
  ^ "    up_acc += x0 * (float(u_lo) * up_s) + x1 * (float(u_hi) * up_s);\n"
  ^ "  }\n"
  ^ "  gate_acc = simd_sum(gate_acc);\n"
  ^ "  up_acc = simd_sum(up_acc);\n"
  ^ "  if (lane == 0) {\n"
  ^ "    const float silu_gate = gate_acc / (1.0f + exp(-gate_acc));\n"
  ^ "    product[row * params.n + col] = half(silu_gate * up_acc);\n"
  ^ "  }\n"
  ^ "}\n"
  ^ "kernel void llmopt_w4a16_down_add_f16_g64(\n"
  ^ "    device const half* product [[buffer(0)]],\n"
  ^ "    device const uchar* down_weight [[buffer(1)]],\n"
  ^ "    device const half* down_scale [[buffer(2)]],\n"
  ^ "    device const half* residual [[buffer(3)]],\n"
  ^ "    device half* output [[buffer(4)]],\n"
  ^ "    constant W4A16SwiGLUFFNParams& params [[buffer(5)]],\n"
  ^ "    uint gid [[thread_position_in_grid]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  const uint simd_idx = gid >> 5;\n"
  ^ "  const uint total_elements = params.m * params.k;\n"
  ^ "  if (simd_idx >= total_elements) return;\n"
  ^ "  const uint row = simd_idx / params.k;\n"
  ^ "  const uint col = simd_idx - row * params.k;\n"
  ^ "  const uint groups = params.n >> 6;\n"
  ^ "  const uint packed_base = col * (params.n >> 1);\n"
  ^ "  const uint scale_base = col * groups;\n"
  ^ "  const uint input_base = row * params.n;\n"
  ^ "  float down_acc = 0.0f;\n"
  ^ "  for (uint g = 0; g < groups; ++g) {\n"
  ^ "    const float s = float(down_scale[scale_base + g]);\n"
  ^ "    const uchar packed = down_weight[packed_base + (g << 5) + lane];\n"
  ^ "    const int lo = int(packed & 15u) - ((packed & 8u) ? 16 : 0);\n"
  ^ "    const int hi = int(packed >> 4) - ((packed & 128u) ? 16 : 0);\n"
  ^ "    const uint in_idx = input_base + (g << 6) + (lane << 1);\n"
  ^ "    const float p0 = float(product[in_idx]);\n"
  ^ "    const float p1 = float(product[in_idx + 1]);\n"
  ^ "    down_acc += p0 * (float(lo) * s) + p1 * (float(hi) * s);\n"
  ^ "  }\n"
  ^ "  down_acc = simd_sum(down_acc);\n"
  ^ "  if (lane == 0) {\n"
  ^ "    const uint out_idx = row * params.k + col;\n"
  ^ "    output[out_idx] = half(down_acc + float(residual[out_idx]));\n"
  ^ "  }\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_w4a16_dual_swiglu_f16_g64_m4(\n"
  ^ "    device const half* normalized [[buffer(0)]],\n"
  ^ "    device const uchar* gate_weight [[buffer(1)]],\n"
  ^ "    device const half* gate_scale [[buffer(2)]],\n"
  ^ "    device const uchar* up_weight [[buffer(3)]],\n"
  ^ "    device const half* up_scale [[buffer(4)]],\n"
  ^ "    device half* product [[buffer(5)]],\n"
  ^ "    constant W4A16SwiGLUFFNParams& params [[buffer(6)]],\n"
  ^ "    uint gid [[thread_position_in_grid]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  const uint simd_idx = gid >> 5;\n"
  ^ "  const uint m_blocks = (params.m + 3u) >> 2;\n"
  ^ "  const uint total_simd = m_blocks * params.n;\n"
  ^ "  if (simd_idx >= total_simd) return;\n"
  ^ "  const uint block_row = simd_idx / params.n;\n"
  ^ "  const uint col = simd_idx - block_row * params.n;\n"
  ^ "  const uint row0 = block_row << 2;\n"
  ^ "  const uint row1 = row0 + 1u;\n"
  ^ "  const uint row2 = row0 + 2u;\n"
  ^ "  const uint row3 = row0 + 3u;\n"
  ^ "  const bool valid1 = row1 < params.m;\n"
  ^ "  const bool valid2 = row2 < params.m;\n"
  ^ "  const bool valid3 = row3 < params.m;\n"
  ^ "  const uint groups = params.k >> 6;\n"
  ^ "  const uint packed_base = col * (params.k >> 1);\n"
  ^ "  const uint scale_base = col * groups;\n"
  ^ "  const uint in_base0 = row0 * params.k;\n"
  ^ "  const uint in_base1 = row1 * params.k;\n"
  ^ "  const uint in_base2 = row2 * params.k;\n"
  ^ "  const uint in_base3 = row3 * params.k;\n"
  ^ "  float gate_acc0 = 0.0f, up_acc0 = 0.0f;\n"
  ^ "  float gate_acc1 = 0.0f, up_acc1 = 0.0f;\n"
  ^ "  float gate_acc2 = 0.0f, up_acc2 = 0.0f;\n"
  ^ "  float gate_acc3 = 0.0f, up_acc3 = 0.0f;\n"
  ^ "  if (valid3) {\n"
  ^ "    for (uint g = 0; g < groups; ++g) {\n"
  ^ "      const float gate_s = float(gate_scale[scale_base + g]);\n"
  ^ "      const float up_s = float(up_scale[scale_base + g]);\n"
  ^ "      const uint p_offset = packed_base + (g << 5) + lane;\n"
  ^ "      const uchar g_packed = gate_weight[p_offset];\n"
  ^ "      const uchar u_packed = up_weight[p_offset];\n"
  ^ "      const float g_lo = float(int(g_packed & 15u) - ((g_packed & 8u) ? 16 : 0)) * gate_s;\n"
  ^ "      const float g_hi = float(int(g_packed >> 4) - ((g_packed & 128u) ? 16 : 0)) * gate_s;\n"
  ^ "      const float u_lo = float(int(u_packed & 15u) - ((u_packed & 8u) ? 16 : 0)) * up_s;\n"
  ^ "      const float u_hi = float(int(u_packed >> 4) - ((u_packed & 128u) ? 16 : 0)) * up_s;\n"
  ^ "      const uint offset = (g << 6) + (lane << 1);\n"
  ^ "      const half2 x0 = *reinterpret_cast<device const half2*>(normalized + in_base0 + offset);\n"
  ^ "      const half2 x1 = *reinterpret_cast<device const half2*>(normalized + in_base1 + offset);\n"
  ^ "      const half2 x2 = *reinterpret_cast<device const half2*>(normalized + in_base2 + offset);\n"
  ^ "      const half2 x3 = *reinterpret_cast<device const half2*>(normalized + in_base3 + offset);\n"
  ^ "      gate_acc0 += float(x0.x) * g_lo + float(x0.y) * g_hi;\n"
  ^ "      up_acc0 += float(x0.x) * u_lo + float(x0.y) * u_hi;\n"
  ^ "      gate_acc1 += float(x1.x) * g_lo + float(x1.y) * g_hi;\n"
  ^ "      up_acc1 += float(x1.x) * u_lo + float(x1.y) * u_hi;\n"
  ^ "      gate_acc2 += float(x2.x) * g_lo + float(x2.y) * g_hi;\n"
  ^ "      up_acc2 += float(x2.x) * u_lo + float(x2.y) * u_hi;\n"
  ^ "      gate_acc3 += float(x3.x) * g_lo + float(x3.y) * g_hi;\n"
  ^ "      up_acc3 += float(x3.x) * u_lo + float(x3.y) * u_hi;\n"
  ^ "    }\n"
  ^ "  } else {\n"
  ^ "    for (uint g = 0; g < groups; ++g) {\n"
  ^ "      const float gate_s = float(gate_scale[scale_base + g]);\n"
  ^ "      const float up_s = float(up_scale[scale_base + g]);\n"
  ^ "      const uint p_offset = packed_base + (g << 5) + lane;\n"
  ^ "      const uchar g_packed = gate_weight[p_offset];\n"
  ^ "      const uchar u_packed = up_weight[p_offset];\n"
  ^ "      const float g_lo = float(int(g_packed & 15u) - ((g_packed & 8u) ? 16 : 0)) * gate_s;\n"
  ^ "      const float g_hi = float(int(g_packed >> 4) - ((g_packed & 128u) ? 16 : 0)) * gate_s;\n"
  ^ "      const float u_lo = float(int(u_packed & 15u) - ((u_packed & 8u) ? 16 : 0)) * up_s;\n"
  ^ "      const float u_hi = float(int(u_packed >> 4) - ((u_packed & 128u) ? 16 : 0)) * up_s;\n"
  ^ "      const uint offset = (g << 6) + (lane << 1);\n"
  ^ "      const half2 x0 = *reinterpret_cast<device const half2*>(normalized + in_base0 + offset);\n"
  ^ "      gate_acc0 += float(x0.x) * g_lo + float(x0.y) * g_hi;\n"
  ^ "      up_acc0 += float(x0.x) * u_lo + float(x0.y) * u_hi;\n"
  ^ "      if (valid1) { const half2 x = *reinterpret_cast<device const half2*>(normalized + in_base1 + offset); gate_acc1 += float(x.x) * g_lo + float(x.y) * g_hi; up_acc1 += float(x.x) * u_lo + float(x.y) * u_hi; }\n"
  ^ "      if (valid2) { const half2 x = *reinterpret_cast<device const half2*>(normalized + in_base2 + offset); gate_acc2 += float(x.x) * g_lo + float(x.y) * g_hi; up_acc2 += float(x.x) * u_lo + float(x.y) * u_hi; }\n"
  ^ "    }\n"
  ^ "  }\n"
  ^ "  gate_acc0 = simd_sum(gate_acc0);\n"
  ^ "  up_acc0 = simd_sum(up_acc0);\n"
  ^ "  if (valid1) { gate_acc1 = simd_sum(gate_acc1); up_acc1 = simd_sum(up_acc1); }\n"
  ^ "  if (valid2) { gate_acc2 = simd_sum(gate_acc2); up_acc2 = simd_sum(up_acc2); }\n"
  ^ "  if (valid3) { gate_acc3 = simd_sum(gate_acc3); up_acc3 = simd_sum(up_acc3); }\n"
  ^ "  if (lane == 0) {\n"
  ^ "    const float silu0 = gate_acc0 / (1.0f + exp(-gate_acc0));\n"
  ^ "    product[row0 * params.n + col] = half(silu0 * up_acc0);\n"
  ^ "    if (valid1) {\n"
  ^ "      const float silu1 = gate_acc1 / (1.0f + exp(-gate_acc1));\n"
  ^ "      product[row1 * params.n + col] = half(silu1 * up_acc1);\n"
  ^ "    }\n"
  ^ "    if (valid2) {\n"
  ^ "      const float silu2 = gate_acc2 / (1.0f + exp(-gate_acc2));\n"
  ^ "      product[row2 * params.n + col] = half(silu2 * up_acc2);\n"
  ^ "    }\n"
  ^ "    if (valid3) {\n"
  ^ "      const float silu3 = gate_acc3 / (1.0f + exp(-gate_acc3));\n"
  ^ "      product[row3 * params.n + col] = half(silu3 * up_acc3);\n"
  ^ "    }\n"
  ^ "  }\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_w4a16_down_add_f16_g64_m4(\n"
  ^ "    device const half* product [[buffer(0)]],\n"
  ^ "    device const uchar* down_weight [[buffer(1)]],\n"
  ^ "    device const half* down_scale [[buffer(2)]],\n"
  ^ "    device const half* residual [[buffer(3)]],\n"
  ^ "    device half* output [[buffer(4)]],\n"
  ^ "    constant W4A16SwiGLUFFNParams& params [[buffer(5)]],\n"
  ^ "    uint gid [[thread_position_in_grid]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  const uint simd_idx = gid >> 5;\n"
  ^ "  const uint m_blocks = (params.m + 3u) >> 2;\n"
  ^ "  const uint total_simd = m_blocks * params.k;\n"
  ^ "  if (simd_idx >= total_simd) return;\n"
  ^ "  const uint block_row = simd_idx / params.k;\n"
  ^ "  const uint col = simd_idx - block_row * params.k;\n"
  ^ "  const uint row0 = block_row << 2;\n"
  ^ "  const uint row1 = row0 + 1u;\n"
  ^ "  const uint row2 = row0 + 2u;\n"
  ^ "  const uint row3 = row0 + 3u;\n"
  ^ "  const bool valid1 = row1 < params.m;\n"
  ^ "  const bool valid2 = row2 < params.m;\n"
  ^ "  const bool valid3 = row3 < params.m;\n"
  ^ "  const uint groups = params.n >> 6;\n"
  ^ "  const uint packed_base = col * (params.n >> 1);\n"
  ^ "  const uint scale_base = col * groups;\n"
  ^ "  const uint in_base0 = row0 * params.n;\n"
  ^ "  const uint in_base1 = row1 * params.n;\n"
  ^ "  const uint in_base2 = row2 * params.n;\n"
  ^ "  const uint in_base3 = row3 * params.n;\n"
  ^ "  float down_acc0 = 0.0f, down_acc1 = 0.0f, down_acc2 = 0.0f, down_acc3 = 0.0f;\n"
  ^ "  if (valid3) {\n"
  ^ "    for (uint g = 0; g < groups; ++g) {\n"
  ^ "      const float s = float(down_scale[scale_base + g]);\n"
  ^ "      const uchar packed = down_weight[packed_base + (g << 5) + lane];\n"
  ^ "      const float lo = float(int(packed & 15u) - ((packed & 8u) ? 16 : 0)) * s;\n"
  ^ "      const float hi = float(int(packed >> 4) - ((packed & 128u) ? 16 : 0)) * s;\n"
  ^ "      const uint offset = (g << 6) + (lane << 1);\n"
  ^ "      const half2 p0 = *reinterpret_cast<device const half2*>(product + in_base0 + offset);\n"
  ^ "      const half2 p1 = *reinterpret_cast<device const half2*>(product + in_base1 + offset);\n"
  ^ "      const half2 p2 = *reinterpret_cast<device const half2*>(product + in_base2 + offset);\n"
  ^ "      const half2 p3 = *reinterpret_cast<device const half2*>(product + in_base3 + offset);\n"
  ^ "      down_acc0 += float(p0.x) * lo + float(p0.y) * hi;\n"
  ^ "      down_acc1 += float(p1.x) * lo + float(p1.y) * hi;\n"
  ^ "      down_acc2 += float(p2.x) * lo + float(p2.y) * hi;\n"
  ^ "      down_acc3 += float(p3.x) * lo + float(p3.y) * hi;\n"
  ^ "    }\n"
  ^ "  } else {\n"
  ^ "    for (uint g = 0; g < groups; ++g) {\n"
  ^ "      const float s = float(down_scale[scale_base + g]);\n"
  ^ "      const uchar packed = down_weight[packed_base + (g << 5) + lane];\n"
  ^ "      const float lo = float(int(packed & 15u) - ((packed & 8u) ? 16 : 0)) * s;\n"
  ^ "      const float hi = float(int(packed >> 4) - ((packed & 128u) ? 16 : 0)) * s;\n"
  ^ "      const uint offset = (g << 6) + (lane << 1);\n"
  ^ "      const half2 p0 = *reinterpret_cast<device const half2*>(product + in_base0 + offset);\n"
  ^ "      down_acc0 += float(p0.x) * lo + float(p0.y) * hi;\n"
  ^ "      if (valid1) { const half2 p = *reinterpret_cast<device const half2*>(product + in_base1 + offset); down_acc1 += float(p.x) * lo + float(p.y) * hi; }\n"
  ^ "      if (valid2) { const half2 p = *reinterpret_cast<device const half2*>(product + in_base2 + offset); down_acc2 += float(p.x) * lo + float(p.y) * hi; }\n"
  ^ "    }\n"
  ^ "  }\n"
  ^ "  down_acc0 = simd_sum(down_acc0);\n"
  ^ "  if (valid1) down_acc1 = simd_sum(down_acc1);\n"
  ^ "  if (valid2) down_acc2 = simd_sum(down_acc2);\n"
  ^ "  if (valid3) down_acc3 = simd_sum(down_acc3);\n"
  ^ "  if (lane == 0) {\n"
  ^ "    const uint out_idx0 = row0 * params.k + col;\n"
  ^ "    output[out_idx0] = half(down_acc0 + float(residual[out_idx0]));\n"
  ^ "    if (valid1) {\n"
  ^ "      const uint out_idx1 = row1 * params.k + col;\n"
  ^ "      output[out_idx1] = half(down_acc1 + float(residual[out_idx1]));\n"
  ^ "    }\n"
  ^ "    if (valid2) {\n"
  ^ "      const uint out_idx2 = row2 * params.k + col;\n"
  ^ "      output[out_idx2] = half(down_acc2 + float(residual[out_idx2]));\n"
  ^ "    }\n"
  ^ "    if (valid3) {\n"
  ^ "      const uint out_idx3 = row3 * params.k + col;\n"
  ^ "      output[out_idx3] = half(down_acc3 + float(residual[out_idx3]));\n"
  ^ "    }\n"
  ^ "  }\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_w4a16_dual_swiglu_f16_g64_m8(\n"
  ^ "    device const half* normalized [[buffer(0)]],\n"
  ^ "    device const uchar* gate_weight [[buffer(1)]],\n"
  ^ "    device const half* gate_scale [[buffer(2)]],\n"
  ^ "    device const uchar* up_weight [[buffer(3)]],\n"
  ^ "    device const half* up_scale [[buffer(4)]],\n"
  ^ "    device half* product [[buffer(5)]],\n"
  ^ "    constant W4A16SwiGLUFFNParams& params [[buffer(6)]],\n"
  ^ "    uint gid [[thread_position_in_grid]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  const uint simd_idx = gid >> 5;\n"
  ^ "  const uint m_blocks = (params.m + 7u) >> 3;\n"
  ^ "  const uint total_simd = m_blocks * params.n;\n"
  ^ "  if (simd_idx >= total_simd) return;\n"
  ^ "  const uint block_row = simd_idx / params.n;\n"
  ^ "  const uint col = simd_idx - block_row * params.n;\n"
  ^ "  const uint row0 = block_row << 3;\n"
  ^ "  const bool valid1 = (row0 + 1u) < params.m;\n"
  ^ "  const bool valid2 = (row0 + 2u) < params.m;\n"
  ^ "  const bool valid3 = (row0 + 3u) < params.m;\n"
  ^ "  const bool valid4 = (row0 + 4u) < params.m;\n"
  ^ "  const bool valid5 = (row0 + 5u) < params.m;\n"
  ^ "  const bool valid6 = (row0 + 6u) < params.m;\n"
  ^ "  const bool valid7 = (row0 + 7u) < params.m;\n"
  ^ "  const uint groups = params.k >> 6;\n"
  ^ "  const uint packed_base = col * (params.k >> 1);\n"
  ^ "  const uint scale_base = col * groups;\n"
  ^ "  const uint in_base0 = row0 * params.k;\n"
  ^ "  const uint in_base1 = (row0 + 1u) * params.k;\n"
  ^ "  const uint in_base2 = (row0 + 2u) * params.k;\n"
  ^ "  const uint in_base3 = (row0 + 3u) * params.k;\n"
  ^ "  const uint in_base4 = (row0 + 4u) * params.k;\n"
  ^ "  const uint in_base5 = (row0 + 5u) * params.k;\n"
  ^ "  const uint in_base6 = (row0 + 6u) * params.k;\n"
  ^ "  const uint in_base7 = (row0 + 7u) * params.k;\n"
  ^ "  float gate_acc0 = 0.0f, up_acc0 = 0.0f;\n"
  ^ "  float gate_acc1 = 0.0f, up_acc1 = 0.0f;\n"
  ^ "  float gate_acc2 = 0.0f, up_acc2 = 0.0f;\n"
  ^ "  float gate_acc3 = 0.0f, up_acc3 = 0.0f;\n"
  ^ "  float gate_acc4 = 0.0f, up_acc4 = 0.0f;\n"
  ^ "  float gate_acc5 = 0.0f, up_acc5 = 0.0f;\n"
  ^ "  float gate_acc6 = 0.0f, up_acc6 = 0.0f;\n"
  ^ "  float gate_acc7 = 0.0f, up_acc7 = 0.0f;\n"
  ^ "  for (uint g = 0; g < groups; ++g) {\n"
  ^ "    const float gate_s = float(gate_scale[scale_base + g]);\n"
  ^ "    const float up_s = float(up_scale[scale_base + g]);\n"
  ^ "    const uint p_offset = packed_base + (g << 5) + lane;\n"
  ^ "    const uchar g_packed = gate_weight[p_offset];\n"
  ^ "    const uchar u_packed = up_weight[p_offset];\n"
  ^ "    const float g_lo = float(int(g_packed & 15u) - ((g_packed & 8u) ? 16 : 0)) * gate_s;\n"
  ^ "    const float g_hi = float(int(g_packed >> 4) - ((g_packed & 128u) ? 16 : 0)) * gate_s;\n"
  ^ "    const float u_lo = float(int(u_packed & 15u) - ((u_packed & 8u) ? 16 : 0)) * up_s;\n"
  ^ "    const float u_hi = float(int(u_packed >> 4) - ((u_packed & 128u) ? 16 : 0)) * up_s;\n"
  ^ "    const uint offset = (g << 6) + (lane << 1);\n"
  ^ "    gate_acc0 += float(normalized[in_base0 + offset]) * g_lo + float(normalized[in_base0 + offset + 1]) * g_hi;\n"
  ^ "    up_acc0 += float(normalized[in_base0 + offset]) * u_lo + float(normalized[in_base0 + offset + 1]) * u_hi;\n"
  ^ "    if (valid1) {\n"
  ^ "      gate_acc1 += float(normalized[in_base1 + offset]) * g_lo + float(normalized[in_base1 + offset + 1]) * g_hi;\n"
  ^ "      up_acc1 += float(normalized[in_base1 + offset]) * u_lo + float(normalized[in_base1 + offset + 1]) * u_hi;\n"
  ^ "    }\n"
  ^ "    if (valid2) {\n"
  ^ "      gate_acc2 += float(normalized[in_base2 + offset]) * g_lo + float(normalized[in_base2 + offset + 1]) * g_hi;\n"
  ^ "      up_acc2 += float(normalized[in_base2 + offset]) * u_lo + float(normalized[in_base2 + offset + 1]) * u_hi;\n"
  ^ "    }\n"
  ^ "    if (valid3) {\n"
  ^ "      gate_acc3 += float(normalized[in_base3 + offset]) * g_lo + float(normalized[in_base3 + offset + 1]) * g_hi;\n"
  ^ "      up_acc3 += float(normalized[in_base3 + offset]) * u_lo + float(normalized[in_base3 + offset + 1]) * u_hi;\n"
  ^ "    }\n"
  ^ "    if (valid4) {\n"
  ^ "      gate_acc4 += float(normalized[in_base4 + offset]) * g_lo + float(normalized[in_base4 + offset + 1]) * g_hi;\n"
  ^ "      up_acc4 += float(normalized[in_base4 + offset]) * u_lo + float(normalized[in_base4 + offset + 1]) * u_hi;\n"
  ^ "    }\n"
  ^ "    if (valid5) {\n"
  ^ "      gate_acc5 += float(normalized[in_base5 + offset]) * g_lo + float(normalized[in_base5 + offset + 1]) * g_hi;\n"
  ^ "      up_acc5 += float(normalized[in_base5 + offset]) * u_lo + float(normalized[in_base5 + offset + 1]) * u_hi;\n"
  ^ "    }\n"
  ^ "    if (valid6) {\n"
  ^ "      gate_acc6 += float(normalized[in_base6 + offset]) * g_lo + float(normalized[in_base6 + offset + 1]) * g_hi;\n"
  ^ "      up_acc6 += float(normalized[in_base6 + offset]) * u_lo + float(normalized[in_base6 + offset + 1]) * u_hi;\n"
  ^ "    }\n"
  ^ "    if (valid7) {\n"
  ^ "      gate_acc7 += float(normalized[in_base7 + offset]) * g_lo + float(normalized[in_base7 + offset + 1]) * g_hi;\n"
  ^ "      up_acc7 += float(normalized[in_base7 + offset]) * u_lo + float(normalized[in_base7 + offset + 1]) * u_hi;\n"
  ^ "    }\n"
  ^ "  }\n"
  ^ "  gate_acc0 = simd_sum(gate_acc0); up_acc0 = simd_sum(up_acc0);\n"
  ^ "  if (valid1) { gate_acc1 = simd_sum(gate_acc1); up_acc1 = simd_sum(up_acc1); }\n"
  ^ "  if (valid2) { gate_acc2 = simd_sum(gate_acc2); up_acc2 = simd_sum(up_acc2); }\n"
  ^ "  if (valid3) { gate_acc3 = simd_sum(gate_acc3); up_acc3 = simd_sum(up_acc3); }\n"
  ^ "  if (valid4) { gate_acc4 = simd_sum(gate_acc4); up_acc4 = simd_sum(up_acc4); }\n"
  ^ "  if (valid5) { gate_acc5 = simd_sum(gate_acc5); up_acc5 = simd_sum(up_acc5); }\n"
  ^ "  if (valid6) { gate_acc6 = simd_sum(gate_acc6); up_acc6 = simd_sum(up_acc6); }\n"
  ^ "  if (valid7) { gate_acc7 = simd_sum(gate_acc7); up_acc7 = simd_sum(up_acc7); }\n"
  ^ "  if (lane == 0) {\n"
  ^ "    const float silu0 = gate_acc0 / (1.0f + exp(-gate_acc0));\n"
  ^ "    product[row0 * params.n + col] = half(silu0 * up_acc0);\n"
  ^ "    if (valid1) { const float s = gate_acc1 / (1.0f + exp(-gate_acc1)); product[(row0 + 1u) * params.n + col] = half(s * up_acc1); }\n"
  ^ "    if (valid2) { const float s = gate_acc2 / (1.0f + exp(-gate_acc2)); product[(row0 + 2u) * params.n + col] = half(s * up_acc2); }\n"
  ^ "    if (valid3) { const float s = gate_acc3 / (1.0f + exp(-gate_acc3)); product[(row0 + 3u) * params.n + col] = half(s * up_acc3); }\n"
  ^ "    if (valid4) { const float s = gate_acc4 / (1.0f + exp(-gate_acc4)); product[(row0 + 4u) * params.n + col] = half(s * up_acc4); }\n"
  ^ "    if (valid5) { const float s = gate_acc5 / (1.0f + exp(-gate_acc5)); product[(row0 + 5u) * params.n + col] = half(s * up_acc5); }\n"
  ^ "    if (valid6) { const float s = gate_acc6 / (1.0f + exp(-gate_acc6)); product[(row0 + 6u) * params.n + col] = half(s * up_acc6); }\n"
  ^ "    if (valid7) { const float s = gate_acc7 / (1.0f + exp(-gate_acc7)); product[(row0 + 7u) * params.n + col] = half(s * up_acc7); }\n"
  ^ "  }\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_w4a16_down_add_f16_g64_m8(\n"
  ^ "    device const half* product [[buffer(0)]],\n"
  ^ "    device const uchar* down_weight [[buffer(1)]],\n"
  ^ "    device const half* down_scale [[buffer(2)]],\n"
  ^ "    device const half* residual [[buffer(3)]],\n"
  ^ "    device half* output [[buffer(4)]],\n"
  ^ "    constant W4A16SwiGLUFFNParams& params [[buffer(5)]],\n"
  ^ "    uint gid [[thread_position_in_grid]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  const uint simd_idx = gid >> 5;\n"
  ^ "  const uint m_blocks = (params.m + 7u) >> 3;\n"
  ^ "  const uint total_simd = m_blocks * params.k;\n"
  ^ "  if (simd_idx >= total_simd) return;\n"
  ^ "  const uint block_row = simd_idx / params.k;\n"
  ^ "  const uint col = simd_idx - block_row * params.k;\n"
  ^ "  const uint row0 = block_row << 3;\n"
  ^ "  const bool valid1 = (row0 + 1u) < params.m;\n"
  ^ "  const bool valid2 = (row0 + 2u) < params.m;\n"
  ^ "  const bool valid3 = (row0 + 3u) < params.m;\n"
  ^ "  const bool valid4 = (row0 + 4u) < params.m;\n"
  ^ "  const bool valid5 = (row0 + 5u) < params.m;\n"
  ^ "  const bool valid6 = (row0 + 6u) < params.m;\n"
  ^ "  const bool valid7 = (row0 + 7u) < params.m;\n"
  ^ "  const uint groups = params.n >> 6;\n"
  ^ "  const uint packed_base = col * (params.n >> 1);\n"
  ^ "  const uint scale_base = col * groups;\n"
  ^ "  const uint in_base0 = row0 * params.n;\n"
  ^ "  const uint in_base1 = (row0 + 1u) * params.n;\n"
  ^ "  const uint in_base2 = (row0 + 2u) * params.n;\n"
  ^ "  const uint in_base3 = (row0 + 3u) * params.n;\n"
  ^ "  const uint in_base4 = (row0 + 4u) * params.n;\n"
  ^ "  const uint in_base5 = (row0 + 5u) * params.n;\n"
  ^ "  const uint in_base6 = (row0 + 6u) * params.n;\n"
  ^ "  const uint in_base7 = (row0 + 7u) * params.n;\n"
  ^ "  float down_acc0 = 0.0f, down_acc1 = 0.0f, down_acc2 = 0.0f, down_acc3 = 0.0f;\n"
  ^ "  float down_acc4 = 0.0f, down_acc5 = 0.0f, down_acc6 = 0.0f, down_acc7 = 0.0f;\n"
  ^ "  for (uint g = 0; g < groups; ++g) {\n"
  ^ "    const float s = float(down_scale[scale_base + g]);\n"
  ^ "    const uchar packed = down_weight[packed_base + (g << 5) + lane];\n"
  ^ "    const float lo = float(int(packed & 15u) - ((packed & 8u) ? 16 : 0)) * s;\n"
  ^ "    const float hi = float(int(packed >> 4) - ((packed & 128u) ? 16 : 0)) * s;\n"
  ^ "    const uint offset = (g << 6) + (lane << 1);\n"
  ^ "    down_acc0 += float(product[in_base0 + offset]) * lo + float(product[in_base0 + offset + 1]) * hi;\n"
  ^ "    if (valid1) down_acc1 += float(product[in_base1 + offset]) * lo + float(product[in_base1 + offset + 1]) * hi;\n"
  ^ "    if (valid2) down_acc2 += float(product[in_base2 + offset]) * lo + float(product[in_base2 + offset + 1]) * hi;\n"
  ^ "    if (valid3) down_acc3 += float(product[in_base3 + offset]) * lo + float(product[in_base3 + offset + 1]) * hi;\n"
  ^ "    if (valid4) down_acc4 += float(product[in_base4 + offset]) * lo + float(product[in_base4 + offset + 1]) * hi;\n"
  ^ "    if (valid5) down_acc5 += float(product[in_base5 + offset]) * lo + float(product[in_base5 + offset + 1]) * hi;\n"
  ^ "    if (valid6) down_acc6 += float(product[in_base6 + offset]) * lo + float(product[in_base6 + offset + 1]) * hi;\n"
  ^ "    if (valid7) down_acc7 += float(product[in_base7 + offset]) * lo + float(product[in_base7 + offset + 1]) * hi;\n"
  ^ "  }\n"
  ^ "  down_acc0 = simd_sum(down_acc0);\n"
  ^ "  if (valid1) down_acc1 = simd_sum(down_acc1);\n"
  ^ "  if (valid2) down_acc2 = simd_sum(down_acc2);\n"
  ^ "  if (valid3) down_acc3 = simd_sum(down_acc3);\n"
  ^ "  if (valid4) down_acc4 = simd_sum(down_acc4);\n"
  ^ "  if (valid5) down_acc5 = simd_sum(down_acc5);\n"
  ^ "  if (valid6) down_acc6 = simd_sum(down_acc6);\n"
  ^ "  if (valid7) down_acc7 = simd_sum(down_acc7);\n"
  ^ "  if (lane == 0) {\n"
  ^ "    output[row0 * params.k + col] = half(down_acc0 + float(residual[row0 * params.k + col]));\n"
  ^ "    if (valid1) { const uint r = row0 + 1u; output[r * params.k + col] = half(down_acc1 + float(residual[r * params.k + col])); }\n"
  ^ "    if (valid2) { const uint r = row0 + 2u; output[r * params.k + col] = half(down_acc2 + float(residual[r * params.k + col])); }\n"
  ^ "    if (valid3) { const uint r = row0 + 3u; output[r * params.k + col] = half(down_acc3 + float(residual[r * params.k + col])); }\n"
  ^ "    if (valid4) { const uint r = row0 + 4u; output[r * params.k + col] = half(down_acc4 + float(residual[r * params.k + col])); }\n"
  ^ "    if (valid5) { const uint r = row0 + 5u; output[r * params.k + col] = half(down_acc5 + float(residual[r * params.k + col])); }\n"
  ^ "    if (valid6) { const uint r = row0 + 6u; output[r * params.k + col] = half(down_acc6 + float(residual[r * params.k + col])); }\n"
  ^ "    if (valid7) { const uint r = row0 + 7u; output[r * params.k + col] = half(down_acc7 + float(residual[r * params.k + col])); }\n"
  ^ "  }\n"
  ^ "}\n"

let w4a16_swiglu_ffn_source =
  "\nstruct W4A16SwiGLUFFNParams { uint m; uint n; uint k; float epsilon; };\n\n"
  ^ w4a16_swiglu_parallel_source

let w4a16_swiglu_ffn_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_w4a16_swiglu_rms_f16_g64"
      ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_w4a16_swiglu_rms_f32_g64"
      ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_w4a16_dual_swiglu_f16_g64"
      ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_w4a16_dual_swiglu_f16_g64_m4"
      ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_w4a16_dual_swiglu_f16_g64_m8"
      ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_w4a16_down_add_f16_g64"
      ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_w4a16_down_add_f16_g64_m4"
      ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_w4a16_down_add_f16_g64_m8"
      ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16 ]

let w4a16_lm_head_argmax_kernel ~name ~value_type ~extra_output =
  "kernel void " ^ name ^ "(\n"
  ^ "    device const " ^ value_type ^ "* input [[buffer(0)]],\n"
  ^ "    device const half* norm_weight [[buffer(1)]],\n"
  ^ "    device const uchar* weight [[buffer(2)]],\n"
  ^ "    device const half* scale [[buffer(3)]],\n"
  ^ "    device uint* token_ids [[buffer(4)]],\n"
  ^ (if extra_output then
       "    device half* logits [[buffer(5)]],\n"
       ^ "    constant W4A16LmHeadParams& params [[buffer(6)]],\n"
     else "    constant W4A16LmHeadParams& params [[buffer(5)]],\n")
  ^ "    uint tid [[thread_index_in_threadgroup]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]],\n"
  ^ "    uint simdgroup [[simdgroup_index_in_threadgroup]],\n"
  ^ "    uint row [[threadgroup_position_in_grid]]) {\n"
  ^ "  if (row >= params.m) return;\n"
  ^ "  float square_sum = 0.0f;\n"
  ^ "  for (uint inner = tid; inner < params.k; inner += 256)\n"
  ^ "    square_sum += float(input[row * params.k + inner]) *\n"
  ^ "        float(input[row * params.k + inner]);\n"
  ^ "  square_sum = simd_sum(square_sum);\n"
  ^ "  threadgroup float partial_sums[8];\n"
  ^ "  threadgroup float inverse;\n"
  ^ "  if (lane == 0) partial_sums[simdgroup] = square_sum;\n"
  ^ "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  if (tid == 0) {\n"
  ^ "    float total = 0.0f;\n"
  ^ "    for (uint group = 0; group < 8; ++group) total += partial_sums[group];\n"
  ^ "    inverse = rsqrt(total / float(params.k) + params.epsilon);\n"
  ^ "  }\n"
  ^ "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  threadgroup float cached_norm[2048];\n"
  ^ "  const bool can_cache = params.k <= 2048;\n"
  ^ "  if (can_cache) {\n"
  ^ "    for (uint inner = tid; inner < params.k; inner += 256) {\n"
  ^ "      cached_norm[inner] = float(input[row * params.k + inner]) *\n"
  ^ "          float(norm_weight[inner]) * inverse;\n"
  ^ "    }\n"
  ^ "  }\n"
  ^ "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  threadgroup float best_values[256];\n"
  ^ "  threadgroup uint best_indices[256];\n"
  ^ "  float best_value = -3.402823466e+38f;\n"
  ^ "  uint best_index = 0;\n"
  ^ "  for (uint col = tid; col < params.n; col += 256) {\n"
  ^ "    const uint weight_base = col * (params.k / 2);\n"
  ^ "    const uint scale_base = col * (params.k / 64);\n"
  ^ "    float accumulator = 0.0f;\n"
  ^ "    if (can_cache) {\n"
  ^ "      const uint groups = params.k >> 6;\n"
  ^ "      for (uint g = 0; g < groups; ++g) {\n"
  ^ "        const float s = float(scale[scale_base + g]);\n"
  ^ "        float group_acc = 0.0f;\n"
  ^ "        const uint g_packed_base = weight_base + (g << 5);\n"
  ^ "        const uint g_norm_base = g << 6;\n"
  ^ "        for (uint pair = 0; pair < 32; pair += 4) {\n"
  ^ "          device const uchar4* wp = reinterpret_cast<device const uchar4*>(weight + g_packed_base + pair);\n"
  ^ "          const uchar4 p4 = *wp;\n"
  ^ "          const int q0 = int(p4.x & 15u) - ((p4.x & 8u) ? 16 : 0);\n"
  ^ "          const int q1 = int(p4.x >> 4) - ((p4.x & 128u) ? 16 : 0);\n"
  ^ "          const int q2 = int(p4.y & 15u) - ((p4.y & 8u) ? 16 : 0);\n"
  ^ "          const int q3 = int(p4.y >> 4) - ((p4.y & 128u) ? 16 : 0);\n"
  ^ "          const int q4 = int(p4.z & 15u) - ((p4.z & 8u) ? 16 : 0);\n"
  ^ "          const int q5 = int(p4.z >> 4) - ((p4.z & 128u) ? 16 : 0);\n"
  ^ "          const int q6 = int(p4.w & 15u) - ((p4.w & 8u) ? 16 : 0);\n"
  ^ "          const int q7 = int(p4.w >> 4) - ((p4.w & 128u) ? 16 : 0);\n"
  ^ "          const uint ni = g_norm_base + (pair << 1);\n"
  ^ "          group_acc += cached_norm[ni + 0] * float(q0)\n"
  ^ "                     + cached_norm[ni + 1] * float(q1)\n"
  ^ "                     + cached_norm[ni + 2] * float(q2)\n"
  ^ "                     + cached_norm[ni + 3] * float(q3)\n"
  ^ "                     + cached_norm[ni + 4] * float(q4)\n"
  ^ "                     + cached_norm[ni + 5] * float(q5)\n"
  ^ "                     + cached_norm[ni + 6] * float(q6)\n"
  ^ "                     + cached_norm[ni + 7] * float(q7);\n"
  ^ "        }\n"
  ^ "        accumulator += group_acc * s;\n"
  ^ "      }\n"
  ^ "    } else {\n"
  ^ "      const uint groups = params.k >> 6;\n"
  ^ "      for (uint g = 0; g < groups; ++g) {\n"
  ^ "        const float s = float(scale[scale_base + g]);\n"
  ^ "        float group_acc = 0.0f;\n"
  ^ "        const uint g_packed_base = weight_base + (g << 5);\n"
  ^ "        const uint g_in_base = row * params.k + (g << 6);\n"
  ^ "        for (uint pair = 0; pair < 32; pair += 4) {\n"
  ^ "          device const uchar4* wp = reinterpret_cast<device const uchar4*>(weight + g_packed_base + pair);\n"
  ^ "          const uchar4 p4 = *wp;\n"
  ^ "          const int q0 = int(p4.x & 15u) - ((p4.x & 8u) ? 16 : 0);\n"
  ^ "          const int q1 = int(p4.x >> 4) - ((p4.x & 128u) ? 16 : 0);\n"
  ^ "          const int q2 = int(p4.y & 15u) - ((p4.y & 8u) ? 16 : 0);\n"
  ^ "          const int q3 = int(p4.y >> 4) - ((p4.y & 128u) ? 16 : 0);\n"
  ^ "          const int q4 = int(p4.z & 15u) - ((p4.z & 8u) ? 16 : 0);\n"
  ^ "          const int q5 = int(p4.z >> 4) - ((p4.z & 128u) ? 16 : 0);\n"
  ^ "          const int q6 = int(p4.w & 15u) - ((p4.w & 8u) ? 16 : 0);\n"
  ^ "          const int q7 = int(p4.w >> 4) - ((p4.w & 128u) ? 16 : 0);\n"
  ^ "          const uint ni = g_in_base + (pair << 1);\n"
  ^ "          const uint wi = (g << 6) + (pair << 1);\n"
  ^ "          group_acc += (float(input[ni + 0]) * float(norm_weight[wi + 0]) * inverse) * float(q0)\n"
  ^ "                     + (float(input[ni + 1]) * float(norm_weight[wi + 1]) * inverse) * float(q1)\n"
  ^ "                     + (float(input[ni + 2]) * float(norm_weight[wi + 2]) * inverse) * float(q2)\n"
  ^ "                     + (float(input[ni + 3]) * float(norm_weight[wi + 3]) * inverse) * float(q3)\n"
  ^ "                     + (float(input[ni + 4]) * float(norm_weight[wi + 4]) * inverse) * float(q4)\n"
  ^ "                     + (float(input[ni + 5]) * float(norm_weight[wi + 5]) * inverse) * float(q5)\n"
  ^ "                     + (float(input[ni + 6]) * float(norm_weight[wi + 6]) * inverse) * float(q6)\n"
  ^ "                     + (float(input[ni + 7]) * float(norm_weight[wi + 7]) * inverse) * float(q7);\n"
  ^ "        }\n"
  ^ "        accumulator += group_acc * s;\n"
  ^ "      }\n"
  ^ "    }\n"
  ^ "    if (accumulator > best_value ||\n"
  ^ "        (accumulator == best_value && col < best_index)) {\n"
  ^ "      best_value = accumulator;\n"
  ^ "      best_index = col;\n"
  ^ "    }\n"
  ^ (if extra_output then
       "    logits[row * params.n + col] = half(accumulator);\n"
     else "")
  ^ "  }\n"
  ^ "  best_values[tid] = best_value;\n"
  ^ "  best_indices[tid] = best_index;\n"
  ^ "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  for (uint width = 128; width > 0; width >>= 1) {\n"
  ^ "    if (tid < width) {\n"
  ^ "      float candidate = best_values[tid + width];\n"
  ^ "      uint candidate_index = best_indices[tid + width];\n"
  ^ "      if (candidate > best_values[tid] ||\n"
  ^ "          (candidate == best_values[tid] && candidate_index < best_indices[tid])) {\n"
  ^ "        best_values[tid] = candidate;\n"
  ^ "        best_indices[tid] = candidate_index;\n"
  ^ "      }\n"
  ^ "    }\n"
  ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  }\n"
  ^ "  if (tid == 0) token_ids[row] = best_indices[0];\n"
  ^ "}\n"

let w4a16_lm_head_argmax_stage1_kernel ~name ~value_type ~extra_output =
  "kernel void " ^ name ^ "(\n"
  ^ "    device const " ^ value_type ^ "* input [[buffer(0)]],\n"
  ^ "    device const half* norm_weight [[buffer(1)]],\n"
  ^ "    device const uchar* weight [[buffer(2)]],\n"
  ^ "    device const half* scale [[buffer(3)]],\n"
  ^ "    device W4A16LmHeadCandidate* candidates [[buffer(4)]],\n"
  ^ (if extra_output then
       "    device half* logits [[buffer(5)]],\n"
       ^ "    constant W4A16LmHeadParams& params [[buffer(6)]],\n"
     else "    constant W4A16LmHeadParams& params [[buffer(5)]],\n")
  ^ "    uint tid [[thread_index_in_threadgroup]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]],\n"
  ^ "    uint simdgroup [[simdgroup_index_in_threadgroup]],\n"
  ^ "    uint3 group_position [[threadgroup_position_in_grid]]) {\n"
  ^ "  const uint row = group_position.y;\n"
  ^ "  if (row >= params.m) return;\n"
  ^ "  const uint group_idx = group_position.x;\n"
  ^ "  if (group_idx >= 256) return;\n\n"
  ^ "  float square_sum = 0.0f;\n"
  ^ "  for (uint inner = tid; inner < params.k; inner += 256)\n"
  ^ "    square_sum += float(input[row * params.k + inner]) *\n"
  ^ "        float(input[row * params.k + inner]);\n"
  ^ "  square_sum = simd_sum(square_sum);\n"
  ^ "  threadgroup float partial_sums[8];\n"
  ^ "  if (lane == 0) partial_sums[simdgroup] = square_sum;\n"
  ^ "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  if (tid == 0) {\n"
  ^ "    float total = 0.0f;\n"
  ^ "    for (uint group = 0; group < 8; ++group) total += partial_sums[group];\n"
  ^ "    partial_sums[0] = rsqrt(total / float(params.k) + params.epsilon);\n"
  ^ "  }\n"
  ^ "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  const float inverse = partial_sums[0];\n"
  ^ "  threadgroup float cached_norm[2048];\n"
  ^ "  const bool can_cache = params.k <= 2048;\n"
  ^ "  if (can_cache) {\n"
  ^ "    for (uint inner = tid; inner < params.k; inner += 256) {\n"
  ^ "      cached_norm[inner] = float(input[row * params.k + inner]) *\n"
  ^ "          float(norm_weight[inner]) * inverse;\n"
  ^ "    }\n"
  ^ "  }\n"
  ^ "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  threadgroup float best_values[256];\n"
  ^ "  threadgroup uint best_indices[256];\n"
  ^ "  float best_value = -3.402823466e+38f;\n"
  ^ "  uint best_index = 0;\n"
  ^ "  for (uint col = group_idx * 256 + tid; col < params.n; col += 256 * 256) {\n"
  ^ "    const uint weight_base = col * (params.k / 2);\n"
  ^ "    const uint scale_base = col * (params.k / 64);\n"
  ^ "    float accumulator = 0.0f;\n"
  ^ "    if (can_cache) {\n"
  ^ "      const uint groups = params.k >> 6;\n"
  ^ "      for (uint g = 0; g < groups; ++g) {\n"
  ^ "        const float s = float(scale[scale_base + g]);\n"
  ^ "        float group_acc = 0.0f;\n"
  ^ "        const uint g_packed_base = weight_base + (g << 5);\n"
  ^ "        const uint g_norm_base = g << 6;\n"
  ^ "        for (uint pair = 0; pair < 32; pair += 4) {\n"
  ^ "          device const uchar4* wp = reinterpret_cast<device const uchar4*>(weight + g_packed_base + pair);\n"
  ^ "          const uchar4 p4 = *wp;\n"
  ^ "          const int q0 = int(p4.x & 15u) - ((p4.x & 8u) ? 16 : 0);\n"
  ^ "          const int q1 = int(p4.x >> 4) - ((p4.x & 128u) ? 16 : 0);\n"
  ^ "          const int q2 = int(p4.y & 15u) - ((p4.y & 8u) ? 16 : 0);\n"
  ^ "          const int q3 = int(p4.y >> 4) - ((p4.y & 128u) ? 16 : 0);\n"
  ^ "          const int q4 = int(p4.z & 15u) - ((p4.z & 8u) ? 16 : 0);\n"
  ^ "          const int q5 = int(p4.z >> 4) - ((p4.z & 128u) ? 16 : 0);\n"
  ^ "          const int q6 = int(p4.w & 15u) - ((p4.w & 8u) ? 16 : 0);\n"
  ^ "          const int q7 = int(p4.w >> 4) - ((p4.w & 128u) ? 16 : 0);\n"
  ^ "          const uint ni = g_norm_base + (pair << 1);\n"
  ^ "          group_acc += cached_norm[ni + 0] * float(q0)\n"
  ^ "                     + cached_norm[ni + 1] * float(q1)\n"
  ^ "                     + cached_norm[ni + 2] * float(q2)\n"
  ^ "                     + cached_norm[ni + 3] * float(q3)\n"
  ^ "                     + cached_norm[ni + 4] * float(q4)\n"
  ^ "                     + cached_norm[ni + 5] * float(q5)\n"
  ^ "                     + cached_norm[ni + 6] * float(q6)\n"
  ^ "                     + cached_norm[ni + 7] * float(q7);\n"
  ^ "        }\n"
  ^ "        accumulator += group_acc * s;\n"
  ^ "      }\n"
  ^ "    } else {\n"
  ^ "      const uint groups = params.k >> 6;\n"
  ^ "      for (uint g = 0; g < groups; ++g) {\n"
  ^ "        const float s = float(scale[scale_base + g]);\n"
  ^ "        float group_acc = 0.0f;\n"
  ^ "        const uint g_packed_base = weight_base + (g << 5);\n"
  ^ "        const uint g_in_base = row * params.k + (g << 6);\n"
  ^ "        for (uint pair = 0; pair < 32; pair += 4) {\n"
  ^ "          device const uchar4* wp = reinterpret_cast<device const uchar4*>(weight + g_packed_base + pair);\n"
  ^ "          const uchar4 p4 = *wp;\n"
  ^ "          const int q0 = int(p4.x & 15u) - ((p4.x & 8u) ? 16 : 0);\n"
  ^ "          const int q1 = int(p4.x >> 4) - ((p4.x & 128u) ? 16 : 0);\n"
  ^ "          const int q2 = int(p4.y & 15u) - ((p4.y & 8u) ? 16 : 0);\n"
  ^ "          const int q3 = int(p4.y >> 4) - ((p4.y & 128u) ? 16 : 0);\n"
  ^ "          const int q4 = int(p4.z & 15u) - ((p4.z & 8u) ? 16 : 0);\n"
  ^ "          const int q5 = int(p4.z >> 4) - ((p4.z & 128u) ? 16 : 0);\n"
  ^ "          const int q6 = int(p4.w & 15u) - ((p4.w & 8u) ? 16 : 0);\n"
  ^ "          const int q7 = int(p4.w >> 4) - ((p4.w & 128u) ? 16 : 0);\n"
  ^ "          const uint ni = g_in_base + (pair << 1);\n"
  ^ "          const uint wi = (g << 6) + (pair << 1);\n"
  ^ "          group_acc += (float(input[ni + 0]) * float(norm_weight[wi + 0]) * inverse) * float(q0)\n"
  ^ "                     + (float(input[ni + 1]) * float(norm_weight[wi + 1]) * inverse) * float(q1)\n"
  ^ "                     + (float(input[ni + 2]) * float(norm_weight[wi + 2]) * inverse) * float(q2)\n"
  ^ "                     + (float(input[ni + 3]) * float(norm_weight[wi + 3]) * inverse) * float(q3)\n"
  ^ "                     + (float(input[ni + 4]) * float(norm_weight[wi + 4]) * inverse) * float(q4)\n"
  ^ "                     + (float(input[ni + 5]) * float(norm_weight[wi + 5]) * inverse) * float(q5)\n"
  ^ "                     + (float(input[ni + 6]) * float(norm_weight[wi + 6]) * inverse) * float(q6)\n"
  ^ "                     + (float(input[ni + 7]) * float(norm_weight[wi + 7]) * inverse) * float(q7);\n"
  ^ "        }\n"
  ^ "        accumulator += group_acc * s;\n"
  ^ "      }\n"
  ^ "    }\n"
  ^ "    if (accumulator > best_value ||\n"
  ^ "        (accumulator == best_value && col < best_index)) {\n"
  ^ "      best_value = accumulator;\n"
  ^ "      best_index = col;\n"
  ^ "    }\n"
  ^ (if extra_output then
       "    logits[row * params.n + col] = half(accumulator);\n"
     else "")
  ^ "  }\n"
  ^ "  best_values[tid] = best_value;\n"
  ^ "  best_indices[tid] = best_index;\n"
  ^ "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  for (uint width = 128; width > 0; width >>= 1) {\n"
  ^ "    if (tid < width) {\n"
  ^ "      float candidate = best_values[tid + width];\n"
  ^ "      uint candidate_index = best_indices[tid + width];\n"
  ^ "      if (candidate > best_values[tid] ||\n"
  ^ "          (candidate == best_values[tid] && candidate_index < best_indices[tid])) {\n"
  ^ "        best_values[tid] = candidate;\n"
  ^ "        best_indices[tid] = candidate_index;\n"
  ^ "      }\n"
  ^ "    }\n"
  ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  }\n"
  ^ "  if (tid == 0) {\n"
  ^ "    candidates[row * 256 + group_idx].value = best_values[0];\n"
  ^ "    candidates[row * 256 + group_idx].index = best_indices[0];\n"
  ^ "  }\n"
  ^ "}\n\n"

let w4a16_lm_head_reduce_kernel =
  "kernel void llmopt_w4a16_lm_head_reduce(\n"
  ^ "    device const W4A16LmHeadCandidate* candidates [[buffer(0)]],\n"
  ^ "    device uint* token_ids [[buffer(1)]],\n"
  ^ "    constant uint& m [[buffer(2)]],\n"
  ^ "    uint tid [[thread_index_in_threadgroup]],\n"
  ^ "    uint3 group_position [[threadgroup_position_in_grid]]) {\n"
  ^ "  const uint row = group_position.x;\n"
  ^ "  if (row >= m) return;\n\n"
  ^ "  threadgroup float best_values[256];\n"
  ^ "  threadgroup uint best_indices[256];\n\n"
  ^ "  best_values[tid] = candidates[row * 256 + tid].value;\n"
  ^ "  best_indices[tid] = candidates[row * 256 + tid].index;\n"
  ^ "  threadgroup_barrier(mem_flags::mem_threadgroup);\n\n"
  ^ "  for (uint width = 128; width > 0; width >>= 1) {\n"
  ^ "    if (tid < width) {\n"
  ^ "      float candidate = best_values[tid + width];\n"
  ^ "      uint candidate_index = best_indices[tid + width];\n"
  ^ "      if (candidate > best_values[tid] ||\n"
  ^ "          (candidate == best_values[tid] && candidate_index < best_indices[tid])) {\n"
  ^ "        best_values[tid] = candidate;\n"
  ^ "        best_indices[tid] = candidate_index;\n"
  ^ "      }\n"
  ^ "    }\n"
  ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  }\n\n"
  ^ "  if (tid == 0) {\n"
  ^ "    token_ids[row] = best_indices[0];\n"
  ^ "  }\n"
  ^ "}\n\n"

let w4a16_lm_head_argmax_source =
  "\nstruct W4A16LmHeadParams { uint m; uint n; uint k; float epsilon; };\n\n"
  ^ "struct W4A16LmHeadCandidate { float value; uint index; };\n\n"
  ^ w4a16_lm_head_argmax_kernel ~name:"llmopt_w4a16_lm_head_argmax_f16"
      ~value_type:"half" ~extra_output:false
  ^ w4a16_lm_head_argmax_kernel ~name:"llmopt_w4a16_lm_head_argmax_f32"
      ~value_type:"float" ~extra_output:false
  ^ w4a16_lm_head_argmax_kernel ~name:"llmopt_w4a16_lm_head_argmax_extra_f16"
      ~value_type:"half" ~extra_output:true
  ^ w4a16_lm_head_argmax_kernel ~name:"llmopt_w4a16_lm_head_argmax_extra_f32"
      ~value_type:"float" ~extra_output:true
  ^ w4a16_lm_head_argmax_stage1_kernel
      ~name:"llmopt_w4a16_lm_head_argmax_stage1_f16"
      ~value_type:"half" ~extra_output:false
  ^ w4a16_lm_head_argmax_stage1_kernel
      ~name:"llmopt_w4a16_lm_head_argmax_stage1_f32"
      ~value_type:"float" ~extra_output:false
  ^ w4a16_lm_head_argmax_stage1_kernel
      ~name:"llmopt_w4a16_lm_head_argmax_stage1_extra_f16"
      ~value_type:"half" ~extra_output:true
  ^ w4a16_lm_head_argmax_stage1_kernel
      ~name:"llmopt_w4a16_lm_head_argmax_stage1_extra_f32"
      ~value_type:"float" ~extra_output:true
  ^ w4a16_lm_head_reduce_kernel

let w4a16_lm_head_argmax_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_w4a16_lm_head_argmax_f16"
      ~operation:Kernel_abi.Operation.W4a16_lm_head_argmax
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Int32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_w4a16_lm_head_argmax_f32"
      ~operation:Kernel_abi.Operation.W4a16_lm_head_argmax
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Int32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_w4a16_lm_head_argmax_extra_f16"
      ~operation:Kernel_abi.Operation.W4a16_lm_head_argmax
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Int32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_w4a16_lm_head_argmax_extra_f32"
      ~operation:Kernel_abi.Operation.W4a16_lm_head_argmax
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Int32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_w4a16_lm_head_argmax_stage1_f16"
      ~operation:Kernel_abi.Operation.W4a16_lm_head_argmax
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Int32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_w4a16_lm_head_argmax_stage1_f32"
      ~operation:Kernel_abi.Operation.W4a16_lm_head_argmax
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Int32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_w4a16_lm_head_argmax_stage1_extra_f16"
      ~operation:Kernel_abi.Operation.W4a16_lm_head_argmax
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Int32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_w4a16_lm_head_argmax_stage1_extra_f32"
      ~operation:Kernel_abi.Operation.W4a16_lm_head_argmax
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Int32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_w4a16_lm_head_reduce"
      ~operation:Kernel_abi.Operation.W4a16_lm_head_argmax
      ~input_dtype:Ir.Dtype.Int32 ~output_dtype:Ir.Dtype.Int32 ]

let cache_source =
  {|

struct AttentionCacheParams {
  uint items;
  uint segment;
  uint heads;
  uint head_dim;
  uint group_size;
  uint token_elements;
  uint token_groups;
  uint token_stride;
  uint source_items;
  uint source_offset;
};

struct CheckpointCacheParams {
  uint checkpoint;
  uint layer;
  uint layer_elements;
  uint group_size;
  uint checkpoint_elements;
  uint checkpoint_groups;
  uint checkpoint_stride;
};

kernel void llmopt_cache_pack_attention_q8(
    device const half* source [[buffer(0)]],
    device const uint* slots [[buffer(1)]],
    device uchar* pool [[buffer(2)]],
    constant AttentionCacheParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint groups_per_head = params.head_dim / params.group_size;
  const uint segment_groups = params.heads * groups_per_head;
  if (gid >= params.items * segment_groups) return;
  const uint item = gid / segment_groups;
  const uint local_group = gid - item * segment_groups;
  const uint head = local_group / groups_per_head;
  const uint group_in_head = local_group - head * groups_per_head;
  const uint source_base =
      (head * params.source_items + params.source_offset + item) * params.head_dim
      + group_in_head * params.group_size;
  float maximum = 0.0f;
  for (uint index = 0; index < params.group_size; ++index)
    maximum = max(maximum, abs(float(source[source_base + index])));
  const half stored_scale = half(
      maximum == 0.0f ? 1.0f : maximum / 127.0f);
  const float scale = float(stored_scale);
  const uint slot_base = slots[item] * params.token_stride;
  device char* values = reinterpret_cast<device char*>(pool + slot_base);
  device half* scales = reinterpret_cast<device half*>(
      pool + slot_base + params.token_elements);
  const uint value_base =
      params.segment * params.heads * params.head_dim
      + head * params.head_dim + group_in_head * params.group_size;
  for (uint index = 0; index < params.group_size; ++index) {
    const int quantized = clamp(
        int(rint(float(source[source_base + index]) / scale)), -127, 127);
    values[value_base + index] = char(quantized);
  }
  scales[params.segment * segment_groups + local_group] = stored_scale;
}

kernel void llmopt_cache_pack_attention_q8_simd(
    device const half* source [[buffer(0)]],
    device const uint* slots [[buffer(1)]],
    device uchar* pool [[buffer(2)]],
    constant AttentionCacheParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint groups_per_head = params.head_dim / params.group_size;
  const uint segment_groups = params.heads * groups_per_head;
  const uint logical_group = gid / 32;
  if (logical_group >= params.items * segment_groups) return;
  const uint item = logical_group / segment_groups;
  const uint local_group = logical_group - item * segment_groups;
  const uint head = local_group / groups_per_head;
  const uint group_in_head = local_group - head * groups_per_head;
  const uint source_base =
      (head * params.source_items + params.source_offset + item) * params.head_dim
      + group_in_head * params.group_size;
  float maximum = 0.0f;
  for (uint index = lane; index < params.group_size; index += 32)
    maximum = max(maximum, abs(float(source[source_base + index])));
  maximum = simd_max(maximum);
  const half stored_scale = half(
      maximum == 0.0f ? 1.0f : maximum / 127.0f);
  const float scale = float(stored_scale);
  const uint slot_base = slots[item] * params.token_stride;
  device char* values = reinterpret_cast<device char*>(pool + slot_base);
  device half* scales = reinterpret_cast<device half*>(
      pool + slot_base + params.token_elements);
  const uint value_base =
      params.segment * params.heads * params.head_dim
      + head * params.head_dim + group_in_head * params.group_size;
  for (uint index = lane; index < params.group_size; index += 32) {
    const int quantized = clamp(
        int(rint(float(source[source_base + index]) / scale)), -127, 127);
    values[value_base + index] = char(quantized);
  }
  if (lane == 0)
    scales[params.segment * segment_groups + local_group] = stored_scale;
}

kernel void llmopt_cache_unpack_attention_q8(
    device const uchar* pool [[buffer(0)]],
    device const uint* slots [[buffer(1)]],
    device half* destination [[buffer(2)]],
    constant AttentionCacheParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint segment_elements = params.heads * params.head_dim;
  if (gid >= params.items * segment_elements) return;
  const uint item = gid / segment_elements;
  const uint local = gid - item * segment_elements;
  const uint head = local / params.head_dim;
  const uint within_head = local - head * params.head_dim;
  const uint groups_per_head = params.head_dim / params.group_size;
  const uint segment_groups = params.heads * groups_per_head;
  const uint local_group = head * groups_per_head
      + within_head / params.group_size;
  const uint slot_base = slots[item] * params.token_stride;
  device const char* values =
      reinterpret_cast<device const char*>(pool + slot_base);
  device const half* scales = reinterpret_cast<device const half*>(
      pool + slot_base + params.token_elements);
  const uint value_index = params.segment * segment_elements + local;
  const uint destination_index =
      (head * params.items + item) * params.head_dim + within_head;
  destination[destination_index] = half(
      float(values[value_index])
      * float(scales[params.segment * segment_groups + local_group]));
}

kernel void llmopt_cache_unpack_attention_q8_vec4(
    device const uchar* pool [[buffer(0)]],
    device const uint* slots [[buffer(1)]],
    device half* destination [[buffer(2)]],
    constant AttentionCacheParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint segment_elements = params.heads * params.head_dim;
  const uint vectors = params.items * segment_elements / 4;
  if (gid >= vectors) return;
  const uint linear = gid * 4;
  const uint item = linear / segment_elements;
  const uint local = linear - item * segment_elements;
  const uint head = local / params.head_dim;
  const uint within_head = local - head * params.head_dim;
  const uint groups_per_head = params.head_dim / params.group_size;
  const uint segment_groups = params.heads * groups_per_head;
  const uint local_group = head * groups_per_head
      + within_head / params.group_size;
  const uint slot_base = slots[item] * params.token_stride;
  device const char* values =
      reinterpret_cast<device const char*>(pool + slot_base);
  device const half* scales = reinterpret_cast<device const half*>(
      pool + slot_base + params.token_elements);
  const uint value_index = params.segment * segment_elements + local;
  const uint destination_index =
      (head * params.items + item) * params.head_dim + within_head;
  const char4 quantized =
      *reinterpret_cast<device const char4*>(values + value_index);
  const float scale =
      float(scales[params.segment * segment_groups + local_group]);
  *reinterpret_cast<device half4*>(destination + destination_index) =
      half4(float4(quantized) * scale);
}

kernel void llmopt_cache_pack_checkpoint_q8(
    device const half* source [[buffer(0)]],
    device uchar* pool [[buffer(1)]],
    constant CheckpointCacheParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  const uint layer_groups = params.layer_elements / params.group_size;
  if (gid >= layer_groups) return;
  const uint source_base = gid * params.group_size;
  float maximum = 0.0f;
  for (uint index = 0; index < params.group_size; ++index)
    maximum = max(maximum, abs(float(source[source_base + index])));
  const half stored_scale = half(
      maximum == 0.0f ? 1.0f : maximum / 127.0f);
  const float scale = float(stored_scale);
  const uint checkpoint_base = params.checkpoint * params.checkpoint_stride;
  device char* values =
      reinterpret_cast<device char*>(pool + checkpoint_base);
  device half* scales = reinterpret_cast<device half*>(
      pool + checkpoint_base + params.checkpoint_elements);
  const uint value_base =
      params.layer * params.layer_elements + gid * params.group_size;
  for (uint index = 0; index < params.group_size; ++index) {
    const int quantized = clamp(
        int(rint(float(source[source_base + index]) / scale)), -127, 127);
    values[value_base + index] = char(quantized);
  }
  scales[params.layer * layer_groups + gid] = stored_scale;
}

kernel void llmopt_cache_pack_checkpoint_q8_simd(
    device const half* source [[buffer(0)]],
    device uchar* pool [[buffer(1)]],
    constant CheckpointCacheParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint layer_groups = params.layer_elements / params.group_size;
  const uint logical_group = gid / 32;
  if (logical_group >= layer_groups) return;
  const uint source_base = logical_group * params.group_size;
  float maximum = 0.0f;
  for (uint index = lane; index < params.group_size; index += 32)
    maximum = max(maximum, abs(float(source[source_base + index])));
  maximum = simd_max(maximum);
  const half stored_scale = half(
      maximum == 0.0f ? 1.0f : maximum / 127.0f);
  const float scale = float(stored_scale);
  const uint checkpoint_base = params.checkpoint * params.checkpoint_stride;
  device char* values =
      reinterpret_cast<device char*>(pool + checkpoint_base);
  device half* scales = reinterpret_cast<device half*>(
      pool + checkpoint_base + params.checkpoint_elements);
  const uint value_base = params.layer * params.layer_elements
      + logical_group * params.group_size;
  for (uint index = lane; index < params.group_size; index += 32) {
    const int quantized = clamp(
        int(rint(float(source[source_base + index]) / scale)), -127, 127);
    values[value_base + index] = char(quantized);
  }
  if (lane == 0)
    scales[params.layer * layer_groups + logical_group] = stored_scale;
}

kernel void llmopt_cache_unpack_checkpoint_q8(
    device const uchar* pool [[buffer(0)]],
    device half* destination [[buffer(1)]],
    constant CheckpointCacheParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.layer_elements) return;
  const uint checkpoint_base = params.checkpoint * params.checkpoint_stride;
  device const char* values =
      reinterpret_cast<device const char*>(pool + checkpoint_base);
  device const half* scales = reinterpret_cast<device const half*>(
      pool + checkpoint_base + params.checkpoint_elements);
  const uint layer_groups = params.layer_elements / params.group_size;
  const uint value_index = params.layer * params.layer_elements + gid;
  const uint scale_index =
      params.layer * layer_groups + gid / params.group_size;
  destination[gid] = half(
      float(values[value_index]) * float(scales[scale_index]));
}

kernel void llmopt_cache_unpack_checkpoint_q8_vec4(
    device const uchar* pool [[buffer(0)]],
    device half* destination [[buffer(1)]],
    constant CheckpointCacheParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  const uint vectors = params.layer_elements / 4;
  if (gid >= vectors) return;
  const uint local = gid * 4;
  const uint checkpoint_base = params.checkpoint * params.checkpoint_stride;
  device const char* values =
      reinterpret_cast<device const char*>(pool + checkpoint_base);
  device const half* scales = reinterpret_cast<device const half*>(
      pool + checkpoint_base + params.checkpoint_elements);
  const uint layer_groups = params.layer_elements / params.group_size;
  const uint value_index = params.layer * params.layer_elements + local;
  const uint scale_index = params.layer * layer_groups + local / params.group_size;
  const char4 quantized =
      *reinterpret_cast<device const char4*>(values + value_index);
  const float scale = float(scales[scale_index]);
  *reinterpret_cast<device half4*>(destination + local) =
      half4(float4(quantized) * scale);
}

constant uint Q8_PAGED_ATTENTION_SIMD_WIDTH = 32;
constant uint Q8_PAGED_ATTENTION_ROWS_PER_THREADGROUP = 8;

struct Q8PagedAttentionParams {
  uint batches;
  uint query_heads;
  uint kv_heads;
  uint past_length;
  uint head_dim;
  uint mask_batches;
  uint mask_heads;
  uint cache_layer;
  uint attention_layers;
  uint group_size;
  uint token_stride;
  float scale;
  uint query_length;
};

inline float llmopt_q8_paged_value(
    device const uchar* pool,
    device const uint* slots,
    constant Q8PagedAttentionParams& params,
    uint key_position,
    uint segment,
    uint kv_head,
    uint dimension) {
  const uint token_elements =
      2 * params.attention_layers * params.kv_heads * params.head_dim;
  const uint groups_per_head = params.head_dim / params.group_size;
  const uint segment_groups = params.kv_heads * groups_per_head;
  const uint slot_base = slots[key_position] * params.token_stride;
  device const char* values =
      reinterpret_cast<device const char*>(pool + slot_base);
  device const half* scales = reinterpret_cast<device const half*>(
      pool + slot_base + token_elements);
  const uint value_index = segment * params.kv_heads * params.head_dim
      + kv_head * params.head_dim + dimension;
  const uint scale_index = segment * segment_groups
      + kv_head * groups_per_head + (dimension / params.group_size);
  return float(values[value_index]) * float(scales[scale_index]);
}

kernel void llmopt_attention_q8_paged_simd_h64(
    device const half* query [[buffer(0)]],
    device const half* current_key [[buffer(1)]],
    device const half* current_value [[buffer(2)]],
    device const uchar* pool [[buffer(3)]],
    device const uint* slots [[buffer(4)]],
    device const uchar* mask [[buffer(5)]],
    device half* output [[buffer(6)]],
    constant Q8PagedAttentionParams& params [[buffer(7)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = threadgroup_position.x
      * Q8_PAGED_ATTENTION_ROWS_PER_THREADGROUP + simdgroup;
  const uint q_len = params.query_length == 0 ? 1 : params.query_length;
  const uint rows = params.batches * params.query_heads * q_len;
  if (row >= rows) return;
  const uint q_pos = row % q_len;
  const uint head = (row / q_len) % params.query_heads;
  const uint batch = row / (q_len * params.query_heads);
  const uint heads_per_kv = params.query_heads / params.kv_heads;
  const uint kv_head = head / heads_per_kv;
  const uint query_base =
      ((batch * params.query_heads + head) * q_len + q_pos) * params.head_dim;
  const uint current_base =
      (batch * params.kv_heads + kv_head) * q_len * params.head_dim;
  const uint key_length = params.past_length + q_pos + 1;
  const uint key_segment = params.cache_layer * 2;
  const uint value_segment = key_segment + 1;
  float maximum = -INFINITY;
  float denominator = 0.0f;
  float result_low = 0.0f;
  float result_high = 0.0f;
  for (uint key_position = 0; key_position < key_length; ++key_position) {
    if (mask != nullptr && params.mask_batches > 0) {
      const uint mask_batch = params.mask_batches == 1 ? 0 : batch;
      const uint mask_head = params.mask_heads == 1 ? 0 : head;
      const uint mask_index =
          ((mask_batch * params.mask_heads + mask_head) * q_len + q_pos)
              * key_length + key_position;
      if (mask[mask_index] == 0) continue;
    }
    float partial_score = 0.0f;
    for (uint dimension = lane; dimension < params.head_dim;
         dimension += Q8_PAGED_ATTENTION_SIMD_WIDTH) {
      const float key_value = key_position < params.past_length
          ? llmopt_q8_paged_value(pool, slots, params, key_position,
                key_segment, kv_head, dimension)
          : float(current_key[current_base + (key_position - params.past_length) * params.head_dim + dimension]);
      partial_score += float(query[query_base + dimension]) * key_value;
    }
    const float score = simd_sum(partial_score) * params.scale;
    const float next_maximum = max(maximum, score);
    const float previous_scale = denominator == 0.0f ? 0.0f
        : exp(maximum - next_maximum);
    const float current_scale = exp(score - next_maximum);
    if (lane < params.head_dim) {
      const float value = key_position < params.past_length
          ? llmopt_q8_paged_value(pool, slots, params, key_position,
                value_segment, kv_head, lane)
          : float(current_value[current_base + (key_position - params.past_length) * params.head_dim + lane]);
      result_low = result_low * previous_scale + current_scale * value;
    }
    const uint high_dimension = lane + Q8_PAGED_ATTENTION_SIMD_WIDTH;
    if (high_dimension < params.head_dim) {
      const float value = key_position < params.past_length
          ? llmopt_q8_paged_value(pool, slots, params, key_position,
                value_segment, kv_head, high_dimension)
          : float(current_value[current_base + (key_position - params.past_length) * params.head_dim + high_dimension]);
      result_high = result_high * previous_scale + current_scale * value;
    }
    denominator = denominator * previous_scale + current_scale;
    maximum = next_maximum;
  }
  const uint output_base =
      ((batch * params.query_heads + head) * q_len + q_pos) * params.head_dim;
  if (lane < params.head_dim)
    output[output_base + lane] = half(
        denominator == 0.0f ? 0.0f : result_low / denominator);
  const uint high_dimension = lane + Q8_PAGED_ATTENTION_SIMD_WIDTH;
  if (high_dimension < params.head_dim)
    output[output_base + high_dimension] = half(
        denominator == 0.0f ? 0.0f : result_high / denominator);
}
|}

let cache_entry name input_dtype output_dtype =
  kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1) ~name
    ~operation:Kernel_abi.Operation.Cache ~input_dtype ~output_dtype

let cache_entries =
  [ cache_entry "llmopt_cache_pack_attention_q8" Ir.Dtype.Float16
      Ir.Dtype.Int8;
    cache_entry "llmopt_cache_pack_attention_q8_simd" Ir.Dtype.Float16
      Ir.Dtype.Int8;
    cache_entry "llmopt_cache_unpack_attention_q8" Ir.Dtype.Int8
      Ir.Dtype.Float16;
    cache_entry "llmopt_cache_unpack_attention_q8_vec4" Ir.Dtype.Int8
      Ir.Dtype.Float16;
    cache_entry "llmopt_cache_pack_checkpoint_q8" Ir.Dtype.Float16
      Ir.Dtype.Int8;
    cache_entry "llmopt_cache_pack_checkpoint_q8_simd" Ir.Dtype.Float16
      Ir.Dtype.Int8;
    cache_entry "llmopt_cache_unpack_checkpoint_q8" Ir.Dtype.Int8
      Ir.Dtype.Float16;
    cache_entry "llmopt_cache_unpack_checkpoint_q8_vec4" Ir.Dtype.Int8
      Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_attention_q8_paged_simd_h64"
      ~operation:Kernel_abi.Operation.Attention
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16 ]

let add_cache_kernels program =
  Program.make ~source:(Program.source program ^ cache_source)
    ~kernels:(Program.kernels program @ cache_entries)

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

let rms_norm_source =
  "\nconstant uint RMS_NORM_SIMD_WIDTH = 32;\n"
  ^ "constant uint RMS_NORM_ROWS_PER_THREADGROUP = 8;\n\n"
  ^ "struct RmsNormParams { uint rows; uint width; float epsilon; };\n\n"
  ^ "kernel void llmopt_rms_norm_f32_f16(\n"
  ^ "    device const float* input [[buffer(0)]],\n"
  ^ "    device const half* weight [[buffer(1)]],\n"
  ^ "    device half* output [[buffer(2)]],\n"
  ^ "    constant RmsNormParams& params [[buffer(3)]],\n"
  ^ "    uint row [[thread_position_in_grid]]) {\n"
  ^ "  if (row >= params.rows) return;\n"
  ^ "  const uint base = row * params.width;\n"
  ^ "  float square_sum = 0.0f;\n"
  ^ "  for (uint col = 0; col < params.width; ++col) {\n"
  ^ "    const float value = input[base + col];\n"
  ^ "    square_sum += value * value;\n"
  ^ "  }\n"
  ^ "  const float inverse = rsqrt(square_sum / float(params.width) + params.epsilon);\n"
  ^ "  for (uint col = 0; col < params.width; ++col)\n"
  ^ "    output[base + col] = half(input[base + col] * inverse * float(weight[col]));\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_rms_norm_f16(\n"
  ^ "    device const half* input [[buffer(0)]],\n"
  ^ "    device const half* weight [[buffer(1)]],\n"
  ^ "    device half* output [[buffer(2)]],\n"
  ^ "    constant RmsNormParams& params [[buffer(3)]],\n"
  ^ "    uint row [[thread_position_in_grid]]) {\n"
  ^ "  if (row >= params.rows) return;\n"
  ^ "  const uint base = row * params.width;\n"
  ^ "  float square_sum = 0.0f;\n"
  ^ "  for (uint col = 0; col < params.width; ++col) {\n"
  ^ "    const float value = float(input[base + col]);\n"
  ^ "    square_sum += value * value;\n"
  ^ "  }\n"
  ^ "  const float inverse = rsqrt(square_sum / float(params.width) + params.epsilon);\n"
  ^ "  for (uint col = 0; col < params.width; ++col)\n"
  ^ "    output[base + col] = half(float(input[base + col]) * inverse * float(weight[col]));\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_rms_norm_f32_f16_simd(\n"
  ^ "    device const float* input [[buffer(0)]],\n"
  ^ "    device const half* weight [[buffer(1)]],\n"
  ^ "    device half* output [[buffer(2)]],\n"
  ^ "    constant RmsNormParams& params [[buffer(3)]],\n"
  ^ "    uint3 threadgroup_position [[threadgroup_position_in_grid]],\n"
  ^ "    uint simdgroup [[simdgroup_index_in_threadgroup]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  const uint row = threadgroup_position.x * RMS_NORM_ROWS_PER_THREADGROUP\n"
  ^ "      + simdgroup;\n"
  ^ "  if (row >= params.rows) return;\n"
  ^ "  const uint base = row * params.width;\n"
  ^ "  float square_sum = 0.0f;\n"
  ^ "  for (uint col = lane; col < params.width; col += RMS_NORM_SIMD_WIDTH) {\n"
  ^ "    const float value = input[base + col];\n"
  ^ "    square_sum += value * value;\n"
  ^ "  }\n"
  ^ "  square_sum = simd_sum(square_sum);\n"
  ^ "  const float inverse = rsqrt(square_sum / float(params.width) + params.epsilon);\n"
  ^ "  for (uint col = lane; col < params.width; col += RMS_NORM_SIMD_WIDTH)\n"
  ^ "    output[base + col] = half(input[base + col] * inverse * float(weight[col]));\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_rms_norm_f16_simd(\n"
  ^ "    device const half* input [[buffer(0)]],\n"
  ^ "    device const half* weight [[buffer(1)]],\n"
  ^ "    device half* output [[buffer(2)]],\n"
  ^ "    constant RmsNormParams& params [[buffer(3)]],\n"
  ^ "    uint3 threadgroup_position [[threadgroup_position_in_grid]],\n"
  ^ "    uint simdgroup [[simdgroup_index_in_threadgroup]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  const uint row = threadgroup_position.x * RMS_NORM_ROWS_PER_THREADGROUP\n"
  ^ "      + simdgroup;\n"
  ^ "  if (row >= params.rows) return;\n"
  ^ "  const uint base = row * params.width;\n"
  ^ "  float square_sum = 0.0f;\n"
  ^ "  for (uint col = lane; col < params.width; col += RMS_NORM_SIMD_WIDTH) {\n"
  ^ "    const float value = float(input[base + col]);\n"
  ^ "    square_sum += value * value;\n"
  ^ "  }\n"
  ^ "  square_sum = simd_sum(square_sum);\n"
  ^ "  const float inverse = rsqrt(square_sum / float(params.width) + params.epsilon);\n"
  ^ "  for (uint col = lane; col < params.width; col += RMS_NORM_SIMD_WIDTH)\n"
  ^ "    output[base + col] = half(float(input[base + col]) * inverse * float(weight[col]));\n"
  ^ "}\n"

let rms_norm_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_rms_norm_f32_f16_simd"
      ~operation:Kernel_abi.Operation.Rms_norm
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_rms_norm_f16_simd"
      ~operation:Kernel_abi.Operation.Rms_norm
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16 ]

let rms_rope_source =
  "\nconstant uint RMS_ROPE_SIMD_WIDTH = 32;\n"
  ^ "constant uint RMS_ROPE_ROWS_PER_THREADGROUP = 8;\n\n"
  ^ "struct RmsRopeParams {\n"
  ^ "  uint batches; uint tokens; uint heads; uint width;\n"
  ^ "  uint half_dimension; uint trig_batches; float epsilon;\n"
  ^ "};\n\n"
  ^ "kernel void llmopt_rms_rope_f16_simd_h64(\n"
  ^ "    device const half* input [[buffer(0)]],\n"
  ^ "    device const half* weight [[buffer(1)]],\n"
  ^ "    device const half* cosine [[buffer(2)]],\n"
  ^ "    device const half* sine [[buffer(3)]],\n"
  ^ "    device half* output [[buffer(4)]],\n"
  ^ "    constant RmsRopeParams& params [[buffer(5)]],\n"
  ^ "    uint3 threadgroup_position [[threadgroup_position_in_grid]],\n"
  ^ "    uint simdgroup [[simdgroup_index_in_threadgroup]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  const uint row = threadgroup_position.x * RMS_ROPE_ROWS_PER_THREADGROUP\n"
  ^ "      + simdgroup;\n"
  ^ "  const uint rows = params.batches * params.heads * params.tokens;\n"
  ^ "  if (row >= rows) return;\n"
  ^ "  const uint batch = row / (params.heads * params.tokens);\n"
  ^ "  const uint head_token = row % (params.heads * params.tokens);\n"
  ^ "  const uint head = head_token / params.tokens;\n"
  ^ "  const uint token = head_token % params.tokens;\n"
  ^ "  const uint input_row = (batch * params.tokens + token) * params.heads + head;\n"
  ^ "  const uint input_base = input_row * params.width;\n"
  ^ "  float square_sum = 0.0f;\n"
  ^ "  for (uint col = lane; col < params.width; col += RMS_ROPE_SIMD_WIDTH) {\n"
  ^ "    const float value = float(input[input_base + col]);\n"
  ^ "    square_sum += value * value;\n"
  ^ "  }\n"
  ^ "  square_sum = simd_sum(square_sum);\n"
  ^ "  const float inverse = rsqrt(square_sum / float(params.width) + params.epsilon);\n"
  ^ "  const uint trig_batch = params.trig_batches == 1 ? 0 : batch;\n"
  ^ "  const uint trig_base = (trig_batch * params.tokens + token) * params.width;\n"
  ^ "  const uint output_base = row * params.width;\n"
  ^ "  for (uint col = lane; col < params.width; col += RMS_ROPE_SIMD_WIDTH) {\n"
  ^ "    const half normalized = half(float(input[input_base + col]) * inverse\n"
  ^ "        * float(weight[col]));\n"
  ^ "    const uint rotated_col = col < params.half_dimension\n"
  ^ "        ? col + params.half_dimension : col - params.half_dimension;\n"
  ^ "    half rotated = half(float(input[input_base + rotated_col]) * inverse\n"
  ^ "        * float(weight[rotated_col]));\n"
  ^ "    if (col < params.half_dimension) rotated = -rotated;\n"
  ^ "    const half direct = half(normalized * cosine[trig_base + col]);\n"
  ^ "    const half turned = half(rotated * sine[trig_base + col]);\n"
  ^ "    output[output_base + col] = half(direct + turned);\n"
  ^ "  }\n"
  ^ "}\n"

let rms_rope_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_rms_rope_f16_simd_h64"
      ~operation:Kernel_abi.Operation.Rms_rope
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16 ]

let rms_rope_qk_source =
  "\nconstant uint RMS_ROPE_QK_SIMD_WIDTH = 32;\n"
  ^ "constant uint RMS_ROPE_QK_ROWS_PER_THREADGROUP = 8;\n\n"
  ^ "struct RmsRopeQKParams {\n"
  ^ "  uint batches; uint tokens; uint q_heads; uint k_heads;\n"
  ^ "  uint width; uint half_dimension; uint trig_batches;\n"
  ^ "  float epsilon;\n"
  ^ "};\n\n"
  ^ "kernel void llmopt_rms_rope_qk_f16_simd_h64(\n"
  ^ "    device const half* q_input [[buffer(0)]],\n"
  ^ "    device const half* q_weight [[buffer(1)]],\n"
  ^ "    device const half* k_input [[buffer(2)]],\n"
  ^ "    device const half* k_weight [[buffer(3)]],\n"
  ^ "    device const half* cosine [[buffer(4)]],\n"
  ^ "    device const half* sine [[buffer(5)]],\n"
  ^ "    device half* q_output [[buffer(6)]],\n"
  ^ "    device half* k_output [[buffer(7)]],\n"
  ^ "    constant RmsRopeQKParams& params [[buffer(8)]],\n"
  ^ "    uint3 threadgroup_position [[threadgroup_position_in_grid]],\n"
  ^ "    uint simdgroup [[simdgroup_index_in_threadgroup]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  const uint row = threadgroup_position.x * RMS_ROPE_QK_ROWS_PER_THREADGROUP + simdgroup;\n"
  ^ "  const uint total_rows = params.batches * (params.q_heads + params.k_heads) * params.tokens;\n"
  ^ "  if (row >= total_rows) return;\n"
  ^ "  const uint q_rows = params.batches * params.q_heads * params.tokens;\n"
  ^ "  const bool is_q = (row < q_rows);\n"
  ^ "  const uint local_row = is_q ? row : (row - q_rows);\n"
  ^ "  const uint heads = is_q ? params.q_heads : params.k_heads;\n"
  ^ "  const uint batch = local_row / (heads * params.tokens);\n"
  ^ "  const uint head_token = local_row % (heads * params.tokens);\n"
  ^ "  const uint head = head_token / params.tokens;\n"
  ^ "  const uint token = head_token % params.tokens;\n"
  ^ "  const uint input_row = (batch * params.tokens + token) * heads + head;\n"
  ^ "  const uint input_base = input_row * params.width;\n"
  ^ "  device const half* input = is_q ? q_input : k_input;\n"
  ^ "  device const half* weight = is_q ? q_weight : k_weight;\n"
  ^ "  device half* output = is_q ? q_output : k_output;\n"
  ^ "  float square_sum = 0.0f;\n"
  ^ "  for (uint col = lane; col < params.width; col += RMS_ROPE_QK_SIMD_WIDTH) {\n"
  ^ "    const float value = float(input[input_base + col]);\n"
  ^ "    square_sum += value * value;\n"
  ^ "  }\n"
  ^ "  square_sum = simd_sum(square_sum);\n"
  ^ "  const float inverse = rsqrt(square_sum / float(params.width) + params.epsilon);\n"
  ^ "  const uint trig_batch = params.trig_batches == 1 ? 0 : batch;\n"
  ^ "  const uint trig_base = (trig_batch * params.tokens + token) * params.width;\n"
  ^ "  const uint output_base = local_row * params.width;\n"
  ^ "  for (uint col = lane; col < params.width; col += RMS_ROPE_QK_SIMD_WIDTH) {\n"
  ^ "    const half normalized = half(float(input[input_base + col]) * inverse * float(weight[col]));\n"
  ^ "    const uint rotated_col = col < params.half_dimension ? col + params.half_dimension : col - params.half_dimension;\n"
  ^ "    half rotated = half(float(input[input_base + rotated_col]) * inverse * float(weight[rotated_col]));\n"
  ^ "    if (col < params.half_dimension) rotated = -rotated;\n"
  ^ "    const half direct = half(normalized * cosine[trig_base + col]);\n"
  ^ "    const half turned = half(rotated * sine[trig_base + col]);\n"
  ^ "    output[output_base + col] = half(direct + turned);\n"
  ^ "  }\n"
  ^ "}\n"

let rms_rope_qk_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_rms_rope_qk_f16_simd_h64"
      ~operation:Kernel_abi.Operation.Rms_rope_qk
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16 ]

let short_conv_source =
  "\nstruct ShortConvParams {\n"
  ^ "  uint batches; uint channels; uint input_width; uint output_width;\n"
  ^ "  uint kernel_width; uint stride; uint padding; uint dilation;\n"
  ^ "};\n\n"
  ^ "kernel void llmopt_short_conv_f16(\n"
  ^ "    device const half* input [[buffer(0)]],\n"
  ^ "    device const half* weight [[buffer(1)]],\n"
  ^ "    device half* output [[buffer(2)]],\n"
  ^ "    constant ShortConvParams& params [[buffer(3)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  const uint count = params.batches * params.channels * params.output_width;\n"
  ^ "  if (gid >= count) return;\n"
  ^ "  const uint position = gid % params.output_width;\n"
  ^ "  const uint channel = (gid / params.output_width) % params.channels;\n"
  ^ "  const uint batch = gid / (params.output_width * params.channels);\n"
  ^ "  float accumulator = 0.0f;\n"
  ^ "  for (uint tap = 0; tap < params.kernel_width; ++tap) {\n"
  ^ "    const int source_position = int(position * params.stride)\n"
  ^ "        - int(params.padding) + int(tap * params.dilation);\n"
  ^ "    if (source_position >= 0 && source_position < int(params.input_width)) {\n"
  ^ "      const uint input_index = ((batch * params.channels + channel)\n"
  ^ "          * params.input_width) + uint(source_position);\n"
  ^ "      const uint weight_index = channel * params.kernel_width + tap;\n"
  ^ "      accumulator += float(input[input_index]) * float(weight[weight_index]);\n"
  ^ "    }\n"
  ^ "  }\n"
  ^ "  output[gid] = half(accumulator);\n"
  ^ "}\n"

let short_conv_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_short_conv_f16"
      ~operation:Kernel_abi.Operation.Short_conv
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16 ]

let short_conv_step_source =
  "\nstruct ShortConvStepParams {\n"
  ^ "  uint channels;\n"
  ^ "};\n\n"
  ^ "kernel void llmopt_short_conv_step_f16(\n"
  ^ "    device const half* in_proj [[buffer(0)]],\n"
  ^ "    device half* conv_state [[buffer(1)]],\n"
  ^ "    device const half* conv_weight [[buffer(2)]],\n"
  ^ "    device half* output [[buffer(3)]],\n"
  ^ "    constant ShortConvStepParams& params [[buffer(4)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  if (gid >= params.channels) return;\n"
  ^ "  const uint c = gid;\n"
  ^ "  const uint channels = params.channels;\n"
  ^ "  const float x0 = float(in_proj[c]);\n"
  ^ "  const float x1 = float(in_proj[channels + c]);\n"
  ^ "  const float x2 = float(in_proj[2 * channels + c]);\n"
  ^ "  const float g = x0 * x2;\n"
  ^ "  const uint state_base = c * 3;\n"
  ^ "  const float s0 = float(conv_state[state_base + 1]);\n"
  ^ "  const float s1 = float(conv_state[state_base + 2]);\n"
  ^ "  const float s2 = g;\n"
  ^ "  conv_state[state_base + 0] = half(s0);\n"
  ^ "  conv_state[state_base + 1] = half(s1);\n"
  ^ "  conv_state[state_base + 2] = half(s2);\n"
  ^ "  const uint weight_base = c * 3;\n"
  ^ "  const float w0 = float(conv_weight[weight_base + 0]);\n"
  ^ "  const float w1 = float(conv_weight[weight_base + 1]);\n"
  ^ "  const float w2 = float(conv_weight[weight_base + 2]);\n"
  ^ "  const float conv_out = s0 * w0 + s1 * w1 + s2 * w2;\n"
  ^ "  output[c] = half(x1 * conv_out);\n"
  ^ "}\n"

let short_conv_step_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_short_conv_step_f16"
      ~operation:Kernel_abi.Operation.Short_conv_step
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
      ^ String.sub source (offset + needle_length) (source_length - offset - needle_length)

let short_conv_step_fused_source =
  replace_first ~needle:"llmopt_short_conv_step_f16"
    ~replacement:"llmopt_short_conv_step_fused_f16" short_conv_step_source

let short_conv_step_fused_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_short_conv_step_fused_f16"
      ~operation:Kernel_abi.Operation.Short_conv_step
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16 ]

let short_conv_prefill_source =
  "\n#ifndef LLMOPT_SHORT_CONV_PREFILL_PARAMS\n"
  ^ "#define LLMOPT_SHORT_CONV_PREFILL_PARAMS\n"
  ^ "struct ShortConvPrefillParams {\n"
  ^ "  uint tokens;\n"
  ^ "  uint channels;\n"
  ^ "  uint has_initial_state;\n"
  ^ "};\n"
  ^ "#endif\n\n"
  ^ "kernel void llmopt_short_conv_prefill_f16(\n"
  ^ "    device const half* in_proj [[buffer(0)]],\n"
  ^ "    device const half* conv_weight [[buffer(1)]],\n"
  ^ "    device half* output [[buffer(2)]],\n"
  ^ "    device half* conv_state_out [[buffer(3)]],\n"
  ^ "    constant ShortConvPrefillParams& params [[buffer(4)]],\n"
  ^ "    device const half* conv_state_in [[buffer(5)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  if (gid >= params.channels) return;\n"
  ^ "  const uint c = gid;\n"
  ^ "  const uint tokens = params.tokens;\n"
  ^ "  const uint channels = params.channels;\n"
  ^ "  const uint weight_base = c * 3;\n"
  ^ "  const float w0 = float(conv_weight[weight_base + 0]);\n"
  ^ "  const float w1 = float(conv_weight[weight_base + 1]);\n"
  ^ "  const float w2 = float(conv_weight[weight_base + 2]);\n"
  ^ "  const uint state_base = c * 3;\n"
  ^ "  float g_prev3 = 0.0f;\n"
  ^ "  float g_prev2 = 0.0f;\n"
  ^ "  float g_prev1 = 0.0f;\n"
  ^ "  if (params.has_initial_state != 0 && conv_state_in != nullptr) {\n"
  ^ "    g_prev3 = float(conv_state_in[state_base + 0]);\n"
  ^ "    g_prev2 = float(conv_state_in[state_base + 1]);\n"
  ^ "    g_prev1 = float(conv_state_in[state_base + 2]);\n"
  ^ "  }\n"
  ^ "  for (uint t = 0; t < tokens; ++t) {\n"
  ^ "    const uint in_offset = t * (3 * channels) + c;\n"
  ^ "    const float x0 = float(in_proj[in_offset]);\n"
  ^ "    const float x1 = float(in_proj[in_offset + channels]);\n"
  ^ "    const float x2 = float(in_proj[in_offset + 2 * channels]);\n"
  ^ "    const float g_curr = x0 * x2;\n"
  ^ "    const float conv_out = g_prev2 * w0 + g_prev1 * w1 + g_curr * w2;\n"
  ^ "    output[t * channels + c] = half(x1 * conv_out);\n"
  ^ "    g_prev3 = g_prev2;\n"
  ^ "    g_prev2 = g_prev1;\n"
  ^ "    g_prev1 = g_curr;\n"
  ^ "  }\n"
  ^ "  conv_state_out[state_base + 0] = half(g_prev3);\n"
  ^ "  conv_state_out[state_base + 1] = half(g_prev2);\n"
  ^ "  conv_state_out[state_base + 2] = half(g_prev1);\n"
  ^ "}\n"

let short_conv_prefill_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_short_conv_prefill_f16"
      ~operation:Kernel_abi.Operation.Short_conv_prefill
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16 ]

(* Suffix prefill feeds the recurrent checkpoint through a fourth input. The
   dispatch binds that buffer at index 4 and the constant parameters at index
   5, so the init variant swaps the two annotations of the base kernel. *)
let short_conv_prefill_init_source =
  short_conv_prefill_source
  |> replace_first ~needle:"llmopt_short_conv_prefill_f16"
       ~replacement:"llmopt_short_conv_prefill_init_f16"
  |> replace_first ~needle:"params [[buffer(4)]]"
       ~replacement:"params [[buffer(5)]]"
  |> replace_first ~needle:"conv_state_in [[buffer(5)]]"
       ~replacement:"conv_state_in [[buffer(4)]]"

let short_conv_prefill_init_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_short_conv_prefill_init_f16"
      ~operation:Kernel_abi.Operation.Short_conv_prefill
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16 ]

let attention_source =
  "\nconstant uint ATTENTION_SIMD_WIDTH = 32;\n"
  ^ "constant uint ATTENTION_ROWS_PER_THREADGROUP = 8;\n\n"
  ^ "struct AttentionParams {\n"
  ^ "  uint batches; uint heads; uint query_length; uint key_length;\n"
  ^ "  uint head_dimension; uint mask_batches; uint mask_heads; uint causal;\n"
  ^ "  float scale; uint kv_heads; uint token_first_output;\n"
  ^ "  uint past_length;\n"
  ^ "};\n\n"
  ^ "inline bool llmopt_attention_allowed(\n"
  ^ "    device const uchar* mask, constant AttentionParams& params,\n"
  ^ "    uint batch, uint head, uint query_position, uint key_position) {\n"
  ^ "  if (params.causal != 0 && key_position > (params.past_length + query_position)) {\n"
  ^ "    return false;\n"
  ^ "  }\n"
  ^ "  if (params.mask_batches == 0 || mask == nullptr) {\n"
  ^ "    return true;\n"
  ^ "  }\n"
  ^ "  const uint mask_batch = params.mask_batches == 1 ? 0 : batch;\n"
  ^ "  const uint mask_head = params.mask_heads == 1 ? 0 : head;\n"
  ^ "  const uint mask_index = (((mask_batch * params.mask_heads + mask_head)\n"
  ^ "      * params.query_length + query_position) * params.key_length)\n"
  ^ "      + key_position;\n"
  ^ "  return mask[mask_index] != 0;\n"
  ^ "}\n\n"
  ^ "inline float llmopt_attention_score(\n"
  ^ "    device const half* query, device const half* key,\n"
  ^ "    constant AttentionParams& params, uint batch, uint head,\n"
  ^ "    uint query_position, uint key_position) {\n"
  ^ "  const uint effective_kv_heads = (params.kv_heads > 0) ? params.kv_heads : params.heads;\n"
  ^ "  const uint kv_head = (params.kv_heads > 0 && params.kv_heads < params.heads)\n"
  ^ "      ? (head / (params.heads / params.kv_heads)) : head;\n"
  ^ "  float result = 0.0f;\n"
  ^ "  for (uint dimension = 0; dimension < params.head_dimension; ++dimension) {\n"
  ^ "    const uint query_index = (((batch * params.heads + head)\n"
  ^ "        * params.query_length + query_position) * params.head_dimension)\n"
  ^ "        + dimension;\n"
  ^ "    const uint key_index = (((batch * effective_kv_heads + kv_head)\n"
  ^ "        * params.key_length + key_position) * params.head_dimension)\n"
  ^ "        + dimension;\n"
  ^ "    result += float(query[query_index]) * float(key[key_index]);\n"
  ^ "  }\n"
  ^ "  return result * params.scale;\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_attention_f16(\n"
  ^ "    device const half* query [[buffer(0)]],\n"
  ^ "    device const half* key [[buffer(1)]],\n"
  ^ "    device const half* value [[buffer(2)]],\n"
  ^ "    device const uchar* mask [[buffer(3)]],\n"
  ^ "    device half* output [[buffer(4)]],\n"
  ^ "    constant AttentionParams& params [[buffer(5)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  const uint count = params.batches * params.heads * params.query_length;\n"
  ^ "  if (gid >= count) return;\n"
  ^ "  const uint query_position = gid % params.query_length;\n"
  ^ "  const uint head = (gid / params.query_length) % params.heads;\n"
  ^ "  const uint batch = gid / (params.query_length * params.heads);\n"
  ^ "  float maximum = -INFINITY;\n"
  ^ "  bool found = false;\n"
  ^ "  for (uint key_position = 0; key_position < params.key_length; ++key_position) {\n"
  ^ "    if (llmopt_attention_allowed(mask, params, batch, head, query_position, key_position)) {\n"
  ^ "      found = true;\n"
  ^ "      maximum = max(maximum, llmopt_attention_score(query, key, params,\n"
  ^ "          batch, head, query_position, key_position));\n"
  ^ "    }\n"
  ^ "  }\n"
  ^ "  float denominator = 0.0f;\n"
  ^ "  if (found) {\n"
  ^ "    for (uint key_position = 0; key_position < params.key_length; ++key_position)\n"
  ^ "      if (llmopt_attention_allowed(mask, params, batch, head, query_position, key_position))\n"
  ^ "        denominator += exp(llmopt_attention_score(query, key, params,\n"
  ^ "            batch, head, query_position, key_position) - maximum);\n"
  ^ "  }\n"
  ^ "  const uint effective_kv_heads = (params.kv_heads > 0) ? params.kv_heads : params.heads;\n"
  ^ "  const uint kv_head = (params.kv_heads > 0 && params.kv_heads < params.heads)\n"
  ^ "      ? (head / (params.heads / params.kv_heads)) : head;\n"
  ^ "  for (uint dimension = 0; dimension < params.head_dimension; ++dimension) {\n"
  ^ "    float result = 0.0f;\n"
  ^ "    if (denominator != 0.0f) {\n"
  ^ "      for (uint key_position = 0; key_position < params.key_length; ++key_position) {\n"
  ^ "        if (llmopt_attention_allowed(mask, params, batch, head, query_position, key_position)) {\n"
  ^ "          const float probability = exp(llmopt_attention_score(query, key, params,\n"
  ^ "              batch, head, query_position, key_position) - maximum) / denominator;\n"
  ^ "          const uint value_index = (((batch * effective_kv_heads + kv_head)\n"
  ^ "              * params.key_length + key_position) * params.head_dimension)\n"
  ^ "              + dimension;\n"
  ^ "          result += probability * float(value[value_index]);\n"
  ^ "        }\n"
  ^ "      }\n"
  ^ "    }\n"
  ^ "    const uint out_base = params.token_first_output != 0\n"
  ^ "        ? (((batch * params.query_length + query_position) * params.heads + head) * params.head_dimension)\n"
  ^ "        : (gid * params.head_dimension);\n"
  ^ "    output[out_base + dimension] = half(result);\n"
  ^ "  }\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_attention_f16_simd_h64(\n"
  ^ "    device const half* query [[buffer(0)]],\n"
  ^ "    device const half* key [[buffer(1)]],\n"
  ^ "    device const half* value [[buffer(2)]],\n"
  ^ "    device const uchar* mask [[buffer(3)]],\n"
  ^ "    device half* output [[buffer(4)]],\n"
  ^ "    constant AttentionParams& params [[buffer(5)]],\n"
  ^ "    uint3 threadgroup_position [[threadgroup_position_in_grid]],\n"
  ^ "    uint simdgroup [[simdgroup_index_in_threadgroup]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  const uint row = threadgroup_position.x * ATTENTION_ROWS_PER_THREADGROUP\n"
  ^ "      + simdgroup;\n"
  ^ "  const uint count = params.batches * params.heads * params.query_length;\n"
  ^ "  if (row >= count) return;\n"
  ^ "  const uint query_position = row % params.query_length;\n"
  ^ "  const uint head = (row / params.query_length) % params.heads;\n"
  ^ "  const uint batch = row / (params.query_length * params.heads);\n"
  ^ "  const uint kv_head = (params.kv_heads > 0 && params.kv_heads < params.heads)\n"
  ^ "      ? (head / (params.heads / params.kv_heads)) : head;\n"
  ^ "  const uint query_base = (((batch * params.heads + head)\n"
  ^ "      * params.query_length + query_position) * params.head_dimension);\n"
  ^ "  const float query_low = float(query[query_base + lane]);\n"
  ^ "  const float query_high = float(query[query_base + lane + ATTENTION_SIMD_WIDTH]);\n"
  ^ "  const uint kv_head_base = (batch * params.kv_heads + kv_head) * params.key_length;\n"
  ^ "  float maximum = -INFINITY;\n"
  ^ "  float denominator = 0.0f;\n"
  ^ "  float result_low = 0.0f;\n"
  ^ "  float result_high = 0.0f;\n"
  ^ "  for (uint key_position = 0; key_position < params.key_length; ++key_position) {\n"
  ^ "    if (!llmopt_attention_allowed(mask, params, batch, head,\n"
  ^ "        query_position, key_position)) continue;\n"
  ^ "    const uint key_base = (kv_head_base + key_position) * 64;\n"
  ^ "    const float partial_score = query_low * float(key[key_base + lane])\n"
  ^ "        + query_high * float(key[key_base + lane + ATTENTION_SIMD_WIDTH]);\n"
  ^ "    const float score = simd_sum(partial_score) * params.scale;\n"
  ^ "    const float next_maximum = max(maximum, score);\n"
  ^ "    const float previous_scale = denominator == 0.0f ? 0.0f\n"
  ^ "        : exp(maximum - next_maximum);\n"
  ^ "    const float current_scale = exp(score - next_maximum);\n"
  ^ "    result_low = result_low * previous_scale\n"
  ^ "        + current_scale * float(value[key_base + lane]);\n"
  ^ "    result_high = result_high * previous_scale\n"
  ^ "        + current_scale * float(value[key_base + lane + ATTENTION_SIMD_WIDTH]);\n"
  ^ "    denominator = denominator * previous_scale + current_scale;\n"
  ^ "    maximum = next_maximum;\n"
  ^ "  }\n"
  ^ "  const uint output_base = params.token_first_output != 0\n"
  ^ "      ? (((batch * params.query_length + query_position) * params.heads + head) * params.head_dimension)\n"
  ^ "      : (row * params.head_dimension);\n"
  ^ "  output[output_base + lane] = half(denominator == 0.0f\n"
  ^ "      ? 0.0f : result_low / denominator);\n"
  ^ "  output[output_base + lane + ATTENTION_SIMD_WIDTH] = half(denominator == 0.0f\n"
  ^ "      ? 0.0f : result_high / denominator);\n"
  ^ "}\n"

let attention_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_attention_f16_simd_h64"
      ~operation:Kernel_abi.Operation.Attention
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_attention_f16"
      ~operation:Kernel_abi.Operation.Attention
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16 ]

let embedding_source =
  "\nstruct EmbeddingParams { uint tokens; uint vocabulary; uint width; };\n\n"
  ^ "kernel void llmopt_embedding_f16(\n"
  ^ "    device const long* indices [[buffer(0)]],\n"
  ^ "    device const half* weight [[buffer(1)]],\n"
  ^ "    device half* output [[buffer(2)]],\n"
  ^ "    constant EmbeddingParams& params [[buffer(3)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  const uint count = params.tokens * params.width;\n"
  ^ "  if (gid >= count) return;\n"
  ^ "  const uint token = gid / params.width;\n"
  ^ "  const uint dimension = gid % params.width;\n"
  ^ "  const long index = indices[token];\n"
  ^ "  output[gid] = (index >= 0 && index < long(params.vocabulary))\n"
  ^ "      ? weight[ulong(index) * params.width + dimension]\n"
  ^ "      : half(0.0h);\n"
  ^ "}\n"

let embedding_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_embedding_f16"
      ~operation:Kernel_abi.Operation.Embedding
      ~input_dtype:Ir.Dtype.Int64 ~output_dtype:Ir.Dtype.Float16 ]

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

let cumsum_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_cumsum_bool_i64" ~operation:Kernel_abi.Operation.Cumsum
      ~input_dtype:Ir.Dtype.Bool ~output_dtype:Ir.Dtype.Int64 ]

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
  ^ pointwise_binary_kernel ~name:"llmopt_add_i64" ~input_type:"long"
      ~output_type:"long" ~scalar_field:"i64" ~scalar_cast:"long"
      ~expression:"left_value + right_value"
  ^ pointwise_binary_kernel ~name:"llmopt_mul_f16" ~input_type:"half"
      ~output_type:"half" ~scalar_field:"f32" ~scalar_cast:"half"
      ~expression:"left_value * right_value"
  ^ pointwise_binary_kernel ~name:"llmopt_mul_f32" ~input_type:"float"
      ~output_type:"float" ~scalar_field:"f32" ~scalar_cast:"float"
      ~expression:"left_value * right_value"
  ^ pointwise_binary_kernel ~name:"llmopt_le_i64" ~input_type:"long"
      ~output_type:"uchar" ~scalar_field:"i64" ~scalar_cast:"long"
      ~expression:"left_value <= right_value ? 1 : 0"
  ^ pointwise_unary_kernel ~name:"llmopt_neg_f16" ~input_type:"half"
      ~output_type:"half" ~expression:"-value"
  ^ pointwise_unary_kernel ~name:"llmopt_silu_f16" ~input_type:"half"
      ~output_type:"half"
      ~expression:"half(float(value) / (1.0f + exp(-float(value))))"
  ^ pointwise_unary_kernel ~name:"llmopt_cos_f32" ~input_type:"float"
      ~output_type:"float" ~expression:"cos(value)"
  ^ pointwise_unary_kernel ~name:"llmopt_sin_f32" ~input_type:"float"
      ~output_type:"float" ~expression:"sin(value)"

let pointwise_entries =
  let entry name input_dtype output_dtype =
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1) ~name
      ~operation:Kernel_abi.Operation.Pointwise ~input_dtype ~output_dtype
  in
  [ entry "llmopt_add_f16" Ir.Dtype.Float16 Ir.Dtype.Float16;
    entry "llmopt_add_i64" Ir.Dtype.Int64 Ir.Dtype.Int64;
    entry "llmopt_mul_f16" Ir.Dtype.Float16 Ir.Dtype.Float16;
    entry "llmopt_mul_f32" Ir.Dtype.Float32 Ir.Dtype.Float32;
    entry "llmopt_le_i64" Ir.Dtype.Int64 Ir.Dtype.Bool;
    entry "llmopt_neg_f16" Ir.Dtype.Float16 Ir.Dtype.Float16;
    entry "llmopt_silu_f16" Ir.Dtype.Float16 Ir.Dtype.Float16;
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

let reduction_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_sum_f16" ~operation:Kernel_abi.Operation.Reduction
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16 ]

let update_slice_source =
  "kernel void llmopt_update_slice_f16(\n"
  ^ "    device const half* destination [[buffer(0)]],\n"
  ^ "    device const half* source [[buffer(1)]],\n"
  ^ "    device half* output [[buffer(2)]],\n"
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

let update_slice_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_update_slice_f16"
      ~operation:Kernel_abi.Operation.Update_slice
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16 ]

let has_rms_norm graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with Ir.Op.Rms_norm _ -> true | _ -> false)

let has_rms_rope graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with Ir.Op.Rms_rope _ -> true | _ -> false)

let has_rms_rope_qk graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with Ir.Op.Rms_rope_qk _ -> true | _ -> false)

let has_short_conv graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Primitive (Ir.Primitive.Short_conv _) -> true
         | _ -> false)

let has_short_conv_step graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Short_conv_step _ -> true
         | _ -> false)

let has_short_conv_step_fused graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Short_conv_step_fused _ -> true
         | _ -> false)

let has_short_conv_prefill graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Short_conv_prefill _ -> true
         | _ -> false)

let has_attention graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Primitive (Ir.Primitive.Attention _) -> true
         | _ -> false)

let has_embedding graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Primitive Ir.Primitive.Embedding -> true
         | _ -> false)

let has_primitive graph predicate =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Primitive primitive -> predicate primitive
         | _ -> false)

let has_materialized_movement graph =
  has_primitive graph (function
    | Ir.Primitive.Movement movement ->
        (match movement with
        | Ir.Movement.Transpose _ | Ir.Movement.Expand | Ir.Movement.Index _
        | Ir.Movement.Concat _ | Ir.Movement.Roll _ -> true
        | Ir.Movement.View | Ir.Movement.Reshape | Ir.Movement.Unsqueeze _
        | Ir.Movement.Contiguous -> false)
    | _ -> false)

let has_w4a16_linear graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.W4a16_linear _ -> true
         | _ -> false)

let has_w4a16_qkv_linear graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.W4a16_qkv_linear _ -> true
         | _ -> false)

let has_w4a16_swiglu_ffn graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.W4a16_swiglu_ffn _ -> true
         | _ -> false)

let has_w4a16_lm_head_argmax graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.W4a16_lm_head_argmax _ -> true
         | _ -> false)

let has_f16_linear graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
         | ( Ir.Op.Linear { bias = false; _ },
             [ input; weight ],
             Some output ) ->
             Ir.Value.dtype input = Ir.Dtype.Float16
             && Ir.Value.dtype weight = Ir.Dtype.Float16
             && Ir.Value.dtype output = Ir.Dtype.Float16
         | _ -> false)

let lower_primary graph =
  let kernel =
    Ir.Graph.nodes graph
    |> List.find_map (fun node ->
           match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
           | Ir.Op.Matmul { m; n; k }, [ lhs; rhs ], Some output ->
               Some (Matmul (m, n, k, lhs, rhs, output))
           | Ir.Op.Fused_matmul_bias { m; n; k }, [ lhs; rhs; bias ], Some output ->
               Some (Fused (m, n, k, lhs, rhs, bias, output))
           | Ir.Op.Linear { m; n; k; bias }, inputs, Some output ->
               (match inputs with
               | [ input; weight ]
                 when Ir.Value.dtype input = Ir.Dtype.Float32
                      && Ir.Value.dtype weight = Ir.Dtype.Float32
                      && Ir.Value.dtype output = Ir.Dtype.Float32 ->
                   Some (Linear (m, n, k, bias, input, weight, None, output))
               | [ input; weight; bias_value ]
                 when Ir.Value.dtype input = Ir.Dtype.Float32
                      && Ir.Value.dtype weight = Ir.Dtype.Float32
                      && Ir.Value.dtype bias_value = Ir.Dtype.Float32
                      && Ir.Value.dtype output = Ir.Dtype.Float32 ->
                   Some (Linear (m, n, k, bias, input, weight, Some bias_value, output))
               | _ -> None)
           | _ -> None)
  in
  match kernel with
  | None -> Error "Metal emitter expects one supported kernel node"
  | Some (Matmul (m, n, k, lhs, rhs, output)) ->
      let a_name = input_name graph lhs in
      let b_name = input_name graph rhs in
      let output_name = input_name graph output in
      let source =
        Printf.sprintf
          "#include <metal_stdlib>\n#include <metal_matrix>\nusing namespace metal;\n\n"
          ^ "constant uint TILE = 16;\n\n"
          ^ "struct MatmulParams { uint m; uint n; uint k; };\n\n"
          ^ "kernel void llmopt_matmul(\n"
          ^ "    device const float* a [[buffer(0)]],\n"
          ^ "    device const float* b [[buffer(1)]],\n"
          ^ "    device float* c [[buffer(2)]],\n"
          ^ "    constant MatmulParams& params [[buffer(3)]],\n"
          ^ "    uint2 gid [[thread_position_in_grid]],\n"
          ^ "    uint2 tid [[thread_position_in_threadgroup]]) {\n"
          ^ "  threadgroup float a_tile[16][16];\n"
          ^ "  threadgroup float b_tile[16][16];\n"
          ^ Printf.sprintf "  const uint row = gid.y; const uint col = gid.x;\n"
          ^ Printf.sprintf "  float acc = 0.0f;\n"
          ^ "  for (uint base = 0; base < params.k; base += TILE) {\n"
          ^ "    a_tile[tid.y][tid.x] = (row < params.m && base + tid.x < params.k)\n"
          ^ "        ? a[row * params.k + base + tid.x] : 0.0f;\n"
          ^ "    b_tile[tid.y][tid.x] = (base + tid.y < params.k && col < params.n)\n"
          ^ "        ? b[(base + tid.y) * params.n + col] : 0.0f;\n"
          ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
          ^ "    for (uint i = 0; i < TILE; ++i) acc += a_tile[tid.y][i] * b_tile[i][tid.x];\n"
          ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
          ^ "  }\n"
          ^ "  if (row < params.m && col < params.n) c[row * params.n + col] = acc;\n"
          ^ "}\n"
      in
      ignore (a_name, b_name, output_name);
      Ok
        (Program.make ~source
           ~kernels:
             [ kernel_entry ~name:"llmopt_matmul"
                 ~operation:Kernel_abi.Operation.Matmul
                 ~input_dtype:Ir.Dtype.Float32
                 ~output_dtype:Ir.Dtype.Float32 ])
  | Some (Fused (m, n, k, lhs, rhs, bias, output)) ->
      let a_name = input_name graph lhs in
      let b_name = input_name graph rhs in
      let bias_name = input_name graph bias in
      let output_name = input_name graph output in
      let source =
        Printf.sprintf
          "#include <metal_stdlib>\n#include <metal_matrix>\nusing namespace metal;\n\n"
          ^ "constant uint TILE = 16;\n\n"
          ^ "struct FusedLinearParams { uint m; uint n; uint k; };\n\n"
          ^ "kernel void llmopt_fused_linear(\n"
          ^ "    device const float* a [[buffer(0)]],\n"
          ^ "    device const float* b [[buffer(1)]],\n"
          ^ "    device const float* bias [[buffer(2)]],\n"
          ^ "    device float* c [[buffer(3)]],\n"
          ^ "    constant FusedLinearParams& params [[buffer(4)]],\n"
          ^ "    uint2 gid [[thread_position_in_grid]],\n"
          ^ "    uint2 tid [[thread_position_in_threadgroup]]) {\n"
          ^ "  threadgroup float a_tile[16][16];\n"
          ^ "  threadgroup float b_tile[16][16];\n"
          ^ "  const uint row = gid.y; const uint col = gid.x;\n"
          ^ "  float acc = 0.0f;\n"
          ^ "  for (uint base = 0; base < params.k; base += TILE) {\n"
          ^ "    a_tile[tid.y][tid.x] = (row < params.m && base + tid.x < params.k)\n"
          ^ "        ? a[row * params.k + base + tid.x] : 0.0f;\n"
          ^ "    b_tile[tid.y][tid.x] = (col < params.n && base + tid.y < params.k)\n"
          ^ "        ? b[(base + tid.y) * params.n + col] : 0.0f;\n"
          ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
          ^ "    for (uint inner = 0; inner < TILE; ++inner) acc += a_tile[tid.y][inner] * b_tile[inner][tid.x];\n"
          ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
          ^ "  }\n"
          ^ "  if (row < params.m && col < params.n)\n"
          ^ "    c[row * params.n + col] = acc + bias[col];\n"
          ^ "}\n"
      in
      ignore (a_name, b_name, bias_name, output_name);
      Ok
        (Program.make ~source
           ~kernels:
             [ kernel_entry ~name:"llmopt_fused_linear"
                 ~operation:Kernel_abi.Operation.Fused_linear
                 ~input_dtype:Ir.Dtype.Float32
                 ~output_dtype:Ir.Dtype.Float32 ])
  | Some (Linear (m, n, k, has_bias, input, weight, bias, output))
    when Ir.Value.dtype input = Ir.Dtype.Float32
         && Ir.Value.dtype weight = Ir.Dtype.Float32
         && Ir.Value.dtype output = Ir.Dtype.Float32
         &&
         (match bias with
         | None -> true
         | Some value -> Ir.Value.dtype value = Ir.Dtype.Float32) ->
      let input_symbol = input_name graph input in
      let weight_symbol = input_name graph weight in
      let bias_symbol = Option.map (input_name graph) bias in
      let output_symbol = input_name graph output in
      let bias_argument =
        if has_bias then "    device const float* bias [[buffer(2)]],\n" else ""
      in
      let output_buffer = if has_bias then 3 else 2 in
      let bias_value = if has_bias then " + bias[col]" else "" in
      let source =
        Printf.sprintf
          "#include <metal_stdlib>\n#include <metal_matrix>\nusing namespace metal;\n\n"
          ^ "constant uint TILE = 16;\n\n"
          ^ "struct LinearF32Params { uint m; uint n; uint k; };\n\n"
          ^ "kernel void llmopt_linear(\n"
          ^ "    device const float* input [[buffer(0)]],\n"
          ^ "    device const float* weight [[buffer(1)]],\n"
          ^ bias_argument
          ^ Printf.sprintf "    device float* output [[buffer(%d)]],\n" output_buffer
          ^ Printf.sprintf "    constant LinearF32Params& params [[buffer(%d)]],\n"
              (output_buffer + 1)
          ^ "    uint2 gid [[thread_position_in_grid]],\n"
          ^ "    uint2 tid [[thread_position_in_threadgroup]]) {\n"
          ^ "  threadgroup float input_tile[16][16];\n"
          ^ "  threadgroup float weight_tile[16][16];\n"
          ^ "  const uint row = gid.y; const uint col = gid.x;\n"
          ^ "  float acc = 0.0f;\n"
          ^ "  for (uint base = 0; base < params.k; base += TILE) {\n"
          ^ "    input_tile[tid.y][tid.x] = (row < params.m && base + tid.x < params.k)\n"
          ^ "        ? input[row * params.k + base + tid.x] : 0.0f;\n"
          ^ "    weight_tile[tid.y][tid.x] = (col < params.n && base + tid.y < params.k)\n"
          ^ "        ? weight[col * params.k + base + tid.y] : 0.0f;\n"
          ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
          ^ "    for (uint inner = 0; inner < TILE; ++inner) acc += input_tile[tid.y][inner] * weight_tile[inner][tid.x];\n"
          ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
          ^ "  }\n"
          ^ Printf.sprintf
              "  if (row < params.m && col < params.n) output[row * params.n + col] = acc%s;\n"
              bias_value
          ^ "}\n"
      in
      ignore (input_symbol, weight_symbol, bias_symbol, output_symbol);
      Ok
        (Program.make ~source
           ~kernels:
             [ kernel_entry ~name:"llmopt_linear"
                 ~operation:Kernel_abi.Operation.Linear
                 ~input_dtype:Ir.Dtype.Float32
                 ~output_dtype:Ir.Dtype.Float32 ])
  | Some (Linear _) -> Error "primary Metal linear supports float32 only"

let lower graph =
  let materialized_movement = has_materialized_movement graph in
  let update_slice =
    has_primitive graph (function Ir.Primitive.Update_slice _ -> true | _ -> false)
  in
  let components =
    [ has_w4a16_linear graph, w4a16_source, w4a16_entries;
      ( has_w4a16_qkv_linear graph,
        w4a16_qkv_linear_source,
        w4a16_qkv_linear_entries );
      ( has_w4a16_swiglu_ffn graph,
        w4a16_swiglu_ffn_source,
        w4a16_swiglu_ffn_entries );
      ( has_w4a16_lm_head_argmax graph,
        w4a16_lm_head_argmax_source,
        w4a16_lm_head_argmax_entries );
      has_f16_linear graph, linear_f16_source, linear_f16_entries;
      has_rms_norm graph, rms_norm_source, rms_norm_entries;
      has_rms_rope graph, rms_rope_source, rms_rope_entries;
      has_rms_rope_qk graph, rms_rope_qk_source, rms_rope_qk_entries;
      has_short_conv graph, short_conv_source, short_conv_entries;
      has_short_conv_step graph, short_conv_step_source, short_conv_step_entries;
      ( has_short_conv_step_fused graph,
        short_conv_step_fused_source,
        short_conv_step_fused_entries );
      has_short_conv_prefill graph, short_conv_prefill_source, short_conv_prefill_entries;
      ( has_short_conv_prefill graph,
        short_conv_prefill_init_source,
        short_conv_prefill_init_entries );
      has_attention graph, attention_source, attention_entries;
      has_embedding graph, embedding_source, embedding_entries;
      ( has_primitive graph (function Ir.Primitive.Arange _ -> true | _ -> false),
        arange_source,
        arange_entries );
      ( has_primitive graph (function Ir.Primitive.Diff _ -> true | _ -> false),
        diff_source,
        diff_entries );
      ( has_primitive graph (function Ir.Primitive.Cumsum _ -> true | _ -> false),
        cumsum_source,
        cumsum_entries );
      ( has_primitive graph (function Ir.Primitive.Fill _ -> true | _ -> false),
        fill_source,
        fill_entries );
      ( has_primitive graph (function Ir.Primitive.Gather2 -> true | _ -> false),
        gather2_source,
        gather2_entries );
      ( has_primitive graph (function Ir.Primitive.Cast _ -> true | _ -> false),
        cast_source,
        cast_entries );
      ( has_primitive graph (function Ir.Primitive.Pointwise _ -> true | _ -> false),
        pointwise_source,
        pointwise_entries );
      (materialized_movement || update_slice, movement_parameters_source, []);
      materialized_movement, movement_source, movement_entries;
      ( has_primitive graph (function
          | Ir.Primitive.Reduce
              { Ir.Reduction.operator = Ir.Reduction.Sum; _ } -> true
          | _ -> false),
        reduction_source,
        reduction_entries );
      update_slice, update_slice_source, update_slice_entries ]
  in
  let auxiliary_source, auxiliary_entries =
    List.fold_left
      (fun (source, entries) (enabled, component_source, component_entries) ->
        if enabled then source ^ component_source, entries @ component_entries
        else source, entries)
      ("", []) components
  in
  match lower_primary graph with
  | Ok program when auxiliary_entries <> [] ->
      Ok
        (Program.make
           ~source:(Program.source program ^ auxiliary_source)
           ~kernels:(Program.kernels program @ auxiliary_entries))
  | Ok program -> Ok program
  | Error _ when auxiliary_entries <> [] ->
      Ok
        (Program.make
           ~source:
             ("#include <metal_stdlib>\n#include <metal_matrix>\nusing namespace metal;\n"
             ^ auxiliary_source)
           ~kernels:auxiliary_entries)
  | Error message -> Error message

let emit graph = Result.map Program.source (lower graph)
