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

module Tactic = struct
  type t = { name : string; threadgroup : int * int * int }

  type linear_problem = {
    target : Target_hardware.t;
    m : int;
    n : int;
    k : int;
    input_dtype : Ir.Dtype.t;
    storage : Ir.Linear_storage.layout;
    output_dtype : Ir.Dtype.t;
  }

  type linear_registration = {
    build : linear_problem -> t;
    accepts : linear_problem -> bool;
  }

  type attention_problem = {
    target : Target_hardware.t;
    head_dimension : int;
    input_dtype : Ir.Dtype.t;
    output_dtype : Ir.Dtype.t;
  }

  type attention_registration = {
    build : attention_problem -> t;
    accepts : attention_problem -> bool;
  }

  let name tactic = tactic.name
  let threadgroup tactic = tactic.threadgroup

  let supports_simd target ~threads =
    target.Target_hardware.memory.simd_lanes = 32
    && target.execution.max_threads_per_threadgroup >= threads

  let f16_io input_dtype output_dtype =
    input_dtype = Ir.Dtype.Float16 && output_dtype = Ir.Dtype.Float16

  let block_quantized problem quant =
    problem.storage = Ir.Linear_storage.Block_quantized quant

  let quant_kernel_name = function
    | Ir.Dtype.Q8_0 -> "llmopt_q8_0_linear_f16"
    | Ir.Dtype.Q4_K -> "llmopt_q4_k_linear_f16"
    | Ir.Dtype.Q5_K -> "llmopt_q5_k_linear_f16"
    | Ir.Dtype.Q6_K -> "llmopt_q6_k_linear_f16"
    | Ir.Dtype.Q5_0 -> "llmopt_q5_0_linear_f16"
    | Ir.Dtype.Q4_0 -> "llmopt_q4_0_linear_f16"
    | Ir.Dtype.IQ4_XS -> "llmopt_iq4_xs_linear_f16"

  let block_elements = Ir.Tensor_layout.block_elements

  let quant_registration quant : linear_registration =
    { build = (fun _ -> { name = quant_kernel_name quant; threadgroup = (256, 1, 1) });
      accepts =
        (fun (problem : linear_problem) ->
          problem.m > 0 && problem.n > 0 && problem.k > 0
          && problem.k mod block_elements quant = 0
          && f16_io problem.input_dtype problem.output_dtype
          && block_quantized problem quant
          && supports_simd problem.target ~threads:256) }

  let paired_quant_registration quant : linear_registration =
    { build =
        (fun _ ->
          { name = quant_kernel_name quant ^ "_m2";
            threadgroup = (256, 1, 1) });
      accepts =
        (fun (problem : linear_problem) ->
          problem.m = 2 && problem.n > 0 && problem.k > 0
          && problem.k mod block_elements quant = 0
          && f16_io problem.input_dtype problem.output_dtype
          && block_quantized problem quant
          && supports_simd problem.target ~threads:256) }

  let four_column_quant_registration quant : linear_registration =
    { build =
        (fun _ ->
          { name = quant_kernel_name quant ^ "_m2_x4";
            threadgroup = (64, 1, 1) });
      accepts =
        (fun (problem : linear_problem) ->
          problem.m = 2 && problem.n > 0 && problem.k > 0
          && problem.k mod block_elements quant = 0
          && f16_io problem.input_dtype problem.output_dtype
          && block_quantized problem quant
          && supports_simd problem.target ~threads:64) }

  let full_simd_kquant_registration ~accepts_superblocks quant suffix :
      linear_registration =
    { build =
        (fun _ ->
          { name = quant_kernel_name quant ^ suffix;
            threadgroup = (64, 1, 1) });
      accepts =
        (fun (problem : linear_problem) ->
          problem.m = 2 && problem.n > 0 && problem.k > 0
          && problem.k mod block_elements quant = 0
          && accepts_superblocks (problem.k / block_elements quant)
          && f16_io problem.input_dtype problem.output_dtype
          && block_quantized problem quant
          && supports_simd problem.target ~threads:64) }

  let linear_registry : linear_registration list =
    [ full_simd_kquant_registration ~accepts_superblocks:(fun _ -> true)
        Ir.Dtype.Q4_K
        "_m2_x2_l32";
      full_simd_kquant_registration ~accepts_superblocks:(( = ) 4) Ir.Dtype.Q5_K
        "_m2_x1_l32" ]
    @ List.map four_column_quant_registration
      [ Ir.Dtype.Q4_K; Ir.Dtype.Q5_K; Ir.Dtype.Q6_K ]
    @ List.map paired_quant_registration
      [ Ir.Dtype.Q8_0; Ir.Dtype.Q4_K; Ir.Dtype.Q5_K; Ir.Dtype.Q6_K;
        Ir.Dtype.Q5_0; Ir.Dtype.Q4_0; Ir.Dtype.IQ4_XS ]
    @ List.map quant_registration
        [ Ir.Dtype.Q8_0; Ir.Dtype.Q4_K; Ir.Dtype.Q5_K; Ir.Dtype.Q6_K;
          Ir.Dtype.Q5_0; Ir.Dtype.Q4_0; Ir.Dtype.IQ4_XS ]

  let select_linear ~target ~m ~n ~k ~input_dtype ~storage ~output_dtype =
    let problem = { target; m; n; k; input_dtype; storage; output_dtype } in
    linear_registry
    |> List.find_map (fun (registration : linear_registration) ->
           if registration.accepts problem then Some (registration.build problem)
           else None)

  let attention_registry : attention_registration list =
    [ { build =
          (fun (problem : attention_problem) ->
            { name =
                Printf.sprintf "llmopt_attention_f16_simd_h%d"
                  problem.head_dimension;
              threadgroup = (256, 1, 1) });
        accepts =
          (fun (problem : attention_problem) ->
            problem.head_dimension > 0 && problem.head_dimension mod 32 = 0
            && f16_io problem.input_dtype problem.output_dtype
            && supports_simd problem.target ~threads:256) };
      { build =
          (fun _ ->
            { name = "llmopt_attention_f16"; threadgroup = (64, 1, 1) });
        accepts =
          (fun (problem : attention_problem) ->
            problem.head_dimension > 0
            && f16_io problem.input_dtype problem.output_dtype
            && problem.target.execution.max_threads_per_threadgroup >= 64) } ]

  let select_attention ~target ~head_dimension ~input_dtype ~output_dtype =
    let problem = { target; head_dimension; input_dtype; output_dtype } in
    attention_registry
    |> List.find_map (fun (registration : attention_registration) ->
           if registration.accepts problem then Some (registration.build problem)
           else None)
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
  ^ "    const half2 in_val = *reinterpret_cast<device const half2*>(input + in_idx);\n"
  ^ "    acc += (float(in_val.x) * float(q0) + float(in_val.y) * float(q1)) * s;\n"
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
  ^ "    const half2 in_val = *reinterpret_cast<device const half2*>(input + in_idx);\n"
  ^ "    acc += (float(in_val.x) * float(q0) + float(in_val.y) * float(q1)) * s;\n"
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
  ^ "    const half2 x = *reinterpret_cast<device const half2*>(normalized + in_idx);\n"
  ^ "    const float x0 = float(x.x);\n"
  ^ "    const float x1 = float(x.y);\n"
  ^ "    gate_acc += (x0 * float(g_lo) + x1 * float(g_hi)) * gate_s;\n"
  ^ "    up_acc += (x0 * float(u_lo) + x1 * float(u_hi)) * up_s;\n"
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
  ^ "    const half2 p = *reinterpret_cast<device const half2*>(product + in_idx);\n"
  ^ "    down_acc += (float(p.x) * float(lo) + float(p.y) * float(hi)) * s;\n"
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

let quant_common_source = {|
struct QuantBlock32Params {
    uint total_blocks;
};

struct QuantBlock256Params {
    uint total_superblocks;
};

struct QuantLinearParams {
    uint m;
    uint n;
    uint k;
    uint has_bias;
};

inline half llmopt_gated_product(
    float gate_acc, float up_acc, uint activation) {
  const half gate = half(gate_acc);
  const half up = half(up_acc);
  const float gate_value = float(gate);
  float activated = 0.0f;
  if (activation == 0u) {
    activated = gate_value / (1.0f + exp(-gate_value));
  } else if (activation == 1u) {
    activated = 0.5f * gate_value
      * (1.0f + tanh(clamp(0.7978845608f
        * (gate_value + 0.044715f * gate_value * gate_value * gate_value),
        -10.0f, 10.0f)));
  } else {
    activated = 1.0f / (1.0f + exp(-gate_value));
  }
  return half(half(activated) * up);
}
|}

let q8_0_source = {|
struct block_q8_0 {
    half d;
    int8_t qs[32];
};

kernel void llmopt_dequant_q8_0(
    device const block_q8_0* blocks [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant QuantBlock32Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint block_idx = gid >> 5;
  if (block_idx >= params.total_blocks) return;
  const half d = blocks[block_idx].d;
  const int8_t q = blocks[block_idx].qs[lane];
  output[block_idx * 32 + lane] = half(float(d) * float(q));
}

kernel void llmopt_q8_0_linear_f16(
    device const half* input [[buffer(0)]],
    device const block_q8_0* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5;
  const uint total_elements = params.m * params.n;
  if (simd_idx >= total_elements) return;
  const uint row = simd_idx / params.n;
  const uint col = simd_idx - row * params.n;
  const uint num_blocks = params.k >> 5;
  device const block_q8_0* row_blocks = weight + col * num_blocks;
  device const half* in_ptr = input + row * params.k;
  float acc = 0.0f;
  for (uint b = 0; b < num_blocks; ++b) {
    const half d = row_blocks[b].d;
    const int8_t q = row_blocks[b].qs[lane];
    const half in_val = in_ptr[b * 32 + lane];
    acc += float(in_val) * (float(d) * float(q));
  }
  acc = simd_sum(acc);
  if (lane == 0) {
    if (params.has_bias != 0u) acc += float(bias[col]);
    output[row * params.n + col] = half(acc);
  }
}

kernel void llmopt_q8_0_linear_f16_m2(
    device const half* input [[buffer(0)]],
    device const block_q8_0* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint col = gid >> 5;
  if (col >= params.n) return;
  const uint num_blocks = params.k >> 5;
  device const block_q8_0* row_blocks = weight + col * num_blocks;
  device const half* in0 = input;
  device const half* in1 = input + params.k;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (uint b = 0; b < num_blocks; ++b) {
    const float w = float(row_blocks[b].d) * float(row_blocks[b].qs[lane]);
    const uint index = b * 32 + lane;
    acc0 += float(in0[index]) * w;
    acc1 += float(in1[index]) * w;
  }
  acc0 = simd_sum(acc0);
  acc1 = simd_sum(acc1);
  if (lane == 0) {
    const float bias_value = params.has_bias != 0u ? float(bias[col]) : 0.0f;
    output[col] = half(acc0 + bias_value);
    output[params.n + col] = half(acc1 + bias_value);
  }
}

kernel void llmopt_q8_0_dual_swiglu_f16(
    device const half* normalized [[buffer(0)]],
    device const block_q8_0* gate_weight [[buffer(1)]],
    device const block_q8_0* up_weight [[buffer(2)]],
    device half* product [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5;
  const uint total_elements = params.m * params.n;
  if (simd_idx >= total_elements) return;
  const uint row = simd_idx / params.n;
  const uint col = simd_idx - row * params.n;
  const uint num_blocks = params.k >> 5;
  device const block_q8_0* gate_row = gate_weight + col * num_blocks;
  device const block_q8_0* up_row = up_weight + col * num_blocks;
  device const half* in_ptr = normalized + row * params.k;
  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  for (uint b = 0; b < num_blocks; ++b) {
    const half gd = gate_row[b].d;
    const int8_t gq = gate_row[b].qs[lane];
    const half ud = up_row[b].d;
    const int8_t uq = up_row[b].qs[lane];
    const half in_val = in_ptr[b * 32 + lane];
    gate_acc += float(in_val) * (float(gd) * float(gq));
    up_acc   += float(in_val) * (float(ud) * float(uq));
  }
  gate_acc = simd_sum(gate_acc);
  up_acc = simd_sum(up_acc);
  if (lane == 0) {
    const float silu_gate = gate_acc / (1.0f + exp(-gate_acc));
    product[row * params.n + col] = half(silu_gate * up_acc);
  }
}

kernel void llmopt_q8_0_down_add_f16(
    device const half* product [[buffer(0)]],
    device const block_q8_0* down_weight [[buffer(1)]],
    device const half* residual [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5;
  const uint total_elements = params.m * params.k;
  if (simd_idx >= total_elements) return;
  const uint row = simd_idx / params.k;
  const uint col = simd_idx - row * params.k;
  const uint num_blocks = params.n >> 5;
  device const block_q8_0* down_row = down_weight + col * num_blocks;
  device const half* in_ptr = product + row * params.n;
  float down_acc = 0.0f;
  for (uint b = 0; b < num_blocks; ++b) {
    const half d = down_row[b].d;
    const int8_t q = down_row[b].qs[lane];
    const half in_val = in_ptr[b * 32 + lane];
    down_acc += float(in_val) * (float(d) * float(q));
  }
  down_acc = simd_sum(down_acc);
  if (lane == 0) {
    output[row * params.k + col] = half(down_acc + float(residual[row * params.k + col]));
  }
}
|}

let q5_0_source = {|
struct block_q5_0 {
    half d;
    uint8_t qh[4];
    uint8_t qs[16];
};

kernel void llmopt_dequant_q5_0(
    device const block_q5_0* blocks [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant QuantBlock32Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint block_idx = gid >> 5;
  if (block_idx >= params.total_blocks) return;
  device const block_q5_0& b = blocks[block_idx];
  const half d = b.d;
  const uint8_t high_byte = b.qh[lane >> 3];
  const uint8_t high_bit = (high_byte >> (lane & 7u)) & 1u;
  const uint8_t low_byte = (lane < 16u) ? b.qs[lane] : b.qs[lane - 16u];
  const uint8_t low_nib = (lane < 16u) ? (low_byte & 0x0Fu) : (low_byte >> 4);
  const int q = int(low_nib | (high_bit << 4)) - 16;
  output[block_idx * 32 + lane] = half(float(d) * float(q));
}

kernel void llmopt_q5_0_linear_f16(
    device const half* input [[buffer(0)]],
    device const block_q5_0* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5;
  const uint total_elements = params.m * params.n;
  if (simd_idx >= total_elements) return;
  const uint row = simd_idx / params.n;
  const uint col = simd_idx - row * params.n;
  const uint num_blocks = params.k >> 5;
  device const block_q5_0* row_blocks = weight + col * num_blocks;
  device const half* in_ptr = input + row * params.k;
  float acc = 0.0f;
  for (uint b = 0; b < num_blocks; ++b) {
    device const block_q5_0& blk = row_blocks[b];
    const half d = blk.d;
    const uint8_t high_byte = blk.qh[lane >> 3];
    const uint8_t high_bit = (high_byte >> (lane & 7u)) & 1u;
    const uint8_t low_byte = (lane < 16u) ? blk.qs[lane] : blk.qs[lane - 16u];
    const uint8_t low_nib = (lane < 16u) ? (low_byte & 0x0Fu) : (low_byte >> 4);
    const int q = int(low_nib | (high_bit << 4)) - 16;
    const half in_val = in_ptr[b * 32 + lane];
    acc += float(in_val) * (float(d) * float(q));
  }
  acc = simd_sum(acc);
  if (lane == 0) {
    if (params.has_bias != 0u) acc += float(bias[col]);
    output[row * params.n + col] = half(acc);
  }
}

kernel void llmopt_q5_0_linear_f16_m2(
    device const half* input [[buffer(0)]],
    device const block_q5_0* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint col = gid >> 5;
  if (col >= params.n) return;
  const uint num_blocks = params.k >> 5;
  device const block_q5_0* row_blocks = weight + col * num_blocks;
  device const half* in0 = input;
  device const half* in1 = input + params.k;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (uint b = 0; b < num_blocks; ++b) {
    device const block_q5_0& blk = row_blocks[b];
    const uint8_t high_byte = blk.qh[lane >> 3];
    const uint8_t high_bit = (high_byte >> (lane & 7u)) & 1u;
    const uint8_t low_byte = (lane < 16u) ? blk.qs[lane] : blk.qs[lane - 16u];
    const uint8_t low_nib = (lane < 16u) ? (low_byte & 0x0Fu) : (low_byte >> 4);
    const int q = int(low_nib | (high_bit << 4)) - 16;
    const float w = float(blk.d) * float(q);
    const uint index = b * 32 + lane;
    acc0 += float(in0[index]) * w;
    acc1 += float(in1[index]) * w;
  }
  acc0 = simd_sum(acc0);
  acc1 = simd_sum(acc1);
  if (lane == 0) {
    const float bias_value = params.has_bias != 0u ? float(bias[col]) : 0.0f;
    output[col] = half(acc0 + bias_value);
    output[params.n + col] = half(acc1 + bias_value);
  }
}
|}

let q4_0_source = {|
struct block_q4_0 {
    half d;
    uint8_t qs[16];
};

kernel void llmopt_dequant_q4_0(
    device const block_q4_0* blocks [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant QuantBlock32Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint block_idx = gid >> 5;
  if (block_idx >= params.total_blocks) return;
  device const block_q4_0& b = blocks[block_idx];
  const half d = b.d;
  const uint8_t byte_val = (lane < 16u) ? b.qs[lane] : b.qs[lane - 16u];
  const uint8_t nib = (lane < 16u) ? (byte_val & 0x0Fu) : (byte_val >> 4);
  const int q = int(nib) - 8;
  output[block_idx * 32 + lane] = half(float(d) * float(q));
}

kernel void llmopt_q4_0_linear_f16(
    device const half* input [[buffer(0)]],
    device const block_q4_0* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5;
  const uint total_elements = params.m * params.n;
  if (simd_idx >= total_elements) return;
  const uint row = simd_idx / params.n;
  const uint col = simd_idx - row * params.n;
  const uint num_blocks = params.k >> 5;
  device const block_q4_0* row_blocks = weight + col * num_blocks;
  device const half* in_ptr = input + row * params.k;
  float acc = 0.0f;
  for (uint b = 0; b < num_blocks; ++b) {
    device const block_q4_0& blk = row_blocks[b];
    const half d = blk.d;
    const uint8_t byte_val = (lane < 16u) ? blk.qs[lane] : blk.qs[lane - 16u];
    const uint8_t nib = (lane < 16u) ? (byte_val & 0x0Fu) : (byte_val >> 4);
    const int q = int(nib) - 8;
    const half in_val = in_ptr[b * 32 + lane];
    acc += float(in_val) * (float(d) * float(q));
  }
  acc = simd_sum(acc);
  if (lane == 0) {
    if (params.has_bias != 0u) acc += float(bias[col]);
    output[row * params.n + col] = half(acc);
  }
}

kernel void llmopt_q4_0_linear_f16_m2(
    device const half* input [[buffer(0)]],
    device const block_q4_0* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint col = gid >> 5;
  if (col >= params.n) return;
  const uint num_blocks = params.k >> 5;
  device const block_q4_0* row_blocks = weight + col * num_blocks;
  device const half* in0 = input;
  device const half* in1 = input + params.k;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (uint b = 0; b < num_blocks; ++b) {
    device const block_q4_0& blk = row_blocks[b];
    const uint8_t byte_value = (lane < 16u) ? blk.qs[lane] : blk.qs[lane - 16u];
    const uint8_t nibble = (lane < 16u) ? (byte_value & 0x0Fu) : (byte_value >> 4);
    const float w = float(blk.d) * float(int(nibble) - 8);
    const uint index = b * 32 + lane;
    acc0 += float(in0[index]) * w;
    acc1 += float(in1[index]) * w;
  }
  acc0 = simd_sum(acc0);
  acc1 = simd_sum(acc1);
  if (lane == 0) {
    const float bias_value = params.has_bias != 0u ? float(bias[col]) : 0.0f;
    output[col] = half(acc0 + bias_value);
    output[params.n + col] = half(acc1 + bias_value);
  }
}

kernel void llmopt_q4_0_dual_swiglu_f16(
    device const half* normalized [[buffer(0)]],
    device const block_q4_0* gate_weight [[buffer(1)]],
    device const block_q4_0* up_weight [[buffer(2)]],
    device half* product [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5;
  const uint total_elements = params.m * params.n;
  if (simd_idx >= total_elements) return;
  const uint row = simd_idx / params.n;
  const uint col = simd_idx - row * params.n;
  const uint num_blocks = params.k >> 5;
  device const block_q4_0* gate_row = gate_weight + col * num_blocks;
  device const block_q4_0* up_row = up_weight + col * num_blocks;
  device const half* in_ptr = normalized + row * params.k;
  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  for (uint b = 0; b < num_blocks; ++b) {
    const half gd = gate_row[b].d;
    const uint8_t g_byte = (lane < 16u) ? gate_row[b].qs[lane] : gate_row[b].qs[lane - 16u];
    const uint8_t g_nib = (lane < 16u) ? (g_byte & 0x0Fu) : (g_byte >> 4);
    const int gq = int(g_nib) - 8;

    const half ud = up_row[b].d;
    const uint8_t u_byte = (lane < 16u) ? up_row[b].qs[lane] : up_row[b].qs[lane - 16u];
    const uint8_t u_nib = (lane < 16u) ? (u_byte & 0x0Fu) : (u_byte >> 4);
    const int uq = int(u_nib) - 8;

    const half in_val = in_ptr[b * 32 + lane];
    gate_acc += float(in_val) * (float(gd) * float(gq));
    up_acc   += float(in_val) * (float(ud) * float(uq));
  }
  gate_acc = simd_sum(gate_acc);
  up_acc = simd_sum(up_acc);
  if (lane == 0) {
    const float silu_gate = gate_acc / (1.0f + exp(-gate_acc));
    product[row * params.n + col] = half(silu_gate * up_acc);
  }
}

kernel void llmopt_q4_0_down_add_f16(
    device const half* product [[buffer(0)]],
    device const block_q4_0* down_weight [[buffer(1)]],
    device const half* residual [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5;
  const uint total_elements = params.m * params.k;
  if (simd_idx >= total_elements) return;
  const uint row = simd_idx / params.k;
  const uint col = simd_idx - row * params.k;
  const uint num_blocks = params.n >> 5;
  device const block_q4_0* down_row = down_weight + col * num_blocks;
  device const half* in_ptr = product + row * params.n;
  float down_acc = 0.0f;
  for (uint b = 0; b < num_blocks; ++b) {
    const half d = down_row[b].d;
    const uint8_t byte_val = (lane < 16u) ? down_row[b].qs[lane] : down_row[b].qs[lane - 16u];
    const uint8_t nib = (lane < 16u) ? (byte_val & 0x0Fu) : (byte_val >> 4);
    const int q = int(nib) - 8;
    const half in_val = in_ptr[b * 32 + lane];
    down_acc += float(in_val) * (float(d) * float(q));
  }
  down_acc = simd_sum(down_acc);
  if (lane == 0) {
    output[row * params.k + col] = half(down_acc + float(residual[row * params.k + col]));
  }
}
|}

let block32_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_dequant_q8_0"
      ~operation:Kernel_abi.Operation.Cast
      ~input_dtype:(Ir.Dtype.Quant Q8_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_dequant_q5_0"
      ~operation:Kernel_abi.Operation.Cast
      ~input_dtype:(Ir.Dtype.Quant Q5_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_dequant_q4_0"
      ~operation:Kernel_abi.Operation.Cast
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_0_linear_f16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q8_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_0_linear_f16_m2"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q8_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q5_0_linear_f16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q5_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q5_0_linear_f16_m2"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q5_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_linear_f16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_linear_f16_m2"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_0_dual_swiglu_f16"
      ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
      ~input_dtype:(Ir.Dtype.Quant Q8_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_0_down_add_f16"
      ~operation:Kernel_abi.Operation.Fused_linear
      ~input_dtype:(Ir.Dtype.Quant Q8_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_dual_swiglu_f16"
      ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_down_add_f16"
      ~operation:Kernel_abi.Operation.Fused_linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16 ]

let add_block32_kernels program =
  Program.make
    ~source:
      (Program.source program ^ quant_common_source ^ q8_0_source ^ q5_0_source ^ q4_0_source)
    ~kernels:(Program.kernels program @ block32_entries)

let metal_header =
  "#include <metal_stdlib>\n#include <metal_matrix>\nusing namespace metal;\n"

let emit_dequant_q8_0 () = metal_header ^ quant_common_source ^ q8_0_source
let emit_dequant_q5_0 () = metal_header ^ quant_common_source ^ q5_0_source
let emit_dequant_q4_0 () = metal_header ^ quant_common_source ^ q4_0_source

let q4_k_source = {|
struct block_q4_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

kernel void llmopt_dequant_q4_k(
    device const block_q4_K* blocks [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant QuantBlock256Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint sb_idx = gid >> 5;
  if (sb_idx >= params.total_superblocks) return;
  device const block_q4_K& b = blocks[sb_idx];
  const float d = float(b.d);
  const float dmin = float(b.dmin);
  float dl[8];
  float ml[8];
  for (int j = 0; j < 4; ++j) {
    const uint8_t sc0 = b.scales[j] & 63;
    const uint8_t m0 = b.scales[j + 4] & 63;
    const uint8_t sc1 = (b.scales[j + 8] & 0x0F) | ((b.scales[j] >> 6) << 4);
    const uint8_t m1 = (b.scales[j + 8] >> 4) | ((b.scales[j + 4] >> 6) << 4);
    dl[j] = d * float(sc0);
    ml[j] = dmin * float(m0);
    dl[j + 4] = d * float(sc1);
    ml[j + 4] = dmin * float(m1);
  }
  const uint8_t q0 = b.qs[lane];
  const uint8_t q1 = b.qs[32 + lane];
  const uint8_t q2 = b.qs[64 + lane];
  const uint8_t q3 = b.qs[96 + lane];
  output[sb_idx * 256 + 0 + lane]   = half(dl[0] * float(q0 & 0x0Fu) - ml[0]);
  output[sb_idx * 256 + 32 + lane]  = half(dl[1] * float(q0 >> 4)    - ml[1]);
  output[sb_idx * 256 + 64 + lane]  = half(dl[2] * float(q1 & 0x0Fu) - ml[2]);
  output[sb_idx * 256 + 96 + lane]  = half(dl[3] * float(q1 >> 4)    - ml[3]);
  output[sb_idx * 256 + 128 + lane] = half(dl[4] * float(q2 & 0x0Fu) - ml[4]);
  output[sb_idx * 256 + 160 + lane] = half(dl[5] * float(q2 >> 4)    - ml[5]);
  output[sb_idx * 256 + 192 + lane] = half(dl[6] * float(q3 & 0x0Fu) - ml[6]);
  output[sb_idx * 256 + 224 + lane] = half(dl[7] * float(q3 >> 4)    - ml[7]);
}

kernel void llmopt_q4_k_linear_f16(
    device const half* input [[buffer(0)]],
    device const block_q4_K* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5;
  const uint total_elements = params.m * params.n;
  if (simd_idx >= total_elements) return;
  const uint row = simd_idx / params.n;
  const uint col = simd_idx - row * params.n;
  const uint num_superblocks = params.k >> 8;
  device const block_q4_K* row_blocks = weight + col * num_superblocks;
  device const half* in_ptr = input + row * params.k;
  float acc = 0.0f;
  for (uint sb = 0; sb < num_superblocks; ++sb) {
    device const block_q4_K& b = row_blocks[sb];
    const float d = float(b.d);
    const float dmin = float(b.dmin);
    float dl[8];
    float ml[8];
    for (int j = 0; j < 4; ++j) {
      const uint8_t sc0 = b.scales[j] & 63;
      const uint8_t m0 = b.scales[j + 4] & 63;
      const uint8_t sc1 = (b.scales[j + 8] & 0x0F) | ((b.scales[j] >> 6) << 4);
      const uint8_t m1 = (b.scales[j + 8] >> 4) | ((b.scales[j + 4] >> 6) << 4);
      dl[j] = d * float(sc0);
      ml[j] = dmin * float(m0);
      dl[j + 4] = d * float(sc1);
      ml[j + 4] = dmin * float(m1);
    }
    const uint8_t q0 = b.qs[lane];
    const uint8_t q1 = b.qs[32 + lane];
    const uint8_t q2 = b.qs[64 + lane];
    const uint8_t q3 = b.qs[96 + lane];
    const uint base_idx = sb * 256 + lane;
    acc += float(in_ptr[base_idx + 0])   * (dl[0] * float(q0 & 0x0Fu) - ml[0]);
    acc += float(in_ptr[base_idx + 32])  * (dl[1] * float(q0 >> 4)    - ml[1]);
    acc += float(in_ptr[base_idx + 64])  * (dl[2] * float(q1 & 0x0Fu) - ml[2]);
    acc += float(in_ptr[base_idx + 96])  * (dl[3] * float(q1 >> 4)    - ml[3]);
    acc += float(in_ptr[base_idx + 128]) * (dl[4] * float(q2 & 0x0Fu) - ml[4]);
    acc += float(in_ptr[base_idx + 160]) * (dl[5] * float(q2 >> 4)    - ml[5]);
    acc += float(in_ptr[base_idx + 192]) * (dl[6] * float(q3 & 0x0Fu) - ml[6]);
    acc += float(in_ptr[base_idx + 224]) * (dl[7] * float(q3 >> 4)    - ml[7]);
  }
  acc = simd_sum(acc);
  if (lane == 0) {
    if (params.has_bias != 0u) acc += float(bias[col]);
    output[row * params.n + col] = half(acc);
  }
}

kernel void llmopt_q4_k_linear_f16_m2(
    device const half* input [[buffer(0)]],
    device const block_q4_K* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint col = gid >> 5;
  if (col >= params.n) return;
  const uint num_superblocks = params.k >> 8;
  device const block_q4_K* row_blocks = weight + col * num_superblocks;
  device const half* in0 = input;
  device const half* in1 = input + params.k;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (uint sb = 0; sb < num_superblocks; ++sb) {
    device const block_q4_K& b = row_blocks[sb];
    const float d = float(b.d);
    const float dmin = float(b.dmin);
    float dl[8];
    float ml[8];
    for (int j = 0; j < 4; ++j) {
      const uint8_t sc0 = b.scales[j] & 63;
      const uint8_t m0 = b.scales[j + 4] & 63;
      const uint8_t sc1 = (b.scales[j + 8] & 0x0F) | ((b.scales[j] >> 6) << 4);
      const uint8_t m1 = (b.scales[j + 8] >> 4) | ((b.scales[j + 4] >> 6) << 4);
      dl[j] = d * float(sc0);
      ml[j] = dmin * float(m0);
      dl[j + 4] = d * float(sc1);
      ml[j + 4] = dmin * float(m1);
    }
    const uint8_t q0 = b.qs[lane];
    const uint8_t q1 = b.qs[32 + lane];
    const uint8_t q2 = b.qs[64 + lane];
    const uint8_t q3 = b.qs[96 + lane];
    const uint base_idx = sb * 256 + lane;
    const float w0 = dl[0] * float(q0 & 0x0Fu) - ml[0];
    const float w1 = dl[1] * float(q0 >> 4) - ml[1];
    const float w2 = dl[2] * float(q1 & 0x0Fu) - ml[2];
    const float w3 = dl[3] * float(q1 >> 4) - ml[3];
    const float w4 = dl[4] * float(q2 & 0x0Fu) - ml[4];
    const float w5 = dl[5] * float(q2 >> 4) - ml[5];
    const float w6 = dl[6] * float(q3 & 0x0Fu) - ml[6];
    const float w7 = dl[7] * float(q3 >> 4) - ml[7];
    acc0 += float(in0[base_idx + 0]) * w0;
    acc0 += float(in0[base_idx + 32]) * w1;
    acc0 += float(in0[base_idx + 64]) * w2;
    acc0 += float(in0[base_idx + 96]) * w3;
    acc0 += float(in0[base_idx + 128]) * w4;
    acc0 += float(in0[base_idx + 160]) * w5;
    acc0 += float(in0[base_idx + 192]) * w6;
    acc0 += float(in0[base_idx + 224]) * w7;
    acc1 += float(in1[base_idx + 0]) * w0;
    acc1 += float(in1[base_idx + 32]) * w1;
    acc1 += float(in1[base_idx + 64]) * w2;
    acc1 += float(in1[base_idx + 96]) * w3;
    acc1 += float(in1[base_idx + 128]) * w4;
    acc1 += float(in1[base_idx + 160]) * w5;
    acc1 += float(in1[base_idx + 192]) * w6;
    acc1 += float(in1[base_idx + 224]) * w7;
  }
  acc0 = simd_sum(acc0);
  acc1 = simd_sum(acc1);
  if (lane == 0) {
    const float bias_value = params.has_bias != 0u ? float(bias[col]) : 0.0f;
    output[col] = half(acc0 + bias_value);
    output[params.n + col] = half(acc1 + bias_value);
  }
}

kernel void llmopt_q4_k_linear_f16_m2_x2_l32(
    device const half* input [[buffer(0)]],
    device const block_q4_K* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint columns_per_row = (params.n + 1u) >> 1;
  const uint task = gid >> 5;
  const uint total_tasks = params.m * columns_per_row;
  if (task >= total_tasks) return;

  const uint row = task / columns_per_row;
  const uint first_col = (task - row * columns_per_row) << 1;
  const uint superblock_quarter = lane >> 3;
  const uint local_lane = lane & 7u;
  const uint quant_half = local_lane >> 2;
  const uint quant_offset = local_lane & 3u;
  const uint num_superblocks = params.k >> 8;
  device const half* activation = input + row * params.k
    + (superblock_quarter << 8) + (quant_half << 6)
    + (quant_offset << 3);

  float low_values[16];
  float high_values[16];
  float accumulators[2] = { 0.0f, 0.0f };
  ushort packed_scales[4];
  thread const uchar* scale_bytes =
    reinterpret_cast<thread const uchar*>(packed_scales);

  for (uint sb = superblock_quarter; sb < num_superblocks; sb += 4u) {
    float4 activation_sums = float4(0.0f);
    for (uint index = 0; index < 8u; ++index) {
      low_values[index] = float(activation[index]);
      activation_sums[0] += low_values[index];
      low_values[index + 8u] = float(activation[index + 32u]);
      activation_sums[1] += low_values[index + 8u];
      high_values[index] = float(activation[index + 128u]);
      activation_sums[2] += high_values[index];
      high_values[index + 8u] = float(activation[index + 160u]);
      activation_sums[3] += high_values[index + 8u];
    }

    for (uint column_offset = 0; column_offset < 2u; ++column_offset) {
      const uint col = first_col + column_offset;
      if (col >= params.n) continue;
      device const block_q4_K& block =
        weight[col * num_superblocks + sb];
      device const ushort* scales =
        reinterpret_cast<device const ushort*>(block.scales) + quant_half;
      packed_scales[0] = scales[0] & 0x3f3fu;
      packed_scales[1] = scales[2] & 0x3f3fu;
      packed_scales[2] = ((scales[4] >> 0) & 0x0f0fu)
        | ((scales[0] & 0xc0c0u) >> 2);
      packed_scales[3] = ((scales[4] >> 4) & 0x0f0fu)
        | ((scales[2] & 0xc0c0u) >> 2);

      device const ushort* low_quant =
        reinterpret_cast<device const ushort*>(block.qs)
        + 16u * quant_half + 4u * quant_offset;
      device const ushort* high_quant = low_quant + 32u;
      float4 low_dot = float4(0.0f);
      float4 high_dot = float4(0.0f);
      for (uint index = 0; index < 4u; ++index) {
        low_dot[0] += low_values[2u * index] * float(low_quant[index] & 0x000fu);
        low_dot[1] += low_values[2u * index + 1u]
          * float(low_quant[index] & 0x0f00u);
        low_dot[2] += low_values[2u * index + 8u]
          * float(low_quant[index] & 0x00f0u);
        low_dot[3] += low_values[2u * index + 9u]
          * float(low_quant[index] & 0xf000u);
        high_dot[0] += high_values[2u * index]
          * float(high_quant[index] & 0x000fu);
        high_dot[1] += high_values[2u * index + 1u]
          * float(high_quant[index] & 0x0f00u);
        high_dot[2] += high_values[2u * index + 8u]
          * float(high_quant[index] & 0x00f0u);
        high_dot[3] += high_values[2u * index + 9u]
          * float(high_quant[index] & 0xf000u);
      }

      accumulators[column_offset] += float(block.d)
        * ((low_dot[0] + low_dot[1] / 256.0f) * float(scale_bytes[0])
          + (low_dot[2] + low_dot[3] / 256.0f) * float(scale_bytes[1]) / 16.0f
          + (high_dot[0] + high_dot[1] / 256.0f) * float(scale_bytes[4])
          + (high_dot[2] + high_dot[3] / 256.0f) * float(scale_bytes[5]) / 16.0f)
        - float(block.dmin)
        * (activation_sums[0] * float(scale_bytes[2])
          + activation_sums[1] * float(scale_bytes[3])
          + activation_sums[2] * float(scale_bytes[6])
          + activation_sums[3] * float(scale_bytes[7]));
    }
    activation += 1024u;
  }

  for (uint column_offset = 0; column_offset < 2u; ++column_offset) {
    const uint col = first_col + column_offset;
    const float sum = simd_sum(accumulators[column_offset]);
    if (lane == 0u && col < params.n) {
      const float bias_value = params.has_bias != 0u ? float(bias[col]) : 0.0f;
      output[row * params.n + col] = half(sum + bias_value);
    }
  }
}

kernel void llmopt_q4_k_linear_f16_m2_x4(
    device const half* input [[buffer(0)]],
    device const block_q4_K* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint column_group = gid >> 5;
  const uint column_in_group = lane >> 3;
  const uint local_lane = lane & 7u;
  const uint col = (column_group << 2) + column_in_group;
  if (col >= params.n) return;
  const uint num_superblocks = params.k >> 8;
  device const block_q4_K* row_blocks = weight + col * num_superblocks;
  device const half* in0 = input;
  device const half* in1 = input + params.k;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (uint sb = 0; sb < num_superblocks; ++sb) {
    device const block_q4_K& b = row_blocks[sb];
    const float d = float(b.d);
    const float dmin = float(b.dmin);
    const uint superblock_base = sb << 8;
    for (uint pair = 0; pair < 4; ++pair) {
      const uint low_group = pair << 1;
      const uint high_group = low_group + 1;
      const uint low_scale = low_group < 4
        ? uint(b.scales[low_group] & 63u)
        : uint((b.scales[low_group + 4] & 0x0Fu)
            | ((b.scales[low_group - 4] >> 6) << 4));
      const uint low_min = low_group < 4
        ? uint(b.scales[low_group + 4] & 63u)
        : uint((b.scales[low_group + 4] >> 4)
            | ((b.scales[low_group] >> 6) << 4));
      const uint high_scale = high_group < 4
        ? uint(b.scales[high_group] & 63u)
        : uint((b.scales[high_group + 4] & 0x0Fu)
            | ((b.scales[high_group - 4] >> 6) << 4));
      const uint high_min = high_group < 4
        ? uint(b.scales[high_group + 4] & 63u)
        : uint((b.scales[high_group + 4] >> 4)
            | ((b.scales[high_group] >> 6) << 4));
      const uint packed_offset = (pair << 5) + (local_lane << 2);
      const uchar4 packed =
        *reinterpret_cast<device const uchar4*>(b.qs + packed_offset);
      const float4 qlow = float4(packed & uchar4(15u));
      const float4 qhigh = float4(packed >> 4);
      const uint input_offset = superblock_base + (low_group << 5)
        + (local_lane << 2);
      const float4 x0low = float4(
        *reinterpret_cast<device const half4*>(in0 + input_offset));
      const float4 x0high = float4(
        *reinterpret_cast<device const half4*>(in0 + input_offset + 32));
      const float4 x1low = float4(
        *reinterpret_cast<device const half4*>(in1 + input_offset));
      const float4 x1high = float4(
        *reinterpret_cast<device const half4*>(in1 + input_offset + 32));
      acc0 += d * float(low_scale) * dot(x0low, qlow)
        - dmin * float(low_min) * dot(x0low, float4(1.0f));
      acc0 += d * float(high_scale) * dot(x0high, qhigh)
        - dmin * float(high_min) * dot(x0high, float4(1.0f));
      acc1 += d * float(low_scale) * dot(x1low, qlow)
        - dmin * float(low_min) * dot(x1low, float4(1.0f));
      acc1 += d * float(high_scale) * dot(x1high, qhigh)
        - dmin * float(high_min) * dot(x1high, float4(1.0f));
    }
  }
  acc0 += simd_shuffle_down(acc0, 4);
  acc1 += simd_shuffle_down(acc1, 4);
  acc0 += simd_shuffle_down(acc0, 2);
  acc1 += simd_shuffle_down(acc1, 2);
  acc0 += simd_shuffle_down(acc0, 1);
  acc1 += simd_shuffle_down(acc1, 1);
  if (local_lane == 0) {
    const float bias_value = params.has_bias != 0u ? float(bias[col]) : 0.0f;
    output[col] = half(acc0 + bias_value);
    output[params.n + col] = half(acc1 + bias_value);
  }
}

kernel void llmopt_q4_k_gated_linear_f16(
    device const half* normalized [[buffer(0)]],
    device const block_q4_K* gate_weight [[buffer(1)]],
    device const block_q4_K* up_weight [[buffer(2)]],
    device half* product [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5;
  const uint total_elements = params.m * params.n;
  if (simd_idx >= total_elements) return;
  const uint row = simd_idx / params.n;
  const uint col = simd_idx - row * params.n;
  const uint num_superblocks = params.k >> 8;
  device const block_q4_K* gate_row = gate_weight + col * num_superblocks;
  device const block_q4_K* up_row = up_weight + col * num_superblocks;
  device const half* in_ptr = normalized + row * params.k;
  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  for (uint sb = 0; sb < num_superblocks; ++sb) {
    device const block_q4_K& gb = gate_row[sb];
    device const block_q4_K& ub = up_row[sb];
    const float gd = float(gb.d);
    const float gdmin = float(gb.dmin);
    const float ud = float(ub.d);
    const float udmin = float(ub.dmin);
    float g_dl[8], g_ml[8], u_dl[8], u_ml[8];
    for (int j = 0; j < 4; ++j) {
      const uint8_t g_sc0 = gb.scales[j] & 63;
      const uint8_t g_m0 = gb.scales[j + 4] & 63;
      const uint8_t g_sc1 = (gb.scales[j + 8] & 0x0F) | ((gb.scales[j] >> 6) << 4);
      const uint8_t g_m1 = (gb.scales[j + 8] >> 4) | ((gb.scales[j + 4] >> 6) << 4);
      g_dl[j] = gd * float(g_sc0);
      g_ml[j] = gdmin * float(g_m0);
      g_dl[j + 4] = gd * float(g_sc1);
      g_ml[j + 4] = gdmin * float(g_m1);

      const uint8_t u_sc0 = ub.scales[j] & 63;
      const uint8_t u_m0 = ub.scales[j + 4] & 63;
      const uint8_t u_sc1 = (ub.scales[j + 8] & 0x0F) | ((ub.scales[j] >> 6) << 4);
      const uint8_t u_m1 = (ub.scales[j + 8] >> 4) | ((ub.scales[j + 4] >> 6) << 4);
      u_dl[j] = ud * float(u_sc0);
      u_ml[j] = udmin * float(u_m0);
      u_dl[j + 4] = ud * float(u_sc1);
      u_ml[j + 4] = udmin * float(u_m1);
    }
    const uint8_t g_q0 = gb.qs[lane];
    const uint8_t g_q1 = gb.qs[32 + lane];
    const uint8_t g_q2 = gb.qs[64 + lane];
    const uint8_t g_q3 = gb.qs[96 + lane];

    const uint8_t u_q0 = ub.qs[lane];
    const uint8_t u_q1 = ub.qs[32 + lane];
    const uint8_t u_q2 = ub.qs[64 + lane];
    const uint8_t u_q3 = ub.qs[96 + lane];

    const uint base_idx = sb * 256 + lane;
    const float in0 = float(in_ptr[base_idx + 0]);
    const float in1 = float(in_ptr[base_idx + 32]);
    const float in2 = float(in_ptr[base_idx + 64]);
    const float in3 = float(in_ptr[base_idx + 96]);
    const float in4 = float(in_ptr[base_idx + 128]);
    const float in5 = float(in_ptr[base_idx + 160]);
    const float in6 = float(in_ptr[base_idx + 192]);
    const float in7 = float(in_ptr[base_idx + 224]);

    gate_acc += in0 * (g_dl[0] * float(g_q0 & 0x0Fu) - g_ml[0])
              + in1 * (g_dl[1] * float(g_q0 >> 4)    - g_ml[1])
              + in2 * (g_dl[2] * float(g_q1 & 0x0Fu) - g_ml[2])
              + in3 * (g_dl[3] * float(g_q1 >> 4)    - g_ml[3])
              + in4 * (g_dl[4] * float(g_q2 & 0x0Fu) - g_ml[4])
              + in5 * (g_dl[5] * float(g_q2 >> 4)    - g_ml[5])
              + in6 * (g_dl[6] * float(g_q3 & 0x0Fu) - g_ml[6])
              + in7 * (g_dl[7] * float(g_q3 >> 4)    - g_ml[7]);

    up_acc   += in0 * (u_dl[0] * float(u_q0 & 0x0Fu) - u_ml[0])
              + in1 * (u_dl[1] * float(u_q0 >> 4)    - u_ml[1])
              + in2 * (u_dl[2] * float(u_q1 & 0x0Fu) - u_ml[2])
              + in3 * (u_dl[3] * float(u_q1 >> 4)    - u_ml[3])
              + in4 * (u_dl[4] * float(u_q2 & 0x0Fu) - u_ml[4])
              + in5 * (u_dl[5] * float(u_q2 >> 4)    - u_ml[5])
              + in6 * (u_dl[6] * float(u_q3 & 0x0Fu) - u_ml[6])
              + in7 * (u_dl[7] * float(u_q3 >> 4)    - u_ml[7]);
  }
  gate_acc = simd_sum(gate_acc);
  up_acc = simd_sum(up_acc);
  if (lane == 0) {
    product[row * params.n + col] =
      llmopt_gated_product(gate_acc, up_acc, params.has_bias);
  }
}

kernel void llmopt_q4_k_gated_linear_f16_m2_x1_l32(
    device const half* input [[buffer(0)]],
    device const block_q4_K* gate_weight [[buffer(1)]],
    device const block_q4_K* up_weight [[buffer(2)]],
    device half* product [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint task = gid >> 5;
  const uint total_tasks = params.m * params.n;
  if (task >= total_tasks) return;

  const uint row = task / params.n;
  const uint col = task - row * params.n;
  const uint superblock_quarter = lane >> 3;
  const uint local_lane = lane & 7u;
  const uint quant_half = local_lane >> 2;
  const uint quant_offset = local_lane & 3u;
  const uint num_superblocks = params.k >> 8;
  device const half* activation = input + row * params.k
    + (superblock_quarter << 8) + (quant_half << 6)
    + (quant_offset << 3);

  float low_values[16];
  float high_values[16];
  float accumulators[2] = { 0.0f, 0.0f };
  ushort packed_scales[4];
  thread const uchar* scale_bytes =
    reinterpret_cast<thread const uchar*>(packed_scales);

  for (uint sb = superblock_quarter; sb < num_superblocks; sb += 4u) {
    float4 activation_sums = float4(0.0f);
    for (uint index = 0; index < 8u; ++index) {
      low_values[index] = float(activation[index]);
      activation_sums[0] += low_values[index];
      low_values[index + 8u] = float(activation[index + 32u]);
      activation_sums[1] += low_values[index + 8u];
      high_values[index] = float(activation[index + 128u]);
      activation_sums[2] += high_values[index];
      high_values[index + 8u] = float(activation[index + 160u]);
      activation_sums[3] += high_values[index + 8u];
    }

    for (uint projection = 0; projection < 2u; ++projection) {
      device const block_q4_K* projection_weight =
        projection == 0u ? gate_weight : up_weight;
      device const block_q4_K& block =
        projection_weight[col * num_superblocks + sb];
      device const ushort* scales =
        reinterpret_cast<device const ushort*>(block.scales) + quant_half;
      packed_scales[0] = scales[0] & 0x3f3fu;
      packed_scales[1] = scales[2] & 0x3f3fu;
      packed_scales[2] = ((scales[4] >> 0) & 0x0f0fu)
        | ((scales[0] & 0xc0c0u) >> 2);
      packed_scales[3] = ((scales[4] >> 4) & 0x0f0fu)
        | ((scales[2] & 0xc0c0u) >> 2);

      device const ushort* low_quant =
        reinterpret_cast<device const ushort*>(block.qs)
        + 16u * quant_half + 4u * quant_offset;
      device const ushort* high_quant = low_quant + 32u;
      float4 low_dot = float4(0.0f);
      float4 high_dot = float4(0.0f);
      for (uint index = 0; index < 4u; ++index) {
        low_dot[0] += low_values[2u * index]
          * float(low_quant[index] & 0x000fu);
        low_dot[1] += low_values[2u * index + 1u]
          * float(low_quant[index] & 0x0f00u);
        low_dot[2] += low_values[2u * index + 8u]
          * float(low_quant[index] & 0x00f0u);
        low_dot[3] += low_values[2u * index + 9u]
          * float(low_quant[index] & 0xf000u);
        high_dot[0] += high_values[2u * index]
          * float(high_quant[index] & 0x000fu);
        high_dot[1] += high_values[2u * index + 1u]
          * float(high_quant[index] & 0x0f00u);
        high_dot[2] += high_values[2u * index + 8u]
          * float(high_quant[index] & 0x00f0u);
        high_dot[3] += high_values[2u * index + 9u]
          * float(high_quant[index] & 0xf000u);
      }

      accumulators[projection] += float(block.d)
        * ((low_dot[0] + low_dot[1] / 256.0f) * float(scale_bytes[0])
          + (low_dot[2] + low_dot[3] / 256.0f) * float(scale_bytes[1]) / 16.0f
          + (high_dot[0] + high_dot[1] / 256.0f) * float(scale_bytes[4])
          + (high_dot[2] + high_dot[3] / 256.0f) * float(scale_bytes[5]) / 16.0f)
        - float(block.dmin)
        * (activation_sums[0] * float(scale_bytes[2])
          + activation_sums[1] * float(scale_bytes[3])
          + activation_sums[2] * float(scale_bytes[6])
          + activation_sums[3] * float(scale_bytes[7]));
    }
    activation += 1024u;
  }

  const float gate_sum = simd_sum(accumulators[0]);
  const float up_sum = simd_sum(accumulators[1]);
  if (lane == 0u) {
    product[row * params.n + col] =
      llmopt_gated_product(gate_sum, up_sum, params.has_bias);
  }
}

kernel void llmopt_q4_k_down_add_f16(
    device const half* product [[buffer(0)]],
    device const block_q4_K* down_weight [[buffer(1)]],
    device const half* residual [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5;
  const uint total_elements = params.m * params.k;
  if (simd_idx >= total_elements) return;
  const uint row = simd_idx / params.k;
  const uint col = simd_idx - row * params.k;
  const uint num_superblocks = params.n >> 8;
  device const block_q4_K* down_row = down_weight + col * num_superblocks;
  device const half* in_ptr = product + row * params.n;
  float down_acc = 0.0f;
  for (uint sb = 0; sb < num_superblocks; ++sb) {
    device const block_q4_K& b = down_row[sb];
    const float d = float(b.d);
    const float dmin = float(b.dmin);
    float dl[8], ml[8];
    for (int j = 0; j < 4; ++j) {
      const uint8_t sc0 = b.scales[j] & 63;
      const uint8_t m0 = b.scales[j + 4] & 63;
      const uint8_t sc1 = (b.scales[j + 8] & 0x0F) | ((b.scales[j] >> 6) << 4);
      const uint8_t m1 = (b.scales[j + 8] >> 4) | ((b.scales[j + 4] >> 6) << 4);
      dl[j] = d * float(sc0);
      ml[j] = dmin * float(m0);
      dl[j + 4] = d * float(sc1);
      ml[j + 4] = dmin * float(m1);
    }
    const uint8_t q0 = b.qs[lane];
    const uint8_t q1 = b.qs[32 + lane];
    const uint8_t q2 = b.qs[64 + lane];
    const uint8_t q3 = b.qs[96 + lane];
    const uint base_idx = sb * 256 + lane;
    down_acc += float(in_ptr[base_idx + 0])   * (dl[0] * float(q0 & 0x0Fu) - ml[0])
              + float(in_ptr[base_idx + 32])  * (dl[1] * float(q0 >> 4)    - ml[1])
              + float(in_ptr[base_idx + 64])  * (dl[2] * float(q1 & 0x0Fu) - ml[2])
              + float(in_ptr[base_idx + 96])  * (dl[3] * float(q1 >> 4)    - ml[3])
              + float(in_ptr[base_idx + 128]) * (dl[4] * float(q2 & 0x0Fu) - ml[4])
              + float(in_ptr[base_idx + 160]) * (dl[5] * float(q2 >> 4)    - ml[5])
              + float(in_ptr[base_idx + 192]) * (dl[6] * float(q3 & 0x0Fu) - ml[6])
              + float(in_ptr[base_idx + 224]) * (dl[7] * float(q3 >> 4)    - ml[7]);
  }
  down_acc = simd_sum(down_acc);
  if (lane == 0) {
    output[row * params.k + col] = half(down_acc + float(residual[row * params.k + col]));
  }
}
|}

let q5_k_source = {|
struct block_q5_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

kernel void llmopt_dequant_q5_k(
    device const block_q5_K* blocks [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant QuantBlock256Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint sb_idx = gid >> 5;
  if (sb_idx >= params.total_superblocks) return;
  device const block_q5_K& b = blocks[sb_idx];
  const float d = float(b.d);
  const float dmin = float(b.dmin);
  float dl[8];
  float ml[8];
  for (int j = 0; j < 4; ++j) {
    const uint8_t sc0 = b.scales[j] & 63;
    const uint8_t m0 = b.scales[j + 4] & 63;
    const uint8_t sc1 = (b.scales[j + 8] & 0x0F) | ((b.scales[j] >> 6) << 4);
    const uint8_t m1 = (b.scales[j + 8] >> 4) | ((b.scales[j + 4] >> 6) << 4);
    dl[j] = d * float(sc0);
    ml[j] = dmin * float(m0);
    dl[j + 4] = d * float(sc1);
    ml[j + 4] = dmin * float(m1);
  }
  const uint8_t qh = b.qh[lane];
  const uint8_t h0 = (qh >> 0) & 1u;
  const uint8_t h1 = (qh >> 1) & 1u;
  const uint8_t h2 = (qh >> 2) & 1u;
  const uint8_t h3 = (qh >> 3) & 1u;
  const uint8_t h4 = (qh >> 4) & 1u;
  const uint8_t h5 = (qh >> 5) & 1u;
  const uint8_t h6 = (qh >> 6) & 1u;
  const uint8_t h7 = (qh >> 7) & 1u;
  const uint8_t q0 = b.qs[lane];
  const uint8_t q1 = b.qs[32 + lane];
  const uint8_t q2 = b.qs[64 + lane];
  const uint8_t q3 = b.qs[96 + lane];
  output[sb_idx * 256 + 0 + lane]   = half(dl[0] * float((q0 & 0x0Fu) | (h0 << 4)) - ml[0]);
  output[sb_idx * 256 + 32 + lane]  = half(dl[1] * float((q0 >> 4)    | (h1 << 4)) - ml[1]);
  output[sb_idx * 256 + 64 + lane]  = half(dl[2] * float((q1 & 0x0Fu) | (h2 << 4)) - ml[2]);
  output[sb_idx * 256 + 96 + lane]  = half(dl[3] * float((q1 >> 4)    | (h3 << 4)) - ml[3]);
  output[sb_idx * 256 + 128 + lane] = half(dl[4] * float((q2 & 0x0Fu) | (h4 << 4)) - ml[4]);
  output[sb_idx * 256 + 160 + lane] = half(dl[5] * float((q2 >> 4)    | (h5 << 4)) - ml[5]);
  output[sb_idx * 256 + 192 + lane] = half(dl[6] * float((q3 & 0x0Fu) | (h6 << 4)) - ml[6]);
  output[sb_idx * 256 + 224 + lane] = half(dl[7] * float((q3 >> 4)    | (h7 << 4)) - ml[7]);
}

kernel void llmopt_q5_k_linear_f16(
    device const half* input [[buffer(0)]],
    device const block_q5_K* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5;
  const uint total_elements = params.m * params.n;
  if (simd_idx >= total_elements) return;
  const uint row = simd_idx / params.n;
  const uint col = simd_idx - row * params.n;
  const uint num_superblocks = params.k >> 8;
  device const block_q5_K* row_blocks = weight + col * num_superblocks;
  device const half* in_ptr = input + row * params.k;
  float acc = 0.0f;
  for (uint sb = 0; sb < num_superblocks; ++sb) {
    device const block_q5_K& b = row_blocks[sb];
    const float d = float(b.d);
    const float dmin = float(b.dmin);
    float dl[8];
    float ml[8];
    for (int j = 0; j < 4; ++j) {
      const uint8_t sc0 = b.scales[j] & 63;
      const uint8_t m0 = b.scales[j + 4] & 63;
      const uint8_t sc1 = (b.scales[j + 8] & 0x0F) | ((b.scales[j] >> 6) << 4);
      const uint8_t m1 = (b.scales[j + 8] >> 4) | ((b.scales[j + 4] >> 6) << 4);
      dl[j] = d * float(sc0);
      ml[j] = dmin * float(m0);
      dl[j + 4] = d * float(sc1);
      ml[j + 4] = dmin * float(m1);
    }
    const uint8_t qh = b.qh[lane];
    const uint8_t h0 = (qh >> 0) & 1u;
    const uint8_t h1 = (qh >> 1) & 1u;
    const uint8_t h2 = (qh >> 2) & 1u;
    const uint8_t h3 = (qh >> 3) & 1u;
    const uint8_t h4 = (qh >> 4) & 1u;
    const uint8_t h5 = (qh >> 5) & 1u;
    const uint8_t h6 = (qh >> 6) & 1u;
    const uint8_t h7 = (qh >> 7) & 1u;
    const uint8_t q0 = b.qs[lane];
    const uint8_t q1 = b.qs[32 + lane];
    const uint8_t q2 = b.qs[64 + lane];
    const uint8_t q3 = b.qs[96 + lane];
    const uint base_idx = sb * 256 + lane;
    acc += float(in_ptr[base_idx + 0])   * (dl[0] * float((q0 & 0x0Fu) | (h0 << 4)) - ml[0]);
    acc += float(in_ptr[base_idx + 32])  * (dl[1] * float((q0 >> 4)    | (h1 << 4)) - ml[1]);
    acc += float(in_ptr[base_idx + 64])  * (dl[2] * float((q1 & 0x0Fu) | (h2 << 4)) - ml[2]);
    acc += float(in_ptr[base_idx + 96])  * (dl[3] * float((q1 >> 4)    | (h3 << 4)) - ml[3]);
    acc += float(in_ptr[base_idx + 128]) * (dl[4] * float((q2 & 0x0Fu) | (h4 << 4)) - ml[4]);
    acc += float(in_ptr[base_idx + 160]) * (dl[5] * float((q2 >> 4)    | (h5 << 4)) - ml[5]);
    acc += float(in_ptr[base_idx + 192]) * (dl[6] * float((q3 & 0x0Fu) | (h6 << 4)) - ml[6]);
    acc += float(in_ptr[base_idx + 224]) * (dl[7] * float((q3 >> 4)    | (h7 << 4)) - ml[7]);
  }
  acc = simd_sum(acc);
  if (lane == 0) {
    if (params.has_bias != 0u) acc += float(bias[col]);
    output[row * params.n + col] = half(acc);
  }
}

kernel void llmopt_q5_k_linear_f16_m2(
    device const half* input [[buffer(0)]],
    device const block_q5_K* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint col = gid >> 5;
  if (col >= params.n) return;
  const uint num_superblocks = params.k >> 8;
  device const block_q5_K* row_blocks = weight + col * num_superblocks;
  device const half* in0 = input;
  device const half* in1 = input + params.k;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (uint sb = 0; sb < num_superblocks; ++sb) {
    device const block_q5_K& b = row_blocks[sb];
    const float d = float(b.d);
    const float dmin = float(b.dmin);
    float dl[8];
    float ml[8];
    for (int j = 0; j < 4; ++j) {
      const uint8_t sc0 = b.scales[j] & 63;
      const uint8_t m0 = b.scales[j + 4] & 63;
      const uint8_t sc1 = (b.scales[j + 8] & 0x0F) | ((b.scales[j] >> 6) << 4);
      const uint8_t m1 = (b.scales[j + 8] >> 4) | ((b.scales[j + 4] >> 6) << 4);
      dl[j] = d * float(sc0);
      ml[j] = dmin * float(m0);
      dl[j + 4] = d * float(sc1);
      ml[j + 4] = dmin * float(m1);
    }
    const uint8_t qh = b.qh[lane];
    const uint8_t q0 = b.qs[lane];
    const uint8_t q1 = b.qs[32 + lane];
    const uint8_t q2 = b.qs[64 + lane];
    const uint8_t q3 = b.qs[96 + lane];
    const uint base_idx = sb * 256 + lane;
    const float w0 = dl[0] * float((q0 & 0x0Fu) | (((qh >> 0) & 1u) << 4)) - ml[0];
    const float w1 = dl[1] * float((q0 >> 4) | (((qh >> 1) & 1u) << 4)) - ml[1];
    const float w2 = dl[2] * float((q1 & 0x0Fu) | (((qh >> 2) & 1u) << 4)) - ml[2];
    const float w3 = dl[3] * float((q1 >> 4) | (((qh >> 3) & 1u) << 4)) - ml[3];
    const float w4 = dl[4] * float((q2 & 0x0Fu) | (((qh >> 4) & 1u) << 4)) - ml[4];
    const float w5 = dl[5] * float((q2 >> 4) | (((qh >> 5) & 1u) << 4)) - ml[5];
    const float w6 = dl[6] * float((q3 & 0x0Fu) | (((qh >> 6) & 1u) << 4)) - ml[6];
    const float w7 = dl[7] * float((q3 >> 4) | (((qh >> 7) & 1u) << 4)) - ml[7];
    acc0 += float(in0[base_idx + 0]) * w0;
    acc0 += float(in0[base_idx + 32]) * w1;
    acc0 += float(in0[base_idx + 64]) * w2;
    acc0 += float(in0[base_idx + 96]) * w3;
    acc0 += float(in0[base_idx + 128]) * w4;
    acc0 += float(in0[base_idx + 160]) * w5;
    acc0 += float(in0[base_idx + 192]) * w6;
    acc0 += float(in0[base_idx + 224]) * w7;
    acc1 += float(in1[base_idx + 0]) * w0;
    acc1 += float(in1[base_idx + 32]) * w1;
    acc1 += float(in1[base_idx + 64]) * w2;
    acc1 += float(in1[base_idx + 96]) * w3;
    acc1 += float(in1[base_idx + 128]) * w4;
    acc1 += float(in1[base_idx + 160]) * w5;
    acc1 += float(in1[base_idx + 192]) * w6;
    acc1 += float(in1[base_idx + 224]) * w7;
  }
  acc0 = simd_sum(acc0);
  acc1 = simd_sum(acc1);
  if (lane == 0) {
    const float bias_value = params.has_bias != 0u ? float(bias[col]) : 0.0f;
    output[col] = half(acc0 + bias_value);
    output[params.n + col] = half(acc1 + bias_value);
  }
}

kernel void llmopt_q5_k_linear_f16_m2_x1_l32(
    device const half* input [[buffer(0)]],
    device const block_q5_K* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint task = gid >> 5;
  const uint total_tasks = params.m * params.n;
  if (task >= total_tasks) return;

  const uint row = task / params.n;
  const uint col = task - row * params.n;
  const uint lane_cluster = lane >> 2;
  const uint superblock_offset = lane & 3u;
  const uint quant_half = lane_cluster >> 2;
  const uint quant_chunk = lane_cluster & 3u;
  const uint element_offset = quant_chunk << 3;
  const uint quant_offset = (quant_half << 5) + element_offset;
  const uint activation_offset = (quant_half << 6) + element_offset;
  const uchar low_high_mask = uchar(1u << (quant_half << 1));
  const uchar high_high_mask = low_high_mask << 1;
  const uchar upper_low_high_mask = low_high_mask << 4;
  const uchar upper_high_high_mask = high_high_mask << 4;
  const uint num_superblocks = params.k >> 8;
  device const half* activation = input + row * params.k
    + (superblock_offset << 8) + activation_offset;

  float low_values[16];
  float high_values[16];
  float accumulator = 0.0f;
  ushort packed_scales[4];
  thread const uchar* scale_bytes =
    reinterpret_cast<thread const uchar*>(packed_scales);

  for (uint sb = superblock_offset; sb < num_superblocks; sb += 4u) {
    float4 activation_sums = float4(0.0f);
    for (uint index = 0; index < 8u; ++index) {
      low_values[index] = float(activation[index]);
      activation_sums[0] += low_values[index];
      low_values[index + 8u] = float(activation[index + 32u]);
      activation_sums[1] += low_values[index + 8u];
      high_values[index] = float(activation[index + 128u]);
      activation_sums[2] += high_values[index];
      high_values[index + 8u] = float(activation[index + 160u]);
      activation_sums[3] += high_values[index + 8u];
    }

    device const block_q5_K& block = weight[col * num_superblocks + sb];
    device const ushort* scales =
      reinterpret_cast<device const ushort*>(block.scales) + quant_half;
    packed_scales[0] = scales[0] & 0x3f3fu;
    packed_scales[1] = scales[2] & 0x3f3fu;
    packed_scales[2] = ((scales[4] >> 0) & 0x0f0fu)
      | ((scales[0] & 0xc0c0u) >> 2);
    packed_scales[3] = ((scales[4] >> 4) & 0x0f0fu)
      | ((scales[2] & 0xc0c0u) >> 2);

    device const uchar* low_quant = block.qs + quant_offset;
    device const uchar* high_quant = low_quant + 64u;
    device const uchar* high_bits = block.qh + element_offset;
    float4 quant_dots = float4(0.0f);
    float4 high_dots = float4(0.0f);
    for (uint index = 0; index < 8u; ++index) {
      const uchar high = high_bits[index];
      quant_dots[0] += low_values[index]
        * float(low_quant[index] & 0x0fu);
      quant_dots[1] += low_values[index + 8u]
        * float(low_quant[index] & 0xf0u);
      quant_dots[2] += high_values[index]
        * float(high_quant[index] & 0x0fu);
      quant_dots[3] += high_values[index + 8u]
        * float(high_quant[index] & 0xf0u);
      high_dots[0] += (high & low_high_mask) != 0u ? low_values[index] : 0.0f;
      high_dots[1] += (high & high_high_mask) != 0u
        ? low_values[index + 8u] : 0.0f;
      high_dots[2] += (high & upper_low_high_mask) != 0u
        ? high_values[index] : 0.0f;
      high_dots[3] += (high & upper_high_high_mask) != 0u
        ? high_values[index + 8u] : 0.0f;
    }

    accumulator += float(block.d)
      * (float(scale_bytes[0]) * (quant_dots[0] + 16.0f * high_dots[0])
        + float(scale_bytes[1]) * (quant_dots[1] / 16.0f
          + 16.0f * high_dots[1])
        + float(scale_bytes[4]) * (quant_dots[2] + 16.0f * high_dots[2])
        + float(scale_bytes[5]) * (quant_dots[3] / 16.0f
          + 16.0f * high_dots[3]))
      - float(block.dmin)
      * (activation_sums[0] * float(scale_bytes[2])
        + activation_sums[1] * float(scale_bytes[3])
        + activation_sums[2] * float(scale_bytes[6])
        + activation_sums[3] * float(scale_bytes[7]));
    activation += 1024u;
  }

  const float sum = simd_sum(accumulator);
  if (lane == 0u) {
    const float bias_value = params.has_bias != 0u ? float(bias[col]) : 0.0f;
    output[row * params.n + col] = half(sum + bias_value);
  }
}

kernel void llmopt_q5_k_gated_linear_f16_m2_x1_l32(
    device const half* input [[buffer(0)]],
    device const block_q5_K* gate_weight [[buffer(1)]],
    device const block_q5_K* up_weight [[buffer(2)]],
    device half* product [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint task = gid >> 5;
  const uint total_tasks = params.m * params.n;
  if (task >= total_tasks) return;

  const uint row = task / params.n;
  const uint col = task - row * params.n;
  const uint lane_cluster = lane >> 2;
  const uint superblock_offset = lane & 3u;
  const uint quant_half = lane_cluster >> 2;
  const uint quant_chunk = lane_cluster & 3u;
  const uint element_offset = quant_chunk << 3;
  const uint quant_offset = (quant_half << 5) + element_offset;
  const uint activation_offset = (quant_half << 6) + element_offset;
  const uchar low_high_mask = uchar(1u << (quant_half << 1));
  const uchar high_high_mask = low_high_mask << 1;
  const uchar upper_low_high_mask = low_high_mask << 4;
  const uchar upper_high_high_mask = high_high_mask << 4;
  const uint num_superblocks = params.k >> 8;
  device const half* activation = input + row * params.k
    + (superblock_offset << 8) + activation_offset;

  float low_values[16];
  float high_values[16];
  float accumulators[2] = { 0.0f, 0.0f };
  ushort packed_scales[4];
  thread const uchar* scale_bytes =
    reinterpret_cast<thread const uchar*>(packed_scales);

  for (uint sb = superblock_offset; sb < num_superblocks; sb += 4u) {
    float4 activation_sums = float4(0.0f);
    for (uint index = 0; index < 8u; ++index) {
      low_values[index] = float(activation[index]);
      activation_sums[0] += low_values[index];
      low_values[index + 8u] = float(activation[index + 32u]);
      activation_sums[1] += low_values[index + 8u];
      high_values[index] = float(activation[index + 128u]);
      activation_sums[2] += high_values[index];
      high_values[index + 8u] = float(activation[index + 160u]);
      activation_sums[3] += high_values[index + 8u];
    }

    for (uint projection = 0; projection < 2u; ++projection) {
      device const block_q5_K* projection_weight =
        projection == 0u ? gate_weight : up_weight;
      device const block_q5_K& block =
        projection_weight[col * num_superblocks + sb];
      device const ushort* scales =
        reinterpret_cast<device const ushort*>(block.scales) + quant_half;
      packed_scales[0] = scales[0] & 0x3f3fu;
      packed_scales[1] = scales[2] & 0x3f3fu;
      packed_scales[2] = ((scales[4] >> 0) & 0x0f0fu)
        | ((scales[0] & 0xc0c0u) >> 2);
      packed_scales[3] = ((scales[4] >> 4) & 0x0f0fu)
        | ((scales[2] & 0xc0c0u) >> 2);

      device const uchar* low_quant = block.qs + quant_offset;
      device const uchar* high_quant = low_quant + 64u;
      device const uchar* high_bits = block.qh + element_offset;
      float4 quant_dots = float4(0.0f);
      float4 high_dots = float4(0.0f);
      for (uint index = 0; index < 8u; ++index) {
        const uchar high = high_bits[index];
        quant_dots[0] += low_values[index]
          * float(low_quant[index] & 0x0fu);
        quant_dots[1] += low_values[index + 8u]
          * float(low_quant[index] & 0xf0u);
        quant_dots[2] += high_values[index]
          * float(high_quant[index] & 0x0fu);
        quant_dots[3] += high_values[index + 8u]
          * float(high_quant[index] & 0xf0u);
        high_dots[0] += (high & low_high_mask) != 0u
          ? low_values[index] : 0.0f;
        high_dots[1] += (high & high_high_mask) != 0u
          ? low_values[index + 8u] : 0.0f;
        high_dots[2] += (high & upper_low_high_mask) != 0u
          ? high_values[index] : 0.0f;
        high_dots[3] += (high & upper_high_high_mask) != 0u
          ? high_values[index + 8u] : 0.0f;
      }

      accumulators[projection] += float(block.d)
        * (float(scale_bytes[0]) * (quant_dots[0] + 16.0f * high_dots[0])
          + float(scale_bytes[1]) * (quant_dots[1] / 16.0f
            + 16.0f * high_dots[1])
          + float(scale_bytes[4]) * (quant_dots[2] + 16.0f * high_dots[2])
          + float(scale_bytes[5]) * (quant_dots[3] / 16.0f
            + 16.0f * high_dots[3]))
        - float(block.dmin)
        * (activation_sums[0] * float(scale_bytes[2])
          + activation_sums[1] * float(scale_bytes[3])
          + activation_sums[2] * float(scale_bytes[6])
          + activation_sums[3] * float(scale_bytes[7]));
    }
    activation += 1024u;
  }

  const float gate_sum = simd_sum(accumulators[0]);
  const float up_sum = simd_sum(accumulators[1]);
  if (lane == 0u) {
    product[row * params.n + col] =
      llmopt_gated_product(gate_sum, up_sum, params.has_bias);
  }
}

kernel void llmopt_q5_k_linear_f16_m2_x4(
    device const half* input [[buffer(0)]],
    device const block_q5_K* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint column_group = gid >> 5;
  const uint column_in_group = lane >> 3;
  const uint local_lane = lane & 7u;
  const uint col = (column_group << 2) + column_in_group;
  if (col >= params.n) return;
  const uint num_superblocks = params.k >> 8;
  device const block_q5_K* row_blocks = weight + col * num_superblocks;
  device const half* in0 = input;
  device const half* in1 = input + params.k;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (uint sb = 0; sb < num_superblocks; ++sb) {
    device const block_q5_K& b = row_blocks[sb];
    const float d = float(b.d);
    const float dmin = float(b.dmin);
    const uint superblock_base = sb << 8;
    const uchar4 high_bits =
      *reinterpret_cast<device const uchar4*>(b.qh + (local_lane << 2));
    for (uint pair = 0; pair < 4; ++pair) {
      const uint low_group = pair << 1;
      const uint high_group = low_group + 1;
      const uint low_scale = low_group < 4
        ? uint(b.scales[low_group] & 63u)
        : uint((b.scales[low_group + 4] & 0x0Fu)
            | ((b.scales[low_group - 4] >> 6) << 4));
      const uint low_min = low_group < 4
        ? uint(b.scales[low_group + 4] & 63u)
        : uint((b.scales[low_group + 4] >> 4)
            | ((b.scales[low_group] >> 6) << 4));
      const uint high_scale = high_group < 4
        ? uint(b.scales[high_group] & 63u)
        : uint((b.scales[high_group + 4] & 0x0Fu)
            | ((b.scales[high_group - 4] >> 6) << 4));
      const uint high_min = high_group < 4
        ? uint(b.scales[high_group + 4] & 63u)
        : uint((b.scales[high_group + 4] >> 4)
            | ((b.scales[high_group] >> 6) << 4));
      const uint packed_offset = (pair << 5) + (local_lane << 2);
      const uchar4 packed =
        *reinterpret_cast<device const uchar4*>(b.qs + packed_offset);
      const float4 qlow = float4((packed & uchar4(15u))
        | (((high_bits >> low_group) & uchar4(1u)) << 4));
      const float4 qhigh = float4((packed >> 4)
        | (((high_bits >> high_group) & uchar4(1u)) << 4));
      const uint input_offset = superblock_base + (low_group << 5)
        + (local_lane << 2);
      const float4 x0low = float4(
        *reinterpret_cast<device const half4*>(in0 + input_offset));
      const float4 x0high = float4(
        *reinterpret_cast<device const half4*>(in0 + input_offset + 32));
      const float4 x1low = float4(
        *reinterpret_cast<device const half4*>(in1 + input_offset));
      const float4 x1high = float4(
        *reinterpret_cast<device const half4*>(in1 + input_offset + 32));
      acc0 += d * float(low_scale) * dot(x0low, qlow)
        - dmin * float(low_min) * dot(x0low, float4(1.0f));
      acc0 += d * float(high_scale) * dot(x0high, qhigh)
        - dmin * float(high_min) * dot(x0high, float4(1.0f));
      acc1 += d * float(low_scale) * dot(x1low, qlow)
        - dmin * float(low_min) * dot(x1low, float4(1.0f));
      acc1 += d * float(high_scale) * dot(x1high, qhigh)
        - dmin * float(high_min) * dot(x1high, float4(1.0f));
    }
  }
  acc0 += simd_shuffle_down(acc0, 4);
  acc1 += simd_shuffle_down(acc1, 4);
  acc0 += simd_shuffle_down(acc0, 2);
  acc1 += simd_shuffle_down(acc1, 2);
  acc0 += simd_shuffle_down(acc0, 1);
  acc1 += simd_shuffle_down(acc1, 1);
  if (local_lane == 0) {
    const float bias_value = params.has_bias != 0u ? float(bias[col]) : 0.0f;
    output[col] = half(acc0 + bias_value);
    output[params.n + col] = half(acc1 + bias_value);
  }
}
|}

let q6_k_source = {|
struct block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};

kernel void llmopt_dequant_q6_k(
    device const block_q6_K* blocks [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant QuantBlock256Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint sb_idx = gid >> 5;
  if (sb_idx >= params.total_superblocks) return;
  device const block_q6_K& b = blocks[sb_idx];
  const float d = float(b.d);
  device const uint8_t* ql = b.ql;
  device const uint8_t* qh = b.qh;
  device const int8_t* sc = b.scales;
  const int is = lane / 16;
  const int q1 = int((ql[lane] & 0x0Fu) | (((qh[lane] >> 0) & 3u) << 4)) - 32;
  const int q2 = int((ql[32 + lane] & 0x0Fu) | (((qh[lane] >> 2) & 3u) << 4)) - 32;
  const int q3 = int((ql[lane] >> 4) | (((qh[lane] >> 4) & 3u) << 4)) - 32;
  const int q4 = int((ql[32 + lane] >> 4) | (((qh[lane] >> 6) & 3u) << 4)) - 32;
  output[sb_idx * 256 + 0 + lane]  = half(d * float(sc[is + 0]) * float(q1));
  output[sb_idx * 256 + 32 + lane] = half(d * float(sc[is + 2]) * float(q2));
  output[sb_idx * 256 + 64 + lane] = half(d * float(sc[is + 4]) * float(q3));
  output[sb_idx * 256 + 96 + lane] = half(d * float(sc[is + 6]) * float(q4));

  const int q5 = int((ql[64 + lane] & 0x0Fu) | (((qh[32 + lane] >> 0) & 3u) << 4)) - 32;
  const int q6 = int((ql[96 + lane] & 0x0Fu) | (((qh[32 + lane] >> 2) & 3u) << 4)) - 32;
  const int q7 = int((ql[64 + lane] >> 4) | (((qh[32 + lane] >> 4) & 3u) << 4)) - 32;
  const int q8 = int((ql[96 + lane] >> 4) | (((qh[32 + lane] >> 6) & 3u) << 4)) - 32;
  output[sb_idx * 256 + 128 + lane] = half(d * float(sc[8 + is + 0]) * float(q5));
  output[sb_idx * 256 + 160 + lane] = half(d * float(sc[8 + is + 2]) * float(q6));
  output[sb_idx * 256 + 192 + lane] = half(d * float(sc[8 + is + 4]) * float(q7));
  output[sb_idx * 256 + 224 + lane] = half(d * float(sc[8 + is + 6]) * float(q8));
}

kernel void llmopt_q6_k_linear_f16(
    device const half* input [[buffer(0)]],
    device const block_q6_K* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5;
  const uint total_elements = params.m * params.n;
  if (simd_idx >= total_elements) return;
  const uint row = simd_idx / params.n;
  const uint col = simd_idx - row * params.n;
  const uint num_superblocks = params.k >> 8;
  device const block_q6_K* row_blocks = weight + col * num_superblocks;
  device const half* in_ptr = input + row * params.k;
  float acc = 0.0f;
  for (uint sb = 0; sb < num_superblocks; ++sb) {
    device const block_q6_K& b = row_blocks[sb];
    const float d = float(b.d);
    device const uint8_t* ql = b.ql;
    device const uint8_t* qh = b.qh;
    device const int8_t* sc = b.scales;
    const int is = lane / 16;
    const int q1 = int((ql[lane] & 0x0Fu) | (((qh[lane] >> 0) & 3u) << 4)) - 32;
    const int q2 = int((ql[32 + lane] & 0x0Fu) | (((qh[lane] >> 2) & 3u) << 4)) - 32;
    const int q3 = int((ql[lane] >> 4) | (((qh[lane] >> 4) & 3u) << 4)) - 32;
    const int q4 = int((ql[32 + lane] >> 4) | (((qh[lane] >> 6) & 3u) << 4)) - 32;
    const uint base_idx = sb * 256 + lane;
    acc += float(in_ptr[base_idx + 0])  * (d * float(sc[is + 0]) * float(q1));
    acc += float(in_ptr[base_idx + 32]) * (d * float(sc[is + 2]) * float(q2));
    acc += float(in_ptr[base_idx + 64]) * (d * float(sc[is + 4]) * float(q3));
    acc += float(in_ptr[base_idx + 96]) * (d * float(sc[is + 6]) * float(q4));

    const int q5 = int((ql[64 + lane] & 0x0Fu) | (((qh[32 + lane] >> 0) & 3u) << 4)) - 32;
    const int q6 = int((ql[96 + lane] & 0x0Fu) | (((qh[32 + lane] >> 2) & 3u) << 4)) - 32;
    const int q7 = int((ql[64 + lane] >> 4) | (((qh[32 + lane] >> 4) & 3u) << 4)) - 32;
    const int q8 = int((ql[96 + lane] >> 4) | (((qh[32 + lane] >> 6) & 3u) << 4)) - 32;
    acc += float(in_ptr[base_idx + 128]) * (d * float(sc[8 + is + 0]) * float(q5));
    acc += float(in_ptr[base_idx + 160]) * (d * float(sc[8 + is + 2]) * float(q6));
    acc += float(in_ptr[base_idx + 192]) * (d * float(sc[8 + is + 4]) * float(q7));
    acc += float(in_ptr[base_idx + 224]) * (d * float(sc[8 + is + 6]) * float(q8));
  }
  acc = simd_sum(acc);
  if (lane == 0) {
    if (params.has_bias != 0u) acc += float(bias[col]);
    output[row * params.n + col] = half(acc);
  }
}

kernel void llmopt_q6_k_linear_f16_m2(
    device const half* input [[buffer(0)]],
    device const block_q6_K* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint col = gid >> 5;
  if (col >= params.n) return;
  const uint num_superblocks = params.k >> 8;
  device const block_q6_K* row_blocks = weight + col * num_superblocks;
  device const half* in0 = input;
  device const half* in1 = input + params.k;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (uint sb = 0; sb < num_superblocks; ++sb) {
    device const block_q6_K& b = row_blocks[sb];
    const float d = float(b.d);
    device const uint8_t* ql = b.ql;
    device const uint8_t* qh = b.qh;
    device const int8_t* sc = b.scales;
    const int is = lane / 16;
    const uint base_idx = sb * 256 + lane;
    const float w0 = d * float(sc[is + 0]) * float(int((ql[lane] & 0x0Fu) | (((qh[lane] >> 0) & 3u) << 4)) - 32);
    const float w1 = d * float(sc[is + 2]) * float(int((ql[32 + lane] & 0x0Fu) | (((qh[lane] >> 2) & 3u) << 4)) - 32);
    const float w2 = d * float(sc[is + 4]) * float(int((ql[lane] >> 4) | (((qh[lane] >> 4) & 3u) << 4)) - 32);
    const float w3 = d * float(sc[is + 6]) * float(int((ql[32 + lane] >> 4) | (((qh[lane] >> 6) & 3u) << 4)) - 32);
    const float w4 = d * float(sc[8 + is + 0]) * float(int((ql[64 + lane] & 0x0Fu) | (((qh[32 + lane] >> 0) & 3u) << 4)) - 32);
    const float w5 = d * float(sc[8 + is + 2]) * float(int((ql[96 + lane] & 0x0Fu) | (((qh[32 + lane] >> 2) & 3u) << 4)) - 32);
    const float w6 = d * float(sc[8 + is + 4]) * float(int((ql[64 + lane] >> 4) | (((qh[32 + lane] >> 4) & 3u) << 4)) - 32);
    const float w7 = d * float(sc[8 + is + 6]) * float(int((ql[96 + lane] >> 4) | (((qh[32 + lane] >> 6) & 3u) << 4)) - 32);
    acc0 += float(in0[base_idx + 0]) * w0;
    acc0 += float(in0[base_idx + 32]) * w1;
    acc0 += float(in0[base_idx + 64]) * w2;
    acc0 += float(in0[base_idx + 96]) * w3;
    acc0 += float(in0[base_idx + 128]) * w4;
    acc0 += float(in0[base_idx + 160]) * w5;
    acc0 += float(in0[base_idx + 192]) * w6;
    acc0 += float(in0[base_idx + 224]) * w7;
    acc1 += float(in1[base_idx + 0]) * w0;
    acc1 += float(in1[base_idx + 32]) * w1;
    acc1 += float(in1[base_idx + 64]) * w2;
    acc1 += float(in1[base_idx + 96]) * w3;
    acc1 += float(in1[base_idx + 128]) * w4;
    acc1 += float(in1[base_idx + 160]) * w5;
    acc1 += float(in1[base_idx + 192]) * w6;
    acc1 += float(in1[base_idx + 224]) * w7;
  }
  acc0 = simd_sum(acc0);
  acc1 = simd_sum(acc1);
  if (lane == 0) {
    const float bias_value = params.has_bias != 0u ? float(bias[col]) : 0.0f;
    output[col] = half(acc0 + bias_value);
    output[params.n + col] = half(acc1 + bias_value);
  }
}

kernel void llmopt_q6_k_linear_f16_m2_x4(
    device const half* input [[buffer(0)]],
    device const block_q6_K* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint column_group = gid >> 5;
  const uint column_in_group = lane >> 3;
  const uint local_lane = lane & 7u;
  const uint col = (column_group << 2) + column_in_group;
  if (col >= params.n) return;
  const uint num_superblocks = params.k >> 8;
  device const block_q6_K* row_blocks = weight + col * num_superblocks;
  device const half* in0 = input;
  device const half* in1 = input + params.k;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (uint sb = 0; sb < num_superblocks; ++sb) {
    device const block_q6_K& b = row_blocks[sb];
    const float d = float(b.d);
    const uint superblock_base = sb << 8;
    for (uint group = 0; group < 8; ++group) {
      const uint half_index = group >> 2;
      const uint group_in_half = group & 3u;
      const uint quant_offset = (half_index << 6)
        + ((group_in_half & 1u) << 5) + (local_lane << 2);
      const uint high_offset = (half_index << 5) + (local_lane << 2);
      const uchar4 packed =
        *reinterpret_cast<device const uchar4*>(b.ql + quant_offset);
      const uchar4 high =
        *reinterpret_cast<device const uchar4*>(b.qh + high_offset);
      const uchar4 low = group_in_half < 2
        ? (packed & uchar4(15u)) : (packed >> 4);
      const uint high_shift = group_in_half << 1;
      const int4 quantized = int4(low
        | (((high >> high_shift) & uchar4(3u)) << 4)) - int4(32);
      const uint scale_index = (half_index << 3)
        + (group_in_half << 1) + (local_lane >> 2);
      const float scale = d * float(b.scales[scale_index]);
      const uint input_offset = superblock_base + (group << 5)
        + (local_lane << 2);
      const float4 x0 = float4(
        *reinterpret_cast<device const half4*>(in0 + input_offset));
      const float4 x1 = float4(
        *reinterpret_cast<device const half4*>(in1 + input_offset));
      const float4 q = float4(quantized);
      acc0 += scale * dot(x0, q);
      acc1 += scale * dot(x1, q);
    }
  }
  acc0 += simd_shuffle_down(acc0, 4);
  acc1 += simd_shuffle_down(acc1, 4);
  acc0 += simd_shuffle_down(acc0, 2);
  acc1 += simd_shuffle_down(acc1, 2);
  acc0 += simd_shuffle_down(acc0, 1);
  acc1 += simd_shuffle_down(acc1, 1);
  if (local_lane == 0) {
    const float bias_value = params.has_bias != 0u ? float(bias[col]) : 0.0f;
    output[col] = half(acc0 + bias_value);
    output[params.n + col] = half(acc1 + bias_value);
  }
}
|}

let iq4_xs_source = {|
struct block_iq4_xs {
    half d;
    ushort scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};

constant int llmopt_iq4nl_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};

kernel void llmopt_iq4_xs_linear_f16(
    device const half* input [[buffer(0)]],
    device const block_iq4_xs* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5;
  const uint total_elements = params.m * params.n;
  if (simd_idx >= total_elements) return;
  const uint row = simd_idx / params.n;
  const uint col = simd_idx - row * params.n;
  const uint num_superblocks = params.k >> 8;
  device const block_iq4_xs* row_blocks = weight + col * num_superblocks;
  device const half* in_ptr = input + row * params.k;
  float accumulator = 0.0f;
  for (uint block = 0; block < num_superblocks; ++block) {
    device const block_iq4_xs& value = row_blocks[block];
    for (uint subblock = 0; subblock < 8; ++subblock) {
      const uint low = (value.scales_l[subblock >> 1]
          >> (4 * (subblock & 1))) & 0x0Fu;
      const uint high = (value.scales_h >> (2 * subblock)) & 0x03u;
      const int logarithmic_scale = int(low | (high << 4)) - 32;
      const float scale = float(value.d)
          * exp2(float(logarithmic_scale) * 0.125f);
      const uint8_t packed = value.qs[(subblock >> 1) * 32 + lane];
      const uint index = (subblock & 1) != 0 ? packed >> 4 : packed & 0x0Fu;
      const float dequantized = scale * float(llmopt_iq4nl_values[index]);
      accumulator += float(in_ptr[block * 256 + subblock * 32 + lane])
          * dequantized;
    }
  }
  accumulator = simd_sum(accumulator);
  if (lane == 0) {
    if (params.has_bias != 0u) accumulator += float(bias[col]);
    output[row * params.n + col] = half(accumulator);
  }
}

kernel void llmopt_iq4_xs_linear_f16_m2(
    device const half* input [[buffer(0)]],
    device const block_iq4_xs* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint col = gid >> 5;
  if (col >= params.n) return;
  const uint num_superblocks = params.k >> 8;
  device const block_iq4_xs* row_blocks = weight + col * num_superblocks;
  device const half* in0 = input;
  device const half* in1 = input + params.k;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (uint block = 0; block < num_superblocks; ++block) {
    device const block_iq4_xs& value = row_blocks[block];
    for (uint subblock = 0; subblock < 8; ++subblock) {
      const uint low = (value.scales_l[subblock >> 1]
          >> (4 * (subblock & 1))) & 0x0Fu;
      const uint high = (value.scales_h >> (2 * subblock)) & 0x03u;
      const int logarithmic_scale = int(low | (high << 4)) - 32;
      const float scale = float(value.d)
          * exp2(float(logarithmic_scale) * 0.125f);
      const uint8_t packed = value.qs[(subblock >> 1) * 32 + lane];
      const uint index = (subblock & 1) != 0 ? packed >> 4 : packed & 0x0Fu;
      const float w = scale * float(llmopt_iq4nl_values[index]);
      const uint input_index = block * 256 + subblock * 32 + lane;
      acc0 += float(in0[input_index]) * w;
      acc1 += float(in1[input_index]) * w;
    }
  }
  acc0 = simd_sum(acc0);
  acc1 = simd_sum(acc1);
  if (lane == 0) {
    const float bias_value = params.has_bias != 0u ? float(bias[col]) : 0.0f;
    output[col] = half(acc0 + bias_value);
    output[params.n + col] = half(acc1 + bias_value);
  }
}

kernel void llmopt_iq4_xs_gated_linear_f16_m2(
    device const half* input [[buffer(0)]],
    device const block_iq4_xs* gate_weight [[buffer(1)]],
    device const block_iq4_xs* up_weight [[buffer(2)]],
    device half* product [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint col = gid >> 5;
  if (col >= params.n) return;
  const uint num_superblocks = params.k >> 8;
  device const half* in0 = input;
  device const half* in1 = input + params.k;
  float accumulators[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  for (uint block = 0; block < num_superblocks; ++block) {
    for (uint subblock = 0; subblock < 8; ++subblock) {
      const uint input_index = block * 256 + subblock * 32 + lane;
      const float x0 = float(in0[input_index]);
      const float x1 = float(in1[input_index]);
      for (uint projection = 0; projection < 2u; ++projection) {
        device const block_iq4_xs* projection_weight =
          projection == 0u ? gate_weight : up_weight;
        device const block_iq4_xs& value =
          projection_weight[col * num_superblocks + block];
        const uint low = (value.scales_l[subblock >> 1]
            >> (4 * (subblock & 1))) & 0x0Fu;
        const uint high = (value.scales_h >> (2 * subblock)) & 0x03u;
        const int logarithmic_scale = int(low | (high << 4)) - 32;
        const float scale = float(value.d)
            * exp2(float(logarithmic_scale) * 0.125f);
        const uint8_t packed = value.qs[(subblock >> 1) * 32 + lane];
        const uint index = (subblock & 1) != 0
          ? packed >> 4 : packed & 0x0Fu;
        const float weight_value =
          scale * float(llmopt_iq4nl_values[index]);
        accumulators[projection * 2] += x0 * weight_value;
        accumulators[projection * 2 + 1] += x1 * weight_value;
      }
    }
  }
  for (uint index = 0; index < 4u; ++index)
    accumulators[index] = simd_sum(accumulators[index]);
  if (lane == 0) {
    product[col] = llmopt_gated_product(
      accumulators[0], accumulators[2], params.has_bias);
    product[params.n + col] = llmopt_gated_product(
      accumulators[1], accumulators[3], params.has_bias);
  }
}
|}

let kquant_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_dequant_q4_k"
      ~operation:Kernel_abi.Operation.Cast
      ~input_dtype:(Ir.Dtype.Quant Q4_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_dequant_q5_k"
      ~operation:Kernel_abi.Operation.Cast
      ~input_dtype:(Ir.Dtype.Quant Q5_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_dequant_q6_k"
      ~operation:Kernel_abi.Operation.Cast
      ~input_dtype:(Ir.Dtype.Quant Q6_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_k_linear_f16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_k_linear_f16_m2"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_q4_k_linear_f16_m2_x2_l32"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_q4_k_linear_f16_m2_x4"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q5_k_linear_f16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q5_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q5_k_linear_f16_m2"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q5_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_q5_k_linear_f16_m2_x1_l32"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q5_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_q5_k_linear_f16_m2_x4"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q5_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q6_k_linear_f16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q6_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q6_k_linear_f16_m2"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q6_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_q6_k_linear_f16_m2_x4"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q6_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_iq4_xs_linear_f16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant IQ4_XS) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_iq4_xs_linear_f16_m2"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant IQ4_XS) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_k_gated_linear_f16"
      ~operation:Kernel_abi.Operation.Gated_linear
      ~input_dtype:(Ir.Dtype.Quant Q4_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_q4_k_gated_linear_f16_m2_x1_l32"
      ~operation:Kernel_abi.Operation.Gated_linear
      ~input_dtype:(Ir.Dtype.Quant Q4_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_q5_k_gated_linear_f16_m2_x1_l32"
      ~operation:Kernel_abi.Operation.Gated_linear
      ~input_dtype:(Ir.Dtype.Quant Q5_K) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_iq4_xs_gated_linear_f16_m2"
      ~operation:Kernel_abi.Operation.Gated_linear
      ~input_dtype:(Ir.Dtype.Quant IQ4_XS) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_k_down_add_f16"
      ~operation:Kernel_abi.Operation.Fused_linear
      ~input_dtype:(Ir.Dtype.Quant Q4_K) ~output_dtype:Ir.Dtype.Float16 ]

let add_kquant_kernels program =
  Program.make
    ~source:
      (Program.source program ^ quant_common_source ^ q4_k_source ^ q5_k_source ^ q6_k_source)
    ~kernels:(Program.kernels program @ kquant_entries)

let emit_dequant_q4_k () = metal_header ^ quant_common_source ^ q4_k_source
let emit_dequant_q5_k () = metal_header ^ quant_common_source ^ q5_k_source
let emit_dequant_q6_k () = metal_header ^ quant_common_source ^ q6_k_source

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

let linear_f16_f32_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_linear_f16_f32" ~operation:Kernel_abi.Operation.Linear
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
  ^ "}\n\n"
  ^ "kernel void llmopt_rms_norm_f32_f16_wf32_simd(\n"
  ^ "    device const float* input [[buffer(0)]],\n"
  ^ "    device const float* weight [[buffer(1)]],\n"
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
  ^ "    output[base + col] = half(input[base + col] * inverse * weight[col]);\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_rms_norm_f16_wf32_simd(\n"
  ^ "    device const half* input [[buffer(0)]],\n"
  ^ "    device const float* weight [[buffer(1)]],\n"
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
  ^ "    output[base + col] = half(float(input[base + col]) * inverse * weight[col]);\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_rms_norm_add_f16_wf32_simd(\n"
  ^ "    device const half* input [[buffer(0)]],\n"
  ^ "    device const float* weight [[buffer(1)]],\n"
  ^ "    device const half* residual [[buffer(2)]],\n"
  ^ "    device half* output [[buffer(3)]],\n"
  ^ "    constant RmsNormParams& params [[buffer(4)]],\n"
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
  ^ "  for (uint col = lane; col < params.width; col += RMS_NORM_SIMD_WIDTH) {\n"
  ^ "    const half normalized = half(float(input[base + col]) * inverse * weight[col]);\n"
  ^ "    output[base + col] = half(normalized + residual[base + col]);\n"
  ^ "  }\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_rms_norm_f16_wf32_wide(\n"
  ^ "    device const half* input [[buffer(0)]],\n"
  ^ "    device const float* weight [[buffer(1)]],\n"
  ^ "    device half* output [[buffer(2)]],\n"
  ^ "    constant RmsNormParams& params [[buffer(3)]],\n"
  ^ "    uint row [[threadgroup_position_in_grid]],\n"
  ^ "    uint tid [[thread_index_in_threadgroup]],\n"
  ^ "    uint simdgroup [[simdgroup_index_in_threadgroup]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  if (row >= params.rows) return;\n"
  ^ "  const uint base = row * params.width;\n"
  ^ "  float square_sum = 0.0f;\n"
  ^ "  for (uint col = tid; col < params.width; col += 256) {\n"
  ^ "    const float value = float(input[base + col]);\n"
  ^ "    square_sum += value * value;\n"
  ^ "  }\n"
  ^ "  square_sum = simd_sum(square_sum);\n"
  ^ "  threadgroup float partial_sums[8];\n"
  ^ "  threadgroup float inverse;\n"
  ^ "  if (lane == 0) partial_sums[simdgroup] = square_sum;\n"
  ^ "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  if (tid == 0) {\n"
  ^ "    float total = 0.0f;\n"
  ^ "    for (uint group = 0; group < 8; ++group) total += partial_sums[group];\n"
  ^ "    inverse = rsqrt(total / float(params.width) + params.epsilon);\n"
  ^ "  }\n"
  ^ "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  for (uint col = tid; col < params.width; col += 256)\n"
  ^ "    output[base + col] = half(float(input[base + col]) * inverse * weight[col]);\n"
  ^ "}\n\n"
  ^ "kernel void llmopt_rms_norm_add_f16_wf32_wide(\n"
  ^ "    device const half* input [[buffer(0)]],\n"
  ^ "    device const float* weight [[buffer(1)]],\n"
  ^ "    device const half* residual [[buffer(2)]],\n"
  ^ "    device half* output [[buffer(3)]],\n"
  ^ "    constant RmsNormParams& params [[buffer(4)]],\n"
  ^ "    uint row [[threadgroup_position_in_grid]],\n"
  ^ "    uint tid [[thread_index_in_threadgroup]],\n"
  ^ "    uint simdgroup [[simdgroup_index_in_threadgroup]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  if (row >= params.rows) return;\n"
  ^ "  const uint base = row * params.width;\n"
  ^ "  float square_sum = 0.0f;\n"
  ^ "  for (uint col = tid; col < params.width; col += 256) {\n"
  ^ "    const float value = float(input[base + col]);\n"
  ^ "    square_sum += value * value;\n"
  ^ "  }\n"
  ^ "  square_sum = simd_sum(square_sum);\n"
  ^ "  threadgroup float partial_sums[8];\n"
  ^ "  threadgroup float inverse;\n"
  ^ "  if (lane == 0) partial_sums[simdgroup] = square_sum;\n"
  ^ "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  if (tid == 0) {\n"
  ^ "    float total = 0.0f;\n"
  ^ "    for (uint group = 0; group < 8; ++group) total += partial_sums[group];\n"
  ^ "    inverse = rsqrt(total / float(params.width) + params.epsilon);\n"
  ^ "  }\n"
  ^ "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  for (uint col = tid; col < params.width; col += 256) {\n"
  ^ "    const half normalized = half(float(input[base + col]) * inverse * weight[col]);\n"
  ^ "    output[base + col] = half(normalized + residual[base + col]);\n"
  ^ "  }\n"
  ^ "}\n"

let rms_norm_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_rms_norm_f32_f16_simd"
      ~operation:Kernel_abi.Operation.Rms_norm
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_rms_norm_f16_simd"
      ~operation:Kernel_abi.Operation.Rms_norm
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_rms_norm_f32_f16_wf32_simd"
      ~operation:Kernel_abi.Operation.Rms_norm
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_rms_norm_f16_wf32_simd"
      ~operation:Kernel_abi.Operation.Rms_norm
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_rms_norm_add_f16_wf32_simd"
      ~operation:Kernel_abi.Operation.Rms_norm
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_rms_norm_f16_wf32_wide"
      ~operation:Kernel_abi.Operation.Rms_norm
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_rms_norm_add_f16_wf32_wide"
      ~operation:Kernel_abi.Operation.Rms_norm
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16 ]

let l2_norm_source =
  "\nconstant uint L2_NORM_SIMD_WIDTH = 32;\n"
  ^ "constant uint L2_NORM_ROWS_PER_THREADGROUP = 8;\n\n"
  ^ "struct L2NormParams { uint rows; uint width; float epsilon; };\n\n"
  ^ "kernel void llmopt_l2_norm_f16_simd(\n"
  ^ "    device const half* input [[buffer(0)]],\n"
  ^ "    device half* output [[buffer(1)]],\n"
  ^ "    constant L2NormParams& params [[buffer(2)]],\n"
  ^ "    uint3 threadgroup_position [[threadgroup_position_in_grid]],\n"
  ^ "    uint simdgroup [[simdgroup_index_in_threadgroup]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]]) {\n"
  ^ "  const uint row = threadgroup_position.x * L2_NORM_ROWS_PER_THREADGROUP\n"
  ^ "      + simdgroup;\n"
  ^ "  if (row >= params.rows) return;\n"
  ^ "  const uint base = row * params.width;\n"
  ^ "  float square_sum = 0.0f;\n"
  ^ "  for (uint col = lane; col < params.width; col += L2_NORM_SIMD_WIDTH) {\n"
  ^ "    const float value = float(input[base + col]);\n"
  ^ "    square_sum += value * value;\n"
  ^ "  }\n"
  ^ "  square_sum = simd_sum(square_sum);\n"
  ^ "  const float inverse = rsqrt(square_sum + params.epsilon);\n"
  ^ "  for (uint col = lane; col < params.width; col += L2_NORM_SIMD_WIDTH)\n"
  ^ "    output[base + col] = half(float(input[base + col]) * inverse);\n"
  ^ "}\n"

let l2_norm_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_l2_norm_f16_simd"
      ~operation:Kernel_abi.Operation.Rms_norm
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16 ]

let rms_rope_f16_weight_source =
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

let rms_rope_f32_weight_source =
  rms_rope_f16_weight_source
  |> replace_all ~needle:"RMS_ROPE_SIMD_WIDTH"
       ~replacement:"RMS_ROPE_WF32_SIMD_WIDTH"
  |> replace_all ~needle:"RMS_ROPE_ROWS_PER_THREADGROUP"
       ~replacement:"RMS_ROPE_WF32_ROWS_PER_THREADGROUP"
  |> replace_all ~needle:"RmsRopeParams"
       ~replacement:"RmsRopeF32WeightParams"
  |> replace_first ~needle:"llmopt_rms_rope_f16_simd_h64"
       ~replacement:"llmopt_rms_rope_f16_wf32_simd"
  |> replace_first ~needle:"device const half* weight [[buffer(1)]]"
       ~replacement:"device const float* weight [[buffer(1)]]"

let rms_rope_source =
  rms_rope_f16_weight_source ^ rms_rope_f32_weight_source

let rms_rope_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_rms_rope_f16_simd_h64"
      ~operation:Kernel_abi.Operation.Rms_rope
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_rms_rope_f16_wf32_simd"
      ~operation:Kernel_abi.Operation.Rms_rope
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16 ]

let rms_rope_qk_f16_weight_source =
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

let rms_rope_qk_f32_weight_source =
  rms_rope_qk_f16_weight_source
  |> replace_all ~needle:"RMS_ROPE_QK_SIMD_WIDTH"
       ~replacement:"RMS_ROPE_QK_WF32_SIMD_WIDTH"
  |> replace_all ~needle:"RMS_ROPE_QK_ROWS_PER_THREADGROUP"
       ~replacement:"RMS_ROPE_QK_WF32_ROWS_PER_THREADGROUP"
  |> replace_all ~needle:"RmsRopeQKParams"
       ~replacement:"RmsRopeQKF32WeightParams"
  |> replace_first ~needle:"llmopt_rms_rope_qk_f16_simd_h64"
       ~replacement:"llmopt_rms_rope_qk_f16_wf32_simd"
  |> replace_first ~needle:"device const half* q_weight [[buffer(1)]]"
       ~replacement:"device const float* q_weight [[buffer(1)]]"
  |> replace_first ~needle:"device const half* k_weight [[buffer(3)]]"
       ~replacement:"device const float* k_weight [[buffer(3)]]"
  |> replace_first ~needle:"device const half* weight ="
       ~replacement:"device const float* weight ="

let rms_rope_qk_source =
  rms_rope_qk_f16_weight_source ^ rms_rope_qk_f32_weight_source

let rms_rope_qk_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_rms_rope_qk_f16_simd_h64"
      ~operation:Kernel_abi.Operation.Rms_rope_qk
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_rms_rope_qk_f16_wf32_simd"
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

let short_conv_f32_weight_source = {|
kernel void llmopt_short_conv_f16_f32(
    device const half* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant ShortConvParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint count = params.batches * params.channels * params.output_width;
  if (gid >= count) return;
  const uint position = gid % params.output_width;
  const uint channel = (gid / params.output_width) % params.channels;
  const uint batch = gid / (params.output_width * params.channels);
  float accumulator = 0.0f;
  for (uint tap = 0; tap < params.kernel_width; ++tap) {
    const int source_position = int(position * params.stride)
        - int(params.padding) + int(tap * params.dilation);
    if (source_position >= 0 && source_position < int(params.input_width)) {
      const uint input_index = ((batch * params.channels + channel)
          * params.input_width) + uint(source_position);
      const uint weight_index = channel * params.kernel_width + tap;
      accumulator += float(input[input_index]) * weight[weight_index];
    }
  }
  output[gid] = half(accumulator);
}
|}

let short_conv_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_short_conv_f16"
      ~operation:Kernel_abi.Operation.Short_conv
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_short_conv_f16_f32"
      ~operation:Kernel_abi.Operation.Short_conv
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float16 ]

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

let attention_simd_template = {|
kernel void __ATTENTION_KERNEL__(
    device const half* query [[buffer(0)]],
    device const half* key [[buffer(1)]],
    device const half* value [[buffer(2)]],
    device const uchar* mask [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant AttentionParams& params [[buffer(5)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = threadgroup_position.x * ATTENTION_ROWS_PER_THREADGROUP
      + simdgroup;
  const uint count = params.batches * params.heads * params.query_length;
  if (row >= count) return;
  const uint query_position = row % params.query_length;
  const uint head = (row / params.query_length) % params.heads;
  const uint batch = row / (params.query_length * params.heads);
  const uint effective_kv_heads = params.kv_heads > 0 ? params.kv_heads : params.heads;
  const uint kv_head = effective_kv_heads < params.heads
      ? head / (params.heads / effective_kv_heads) : head;
  const uint query_base = (((batch * params.heads + head)
      * params.query_length + query_position) * params.head_dimension);
  const uint kv_head_base =
      (batch * effective_kv_heads + kv_head) * params.key_length;
  float query_values[__ATTENTION_SIMD_CHUNKS__];
  float results[__ATTENTION_SIMD_CHUNKS__];
  for (uint chunk = 0; chunk < __ATTENTION_SIMD_CHUNKS__; ++chunk) {
    query_values[chunk] =
        float(query[query_base + chunk * ATTENTION_SIMD_WIDTH + lane]);
    results[chunk] = 0.0f;
  }
  float maximum = -INFINITY;
  float denominator = 0.0f;
  for (uint key_position = 0; key_position < params.key_length; ++key_position) {
    if (!llmopt_attention_allowed(mask, params, batch, head,
        query_position, key_position)) continue;
    const uint key_base =
        (kv_head_base + key_position) * __ATTENTION_HEAD_DIMENSION__;
    float partial_score = 0.0f;
    for (uint chunk = 0; chunk < __ATTENTION_SIMD_CHUNKS__; ++chunk) {
      const uint dimension = chunk * ATTENTION_SIMD_WIDTH + lane;
      partial_score += query_values[chunk] * float(key[key_base + dimension]);
    }
    const float score = simd_sum(partial_score) * params.scale;
    const float next_maximum = max(maximum, score);
    const float previous_scale = denominator == 0.0f ? 0.0f
        : exp(maximum - next_maximum);
    const float current_scale = exp(score - next_maximum);
    for (uint chunk = 0; chunk < __ATTENTION_SIMD_CHUNKS__; ++chunk) {
      const uint dimension = chunk * ATTENTION_SIMD_WIDTH + lane;
      results[chunk] = results[chunk] * previous_scale
          + current_scale * float(value[key_base + dimension]);
    }
    denominator = denominator * previous_scale + current_scale;
    maximum = next_maximum;
  }
  const uint output_base = params.token_first_output != 0
      ? (((batch * params.query_length + query_position) * params.heads + head)
          * __ATTENTION_HEAD_DIMENSION__)
      : (row * __ATTENTION_HEAD_DIMENSION__);
  for (uint chunk = 0; chunk < __ATTENTION_SIMD_CHUNKS__; ++chunk) {
    const uint dimension = chunk * ATTENTION_SIMD_WIDTH + lane;
    output[output_base + dimension] = half(denominator == 0.0f
        ? 0.0f : results[chunk] / denominator);
  }
}
|}

let attention_simd_kernel_name head_dimension =
  Printf.sprintf "llmopt_attention_f16_simd_h%d" head_dimension

let attention_simd_source head_dimension =
  attention_simd_template
  |> replace_all ~needle:"__ATTENTION_KERNEL__"
       ~replacement:(attention_simd_kernel_name head_dimension)
  |> replace_all ~needle:"__ATTENTION_HEAD_DIMENSION__"
       ~replacement:(string_of_int head_dimension)
  |> replace_all ~needle:"__ATTENTION_SIMD_CHUNKS__"
       ~replacement:(string_of_int (head_dimension / 32))

let attention_simd_entry head_dimension =
  kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
    ~name:(attention_simd_kernel_name head_dimension)
    ~operation:Kernel_abi.Operation.Attention ~input_dtype:Ir.Dtype.Float16
    ~output_dtype:Ir.Dtype.Float16

let attention_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_attention_f16_simd_h64"
      ~operation:Kernel_abi.Operation.Attention
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_attention_f16"
      ~operation:Kernel_abi.Operation.Attention
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16 ]

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
      ? uint(block.scales[j] & 63u)
      : uint((block.scales[j + 8u] & 15u) | ((block.scales[j] >> 6) << 4));
  const uint minimum = segment < 4u
      ? uint(block.scales[j + 4u] & 63u)
      : uint((block.scales[j + 8u] >> 4) | ((block.scales[j + 4u] >> 6) << 4));
  const uint packed = block.qs[(segment >> 1) * 32u + lane];
  const uint q = (segment & 1u) == 0u ? packed & 15u : packed >> 4;
  return half(float(block.d) * float(scale) * float(q)
      - float(block.dmin) * float(minimum));
}

inline half llmopt_embedding_q5_k_value(
    device const embedding_block_q5_K& block, uint within) {
  const uint segment = within >> 5;
  const uint lane = within & 31u;
  const uint j = segment & 3u;
  const uint scale = segment < 4u
      ? uint(block.scales[j] & 63u)
      : uint((block.scales[j + 8u] & 15u) | ((block.scales[j] >> 6) << 4));
  const uint minimum = segment < 4u
      ? uint(block.scales[j + 4u] & 63u)
      : uint((block.scales[j + 8u] >> 4) | ((block.scales[j + 4u] >> 6) << 4));
  const uint packed = block.qs[(segment >> 1) * 32u + lane];
  const uint low = (segment & 1u) == 0u ? packed & 15u : packed >> 4;
  const uint q = low | (((block.qh[lane] >> segment) & 1u) << 4);
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
      ? uint(block.ql[ql_base + lane] & 15u)
      : uint(block.ql[ql_base + lane] >> 4);
  const uint high = (uint(block.qh[qh_base + lane]) >> (local * 2u)) & 3u;
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
|}

let embedding_entries =
  [ "llmopt_embedding_f16";
    "llmopt_embedding_q8_0";
    "llmopt_embedding_q5_0";
    "llmopt_embedding_q4_0";
    "llmopt_embedding_q4_k";
    "llmopt_embedding_q5_k";
    "llmopt_embedding_q6_k" ]
  |> List.map (fun name ->
         kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1) ~name
           ~operation:Kernel_abi.Operation.Embedding
           ~input_dtype:Ir.Dtype.Int64 ~output_dtype:Ir.Dtype.Float16)

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

let reduction_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_sum_f16" ~operation:Kernel_abi.Operation.Reduction
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_sum_f32" ~operation:Kernel_abi.Operation.Reduction
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_mean_f32" ~operation:Kernel_abi.Operation.Reduction
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32 ]

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

let triangular_recurrence_source =
  "\nstruct TriangularRecurrenceParams {\n"
  ^ "  uint outer; uint width; uint start; uint stop;\n"
  ^ "};\n\n"
  ^ "kernel void llmopt_triangular_recurrence_f32(\n"
  ^ "    device const float* input [[buffer(0)]],\n"
  ^ "    device float* output [[buffer(1)]],\n"
  ^ "    constant TriangularRecurrenceParams& params [[buffer(2)]],\n"
  ^ "    uint tid [[thread_index_in_threadgroup]],\n"
  ^ "    uint group [[threadgroup_position_in_grid]]) {\n"
  ^ "  if (group >= params.outer || params.width > 64) return;\n"
  ^ "  threadgroup float state[64 * 64];\n"
  ^ "  threadgroup float row_values[64];\n"
  ^ "  const uint elements = params.width * params.width;\n"
  ^ "  const uint base = group * elements;\n"
  ^ "  for (uint offset = tid; offset < elements; offset += 64) {\n"
  ^ "    state[offset] = input[base + offset];\n"
  ^ "  }\n"
  ^ "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  for (uint row = params.start; row < params.stop; ++row) {\n"
  ^ "    if (tid < row) row_values[tid] = state[row * params.width + tid];\n"
  ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "    if (tid < row) {\n"
  ^ "      float accumulator = 0.0f;\n"
  ^ "      for (uint inner = 0; inner < row; ++inner) {\n"
  ^ "        volatile float product = row_values[inner]\n"
  ^ "            * state[inner * params.width + tid];\n"
  ^ "        accumulator = accumulator + product;\n"
  ^ "      }\n"
  ^ "      state[row * params.width + tid] = row_values[tid] + accumulator;\n"
  ^ "    }\n"
  ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  }\n"
  ^ "  for (uint offset = tid; offset < elements; offset += 64) {\n"
  ^ "    output[base + offset] = state[offset];\n"
  ^ "  }\n"
  ^ "}\n\n"

let triangular_recurrence_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_triangular_recurrence_f32"
      ~operation:Kernel_abi.Operation.Triangular_recurrence
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32 ]

type gated_delta_tactic =
  | Gated_delta_head_major_f32 of int
  | Gated_delta_token_major_f16 of int

let gated_delta_width = function
  | Gated_delta_head_major_f32 width
  | Gated_delta_token_major_f16 width ->
      width

let gated_delta_kernel_name = function
  | Gated_delta_head_major_f32 width ->
      Printf.sprintf "llmopt_gated_delta_f32_d%d" width
  | Gated_delta_token_major_f16 width ->
      Printf.sprintf "llmopt_gated_delta_tm_f16_d%d" width

let gated_delta_source tactic =
  let width = gated_delta_width tactic in
  let segments = width / 32 in
  let vector_type, beta_type, vector_base, scalar_index =
    match tactic with
    | Gated_delta_head_major_f32 _ ->
        ( "float", "float",
          "((batch * params.heads + head) * params.tokens + token) * params.width",
          "(batch * params.heads + head) * params.tokens + token" )
    | Gated_delta_token_major_f16 _ ->
        ( "half", "half",
          "((batch * params.tokens + token) * params.heads + head) * params.width",
          "(batch * params.tokens + token) * params.heads + head" )
  in
  Printf.sprintf
    {|
struct GatedDeltaParams {
  uint batch; uint heads; uint tokens; uint width;
};

kernel void %s(
    device const %s* query [[buffer(0)]],
    device const %s* key [[buffer(1)]],
    device const %s* value [[buffer(2)]],
    device const float* gate [[buffer(3)]],
    device const %s* beta [[buffer(4)]],
    device half* output [[buffer(5)]],
    constant GatedDeltaParams& params [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  const uint state_row = group * 4 + simdgroup;
  const uint state_rows = params.batch * params.heads * params.width;
  if (state_row >= state_rows || params.width != %d) return;
  const uint row = state_row %% params.width;
  const uint head_batch = state_row / params.width;
  const uint head = head_batch %% params.heads;
  const uint batch = head_batch / params.heads;
  float state[%d];
  for (uint segment = 0; segment < %d; ++segment) state[segment] = 0.0f;
  const float scale = rsqrt(float(params.width));
  for (uint token = 0; token < params.tokens; ++token) {
    const uint vector_base = %s;
    const uint scalar_index = %s;
    const float decay = exp(float(gate[scalar_index]));
    float state_dot_key = 0.0f;
    for (uint segment = 0; segment < %d; ++segment) {
      const uint column = segment * 32 + lane;
      state[segment] *= decay;
      state_dot_key += state[segment] * float(key[vector_base + column]);
    }
    state_dot_key = simd_sum(state_dot_key);
    const float delta = (float(value[vector_base + row]) - state_dot_key)
        * float(beta[scalar_index]);
    float state_dot_query = 0.0f;
    for (uint segment = 0; segment < %d; ++segment) {
      const uint column = segment * 32 + lane;
      state[segment] += float(key[vector_base + column]) * delta;
      state_dot_query += state[segment] * float(query[vector_base + column]);
    }
    state_dot_query = simd_sum(state_dot_query);
    if (lane == 0) {
      const uint output_index = ((batch * params.tokens + token) * params.heads + head)
          * params.width + row;
      output[output_index] = half(state_dot_query * scale);
    }
  }
}
|}
    (gated_delta_kernel_name tactic) vector_type vector_type vector_type beta_type
    width segments segments vector_base scalar_index segments segments

let gated_delta_entry tactic =
  kernel_entry_with_threadgroup ~threadgroup:(128, 1, 1)
    ~name:(gated_delta_kernel_name tactic)
    ~operation:Kernel_abi.Operation.Gated_delta
    ~input_dtype:
      (match tactic with
      | Gated_delta_head_major_f32 _ -> Ir.Dtype.Float32
      | Gated_delta_token_major_f16 _ -> Ir.Dtype.Float16)
    ~output_dtype:Ir.Dtype.Float16

let gated_delta_tactics graph =
  Ir.Graph.nodes graph
  |> List.filter_map (fun node ->
         match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
         | Ir.Op.Primitive Ir.Primitive.Gated_delta,
           [ query; key; value; gate; beta ], Some output ->
             let shape value =
               Tensor_shape.dimensions (Ir.Value.logical_shape value)
             in
             (match Ir.Value.dtype query, shape query, shape output with
             | Ir.Dtype.Float32,
               [ batch; heads; tokens; width ],
               [ output_batch; output_tokens; output_heads; output_width ]
               when width > 0 && width mod 32 = 0
                    && output_batch = batch && output_tokens = tokens
                    && output_heads = heads && output_width = width
                    && Ir.Value.dtype key = Ir.Dtype.Float32
                    && Ir.Value.dtype value = Ir.Dtype.Float32
                    && Ir.Value.dtype gate = Ir.Dtype.Float32
                    && Ir.Value.dtype beta = Ir.Dtype.Float32
                    && shape key = shape query && shape value = shape query
                    && shape gate = [ batch; heads; tokens ]
                    && shape beta = [ batch; heads; tokens ] ->
                 Some (Gated_delta_head_major_f32 width)
             | Ir.Dtype.Float16,
               [ batch; tokens; heads; width ],
               [ output_batch; output_tokens; output_heads; output_width ]
               when width > 0 && width mod 32 = 0
                    && output_batch = batch && output_tokens = tokens
                    && output_heads = heads && output_width = width
                    && Ir.Value.dtype key = Ir.Dtype.Float16
                    && Ir.Value.dtype value = Ir.Dtype.Float16
                    && Ir.Value.dtype gate = Ir.Dtype.Float32
                    && Ir.Value.dtype beta = Ir.Dtype.Float16
                    && shape key = shape query && shape value = shape query
                    && shape gate = [ batch; tokens; heads ]
                    && shape beta = [ batch; tokens; heads ] ->
                 Some (Gated_delta_token_major_f16 width)
             | _ -> None)
         | _ -> None)
  |> List.sort_uniq Stdlib.compare

let has_rms_norm graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Rms_norm _ | Ir.Op.Rms_norm_add _ -> true
         | _ -> false)

let has_l2_norm graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Primitive (Ir.Primitive.L2_norm _) -> true
         | _ -> false)

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

let attention_tactics ~target graph =
  Ir.Graph.nodes graph
  |> List.filter_map (fun node ->
         match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
         | Ir.Op.Primitive (Ir.Primitive.Attention _), query :: _, Some output -> (
             match
               Tensor_shape.dimensions (Ir.Value.logical_shape query) |> List.rev
             with
             | head_dimension :: _ ->
                 Tactic.select_attention ~target ~head_dimension
                   ~input_dtype:(Ir.Value.dtype query)
                   ~output_dtype:(Ir.Value.dtype output)
                 |> Option.map (fun tactic -> head_dimension, tactic)
             | _ -> None)
         | _ -> None)
  |> List.sort_uniq (fun (_, left) (_, right) ->
         String.compare (Tactic.name left) (Tactic.name right))

let attention_simd_head_dimensions tactics =
  tactics
  |> List.filter_map (fun (head_dimension, tactic) ->
         if
           head_dimension <> 64
           && Tactic.name tactic = attention_simd_kernel_name head_dimension
         then Some head_dimension
         else None)

let attention_tactic_entries tactics =
  tactics
  |> List.map (fun (head_dimension, tactic) ->
         match
           attention_entries
           |> List.find_opt (fun entry ->
                  Kernel_abi.Entry.name entry = Tactic.name tactic)
         with
         | Some entry -> entry
         | None -> attention_simd_entry head_dimension)

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

let has_tensor_primitive graph =
  has_primitive graph (function
    | Ir.Primitive.Pad_right_zero _ | Ir.Primitive.Triangular _
    | Ir.Primitive.Masked_fill _ | Ir.Primitive.Eye
    | Ir.Primitive.Batched_matmul -> true
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

let has_f16_bf16_linear graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
         | ( Ir.Op.Linear { bias = false; _ },
             [ input; weight ],
             Some output ) ->
             Ir.Value.dtype input = Ir.Dtype.Float16
             && Ir.Value.dtype weight = Ir.Dtype.Bfloat16
             && Ir.Value.dtype output = Ir.Dtype.Float16
         | _ -> false)

let has_f16_f32_linear graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
         | ( Ir.Op.Linear { bias = false; _ },
             [ input; weight ],
             Some output ) ->
             Ir.Value.dtype input = Ir.Dtype.Float16
             && Ir.Value.dtype weight = Ir.Dtype.Float32
             && Ir.Value.dtype output = Ir.Dtype.Float16
         | _ -> false)

let has_quant_linear graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
         | ( Ir.Op.Linear _,
             _input :: weight :: _,
             Some output ) ->
             (match Ir.Value.dtype weight with
             | Ir.Dtype.Quant
                 (Q8_0 | Q4_K | Q5_K | Q6_K | Q5_0 | Q4_0 | IQ4_XS) ->
                 Ir.Value.dtype output = Ir.Dtype.Float16
             | _ -> false)
         | ( Ir.Op.Gated_linear _,
             _input :: gate_weight :: up_weight :: _,
             Some output ) ->
             (match Ir.Value.dtype gate_weight with
             | Ir.Dtype.Quant
                 (Ir.Dtype.Q4_K | Ir.Dtype.Q5_K | Ir.Dtype.IQ4_XS) -> true
             | _ -> false)
             && Ir.Value.dtype up_weight = Ir.Value.dtype gate_weight
             && Ir.Value.dtype output = Ir.Dtype.Float16
         | _ -> false)

let quant_linear_tactics ~target graph =
  Ir.Graph.nodes graph
  |> List.filter_map (fun node ->
         match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
         | Ir.Op.Linear { m; n; k; bias }, input :: weight :: parameters, Some output -> (
             match Ir.Linear_storage.classify ~has_bias:bias ~weight ~parameters with
             | Error _ -> None
             | Ok storage ->
                 Tactic.select_linear ~target ~m ~n ~k
                   ~input_dtype:(Ir.Value.dtype input) ~storage:storage.layout
                   ~output_dtype:(Ir.Value.dtype output))
         | _ -> None)
  |> List.sort_uniq (fun left right ->
         String.compare (Tactic.name left) (Tactic.name right))

let quant_tactic_entries tactics =
  let selected_names = List.map Tactic.name tactics in
  block32_entries @ kquant_entries
  |> List.filter (fun entry ->
         Kernel_abi.Entry.operation entry <> Kernel_abi.Operation.Linear
         || List.mem (Kernel_abi.Entry.name entry) selected_names)

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

let lower ?(target = Target_hardware.default) graph =
  let materialized_movement = has_materialized_movement graph in
  let update_slice =
    has_primitive graph (function Ir.Primitive.Update_slice _ -> true | _ -> false)
  in
  let triangular_recurrence =
    has_primitive graph (function
      | Ir.Primitive.Triangular_recurrence _ -> true
      | _ -> false)
  in
  let attention_tactics = attention_tactics ~target graph in
  let attention_head_dimensions =
    attention_simd_head_dimensions attention_tactics
  in
  let attention_specialized_source =
    attention_head_dimensions |> List.map attention_simd_source
    |> String.concat "\n"
  in
  let selected_attention_entries = attention_tactic_entries attention_tactics in
  let quant_tactics = quant_linear_tactics ~target graph in
  let selected_quant_entries = quant_tactic_entries quant_tactics in
  let gated_delta_tactics = gated_delta_tactics graph in
  let gated_delta_source =
    gated_delta_tactics |> List.map gated_delta_source |> String.concat "\n"
  in
  let gated_delta_entries = List.map gated_delta_entry gated_delta_tactics in
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
      ( has_quant_linear graph,
        quant_common_source ^ q8_0_source ^ q5_0_source ^ q4_0_source
        ^ q4_k_source ^ q5_k_source ^ q6_k_source ^ iq4_xs_source,
        selected_quant_entries );
      has_f16_linear graph, linear_f16_source, linear_f16_entries;
      ( has_f16_bf16_linear graph,
        linear_f16_bf16_source,
        linear_f16_bf16_entries );
      ( has_f16_f32_linear graph,
        linear_f16_f32_source,
        linear_f16_f32_entries );
      has_rms_norm graph, rms_norm_source, rms_norm_entries;
      has_l2_norm graph, l2_norm_source, l2_norm_entries;
      has_rms_rope graph, rms_rope_source, rms_rope_entries;
      has_rms_rope_qk graph, rms_rope_qk_source, rms_rope_qk_entries;
      ( has_short_conv graph,
        short_conv_source ^ short_conv_f32_weight_source,
        short_conv_entries );
      has_short_conv_step graph, short_conv_step_source, short_conv_step_entries;
      ( has_short_conv_step_fused graph,
        short_conv_step_fused_source,
        short_conv_step_fused_entries );
      has_short_conv_prefill graph, short_conv_prefill_source, short_conv_prefill_entries;
      ( has_short_conv_prefill graph,
        short_conv_prefill_init_source,
        short_conv_prefill_init_entries );
      ( has_attention graph,
        attention_source ^ attention_specialized_source,
        selected_attention_entries );
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
      ( triangular_recurrence,
        triangular_recurrence_source,
        triangular_recurrence_entries );
      (gated_delta_tactics <> [], gated_delta_source, gated_delta_entries);
      (has_tensor_primitive graph, tensor_primitive_source, tensor_primitive_entries);
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
          | Ir.Primitive.Reduce _ -> true
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

let emit ?target graph = Result.map Program.source (lower ?target graph)
