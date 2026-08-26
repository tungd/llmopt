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

type q8_input_mode = Direct | Product

let q8_input_signature ~value_type = function
  | Direct ->
      "    device const " ^ value_type ^ "* input [[buffer(0)]],\n", 0
  | Product ->
      ( "    device const " ^ value_type ^ "* input_left [[buffer(0)]],\n"
        ^ "    device const " ^ value_type ^ "* input_right [[buffer(1)]],\n",
        1 )

let q8_input_vector_load ~value_type ~vector_type ~offset ~values_name = function
  | Direct ->
      "        device const " ^ vector_type ^ "* input_vector =\n"
      ^ "            reinterpret_cast<device const " ^ vector_type
      ^ "*>(input + " ^ offset ^ ");\n"
      ^ "        " ^ vector_type ^ " " ^ values_name ^ " = *input_vector;\n"
  | Product ->
      "        device const " ^ vector_type ^ "* input_left_vector =\n"
      ^ "            reinterpret_cast<device const " ^ vector_type
      ^ "*>(input_left + " ^ offset ^ ");\n"
      ^ "        device const " ^ vector_type ^ "* input_right_vector =\n"
      ^ "            reinterpret_cast<device const " ^ vector_type
      ^ "*>(input_right + " ^ offset ^ ");\n"
      ^ "        const " ^ vector_type ^ " left_values = *input_left_vector;\n"
      ^ "        const " ^ vector_type ^ " right_values = *input_right_vector;\n"
      ^ "        " ^ vector_type ^ " " ^ values_name ^ " = " ^ vector_type
      ^ "(left_values * right_values);\n"

let q8_input_scalar ~value_type index = function
  | Direct -> "input[" ^ index ^ "]"
  | Product ->
      value_type ^ "(input_left[" ^ index ^ "] * input_right[" ^ index ^ "])"

let q8_kernel_with_input ~input_mode ~extra_argument ~output_buffer
    ~parameter_buffer ~name ~value_type ~vector_type ~alignment
    ~weight_cast ~scale_type ~scale_zero ~zero_value ~store_value =
  let input_signature, input_shift = q8_input_signature ~value_type input_mode in
  let input_vector_load =
    q8_input_vector_load ~value_type ~vector_type ~offset:"input_offset"
      ~values_name:"input_values" input_mode
  in
  let input_scalar =
    q8_input_scalar ~value_type "input_offset + lane" input_mode
  in
  "kernel void " ^ name ^ "(\n"
  ^ input_signature
  ^ "    device const char* weight [[buffer(" ^ string_of_int (1 + input_shift)
  ^ ")]],\n"
  ^ "    device const half* scale [[buffer(" ^ string_of_int (2 + input_shift)
  ^ ")]],\n"
  ^ "    device const half* bias_or_scale [[buffer("
  ^ string_of_int (3 + input_shift) ^ ")]],\n"
  ^ extra_argument
  ^ "    device " ^ value_type ^ "* output [[buffer("
  ^ string_of_int output_buffer ^ ")]],\n"
  ^ "    constant Q8Params& params [[buffer(" ^ string_of_int parameter_buffer
  ^ ")]],\n"
  ^ "    uint2 gid [[thread_position_in_grid]],\n"
  ^ "    uint2 tid [[thread_position_in_threadgroup]]) {\n"
  ^ "  const uint row = gid.y;\n"
  ^ "  const uint col = gid.x;\n"
  ^ "  threadgroup " ^ vector_type ^ " input_tile[16][16];\n"
  ^ "  threadgroup " ^ vector_type ^ " weight_tile[16][16];\n"
  ^ "  float acc = 0.0f;\n"
  ^ "  const " ^ scale_type ^ " channel_scale =\n"
  ^ "      (col < params.n) ? scale[col] : " ^ scale_zero ^ ";\n"
  ^ "  for (uint base = 0; base < params.k; base += Q8_TILE) {\n"
  ^ "    const uint input_offset = row * params.k + base + tid.x * 4;\n"
  ^ "    if (row < params.m && base + tid.x * 4 + 3 < params.k &&\n"
  ^ "        input_offset % " ^ alignment ^ " == 0) {\n"
  ^ input_vector_load
  ^ "      input_tile[tid.y][tid.x] = input_values;\n"
  ^ "    } else {\n"
  ^ "      " ^ vector_type ^ " input_values = " ^ vector_type ^ "("
  ^ zero_value ^ ");\n"
  ^ "      for (uint lane = 0; lane < 4; ++lane)\n"
  ^ "        if (row < params.m && base + tid.x * 4 + lane < params.k)\n"
  ^ "          input_values[lane] = " ^ input_scalar ^ ";\n"
  ^ "      input_tile[tid.y][tid.x] = input_values;\n"
  ^ "    }\n"
  ^ "    const uint weight_offset = col * params.k + base + tid.y * 4;\n"
  ^ "    if (col < params.n && base + tid.y * 4 + 3 < params.k &&\n"
  ^ "        weight_offset % 4 == 0) {\n"
  ^ "      device const char4* weight_vector =\n"
  ^ "          reinterpret_cast<device const char4*>(weight + weight_offset);\n"
  ^ "      const char4 weight_values = *weight_vector;\n"
  ^ "      const " ^ vector_type ^ " dequantized_weights = " ^ vector_type
  ^ "(weight_values) * channel_scale;\n"
  ^ "      weight_tile[tid.y][tid.x] = dequantized_weights;\n"
  ^ "    } else {\n"
  ^ "      " ^ vector_type ^ " dequantized_weights = " ^ vector_type ^ "("
  ^ weight_cast ^ "(0));\n"
  ^ "      for (uint lane = 0; lane < 4; ++lane)\n"
  ^ "        if (col < params.n && base + tid.y * 4 + lane < params.k)\n"
  ^ "          dequantized_weights[lane] = " ^ weight_cast
  ^ "(weight[weight_offset + lane]) * channel_scale;\n"
  ^ "      weight_tile[tid.y][tid.x] = dequantized_weights;\n"
  ^ "    }\n"
  ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "    #pragma clang loop unroll(full)\n"
  ^ "    for (uint inner = 0; inner < 16; ++inner)\n"
  ^ "      acc += dot(float4(input_tile[tid.y][inner]),\n"
  ^ "                 float4(weight_tile[inner][tid.x]));\n"
  ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  }\n"
  ^ "  if (row < params.m && col < params.n) {\n"
  ^ "    if (params.has_bias != 0) acc += float(bias_or_scale[col]);\n"
  ^ "    output[row * params.n + col] = " ^ store_value ^ ";\n"
  ^ "  }\n"
  ^ "}\n"

let q8_kernel = q8_kernel_with_input ~input_mode:Direct

let validate_q8_tile ~tile_m ~tile_n ~tile_k =
  if tile_m <= 0 || tile_n <= 0 then
    invalid_arg "Q8 tile output dimensions must be positive"
  else if tile_k <= 0 || tile_k mod 4 <> 0 then
    invalid_arg "Q8 tile reduction dimension must be a positive multiple of four"

let q8_parameterized_kernel_with_input ~tile_m ~tile_n ~tile_k ~input_mode
    ~extra_argument ~output_buffer ~parameter_buffer ~name ~value_type
    ~vector_type ~alignment ~weight_cast ~scale_type ~scale_zero ~zero_value
    ~store_value =
  validate_q8_tile ~tile_m ~tile_n ~tile_k;
  let tile_k_vectors = tile_k / 4 in
  let thread_count = tile_m * tile_n in
  let input_vector_load =
    q8_input_vector_load ~value_type ~vector_type ~offset:"input_offset"
      ~values_name:"input_values" input_mode
  in
  let input_scalar =
    q8_input_scalar ~value_type "input_offset + lane" input_mode
  in
  let input_signature, input_shift = q8_input_signature ~value_type input_mode in
  "kernel void " ^ name ^ "(\n"
  ^ input_signature
  ^ "    device const char* weight [[buffer(" ^ string_of_int (1 + input_shift)
  ^ ")]],\n"
  ^ "    device const half* scale [[buffer(" ^ string_of_int (2 + input_shift)
  ^ ")]],\n"
  ^ "    device const half* bias_or_scale [[buffer("
  ^ string_of_int (3 + input_shift) ^ ")]],\n"
  ^ extra_argument
  ^ "    device " ^ value_type ^ "* output [[buffer(" ^ string_of_int output_buffer
  ^ ")]],\n"
  ^ "    constant Q8Params& params [[buffer(" ^ string_of_int parameter_buffer
  ^ ")]],\n"
  ^ "    uint2 gid [[thread_position_in_grid]],\n"
  ^ "    uint2 tid [[thread_position_in_threadgroup]],\n"
  ^ "    uint2 group_position [[threadgroup_position_in_grid]]) {\n"
  ^ "  const uint row = gid.y;\n"
  ^ "  const uint col = gid.x;\n"
  ^ "  const uint group_row = group_position.y * " ^ string_of_int tile_m ^ ";\n"
  ^ "  const uint group_col = group_position.x * " ^ string_of_int tile_n ^ ";\n"
  ^ "  const uint local_thread = tid.y * " ^ string_of_int tile_n ^ " + tid.x;\n"
  ^ "  const uint thread_count = " ^ string_of_int thread_count ^ ";\n"
  ^ "  threadgroup " ^ vector_type ^ " input_tile[" ^ string_of_int tile_m
  ^ "][" ^ string_of_int tile_k_vectors ^ "];\n"
  ^ "  threadgroup " ^ vector_type ^ " weight_tile[" ^ string_of_int tile_k_vectors
  ^ "][" ^ string_of_int tile_n ^ "];\n"
  ^ "  float acc = 0.0f;\n"
  ^ "  for (uint base = 0; base < params.k; base += "
  ^ string_of_int tile_k ^ ") {\n"
  ^ "    for (uint load = local_thread; load < "
  ^ string_of_int (tile_m * tile_k_vectors)
  ^ "; load += thread_count) {\n"
  ^ "      const uint local_row = load / " ^ string_of_int tile_k_vectors ^ ";\n"
  ^ "      const uint vector_index = load % " ^ string_of_int tile_k_vectors ^ ";\n"
  ^ "      const uint input_row = group_row + local_row;\n"
  ^ "      const uint input_offset = input_row * params.k + base + vector_index * 4;\n"
  ^ "      if (input_row < params.m && base + vector_index * 4 + 3 < params.k &&\n"
  ^ "          input_offset % " ^ alignment ^ " == 0) {\n"
  ^ input_vector_load
  ^ "        input_tile[local_row][vector_index] = input_values;\n"
  ^ "      } else {\n"
  ^ "        " ^ vector_type ^ " input_values = " ^ vector_type ^ "(" ^ zero_value
  ^ ");\n"
  ^ "        for (uint lane = 0; lane < 4; ++lane)\n"
  ^ "          if (input_row < params.m && base + vector_index * 4 + lane < params.k)\n"
  ^ "            input_values[lane] = " ^ input_scalar ^ ";\n"
  ^ "        input_tile[local_row][vector_index] = input_values;\n"
  ^ "      }\n"
  ^ "    }\n"
  ^ "    for (uint load = local_thread; load < "
  ^ string_of_int (tile_k_vectors * tile_n)
  ^ "; load += thread_count) {\n"
  ^ "      const uint inner = load / " ^ string_of_int tile_n ^ ";\n"
  ^ "      const uint local_col = load % " ^ string_of_int tile_n ^ ";\n"
  ^ "      const uint weight_col = group_col + local_col;\n"
  ^ "      const uint weight_offset = weight_col * params.k + base + inner * 4;\n"
  ^ "      const " ^ scale_type ^ " weight_scale =\n"
  ^ "          (weight_col < params.n) ? scale[weight_col] : " ^ scale_zero ^ ";\n"
  ^ "      if (weight_col < params.n && base + inner * 4 + 3 < params.k &&\n"
  ^ "          weight_offset % 4 == 0) {\n"
  ^ "        device const char4* weight_vector =\n"
  ^ "            reinterpret_cast<device const char4*>(weight + weight_offset);\n"
  ^ "        const char4 weight_values = *weight_vector;\n"
  ^ "        weight_tile[inner][local_col] = " ^ vector_type
  ^ "(weight_values) * weight_scale;\n"
  ^ "      } else {\n"
  ^ "        " ^ vector_type ^ " dequantized_weights = " ^ vector_type ^ "("
  ^ weight_cast ^ "(0));\n"
  ^ "        for (uint lane = 0; lane < 4; ++lane)\n"
  ^ "          if (weight_col < params.n && base + inner * 4 + lane < params.k)\n"
  ^ "            dequantized_weights[lane] = " ^ weight_cast
  ^ "(weight[weight_offset + lane]) * weight_scale;\n"
  ^ "        weight_tile[inner][local_col] = dequantized_weights;\n"
  ^ "      }\n"
  ^ "    }\n"
  ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "    #pragma clang loop unroll(full)\n"
  ^ "    for (uint inner = 0; inner < " ^ string_of_int tile_k_vectors ^ "; ++inner)\n"
  ^ "      acc += dot(float4(input_tile[tid.y][inner]),\n"
  ^ "                 float4(weight_tile[inner][tid.x]));\n"
  ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  }\n"
  ^ "  if (row < params.m && col < params.n) {\n"
  ^ "    if (params.has_bias != 0) acc += float(bias_or_scale[col]);\n"
  ^ "    output[row * params.n + col] = " ^ store_value ^ ";\n"
  ^ "  }\n"
  ^ "}\n"

let q8_parameterized_name ~tile_m ~tile_n ~tile_k =
  Printf.sprintf "llmopt_q8_linear_tm%d_tn%d_tk%d" tile_m tile_n tile_k

let q8_kernel_parameterized ~tile_m ~tile_n ~tile_k =
  q8_parameterized_kernel_with_input ~tile_m ~tile_n ~tile_k
    ~input_mode:Direct ~extra_argument:"" ~output_buffer:4 ~parameter_buffer:5
    ~name:(q8_parameterized_name ~tile_m ~tile_n ~tile_k) ~value_type:"half"
    ~vector_type:"half4" ~alignment:"8" ~weight_cast:"half"
    ~scale_type:"half" ~scale_zero:"half(0.0h)" ~zero_value:"half(0.0h)"
    ~store_value:"half(acc)"

let q8_kernel_parameterized_f32 ~tile_m ~tile_n ~tile_k =
  q8_parameterized_kernel_with_input ~tile_m ~tile_n ~tile_k
    ~input_mode:Direct ~extra_argument:"" ~output_buffer:4 ~parameter_buffer:5
    ~name:(q8_parameterized_name ~tile_m ~tile_n ~tile_k ^ "_f32")
    ~value_type:"float" ~vector_type:"float4" ~alignment:"16"
    ~weight_cast:"float" ~scale_type:"float" ~scale_zero:"0.0f"
    ~zero_value:"0.0f" ~store_value:"acc"

let q8_gemv_kernel_with_input ~input_mode ~extra_argument ~output_buffer
    ~parameter_buffer ~name ~value_type ~vector_type ~weight_cast ~scale_type
    ~store_value =
  let input_signature, input_shift = q8_input_signature ~value_type input_mode in
  let input_vector_load =
    q8_input_vector_load ~value_type ~vector_type ~offset:"inner"
      ~values_name:"input_values" input_mode
  in
  let input_scalar = q8_input_scalar ~value_type "inner" input_mode in
  "kernel void " ^ name ^ "(\n"
  ^ input_signature
  ^ "    device const char* weight [[buffer(" ^ string_of_int (1 + input_shift)
  ^ ")]],\n"
  ^ "    device const half* scale [[buffer(" ^ string_of_int (2 + input_shift)
  ^ ")]],\n"
  ^ "    device const half* bias_or_scale [[buffer("
  ^ string_of_int (3 + input_shift) ^ ")]],\n"
  ^ extra_argument
  ^ "    device " ^ value_type ^ "* output [[buffer("
  ^ string_of_int output_buffer ^ ")]],\n"
  ^ "    constant Q8Params& params [[buffer(" ^ string_of_int parameter_buffer
  ^ ")]],\n"
  ^ "    uint col [[thread_position_in_grid]]) {\n"
  ^ "  if (params.m != 1 || col >= params.n) return;\n"
  ^ "  const " ^ scale_type ^ " channel_scale = scale[col];\n"
  ^ "  const uint weight_base = col * params.k;\n"
  ^ "  float acc = 0.0f;\n"
  ^ "  uint inner = 0;\n"
  ^ "  if (weight_base % 4 == 0) {\n"
  ^ "    for (; inner + 3 < params.k; inner += 4) {\n"
  ^ input_vector_load
  ^ "      device const char4* weight_vector =\n"
  ^ "          reinterpret_cast<device const char4*>(weight + weight_base + inner);\n"
  ^ "      const char4 weight_values = *weight_vector;\n"
  ^ "      acc += float(input_values[0]) * float("
  ^ weight_cast ^ "(weight_values[0]) * channel_scale);\n"
  ^ "      acc += float(input_values[1]) * float("
  ^ weight_cast ^ "(weight_values[1]) * channel_scale);\n"
  ^ "      acc += float(input_values[2]) * float("
  ^ weight_cast ^ "(weight_values[2]) * channel_scale);\n"
  ^ "      acc += float(input_values[3]) * float("
  ^ weight_cast ^ "(weight_values[3]) * channel_scale);\n"
  ^ "    }\n"
  ^ "  }\n"
  ^ "  for (; inner < params.k; ++inner)\n"
  ^ "    acc += float(" ^ input_scalar ^ ") *\n"
  ^ "           float(" ^ weight_cast
  ^ "(weight[weight_base + inner]) * channel_scale);\n"
  ^ "  if (params.has_bias != 0) acc += float(bias_or_scale[col]);\n"
  ^ "  output[col] = " ^ store_value ^ ";\n"
  ^ "}\n"

let q8_gemv_kernel = q8_gemv_kernel_with_input ~input_mode:Direct

let q8_gemv_simd_kernel_with_input ~input_mode ~extra_argument ~output_buffer
    ~parameter_buffer ~name ~value_type ~vector_type ~weight_cast ~scale_type
    ~store_value =
  let input_signature, input_shift = q8_input_signature ~value_type input_mode in
  let input_vector_load =
    q8_input_vector_load ~value_type ~vector_type ~offset:"inner"
      ~values_name:"input_values" input_mode
  in
  let input_scalar = q8_input_scalar ~value_type "inner" input_mode in
  "kernel void " ^ name ^ "(\n"
  ^ input_signature
  ^ "    device const char* weight [[buffer(" ^ string_of_int (1 + input_shift)
  ^ ")]],\n"
  ^ "    device const half* scale [[buffer(" ^ string_of_int (2 + input_shift)
  ^ ")]],\n"
  ^ "    device const half* bias_or_scale [[buffer("
  ^ string_of_int (3 + input_shift) ^ ")]],\n"
  ^ extra_argument
  ^ "    device " ^ value_type ^ "* output [[buffer("
  ^ string_of_int output_buffer ^ ")]],\n"
  ^ "    constant Q8Params& params [[buffer(" ^ string_of_int parameter_buffer
  ^ ")]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]],\n"
  ^ "    uint simdgroup [[simdgroup_index_in_threadgroup]],\n"
  ^ "    uint3 threadgroup_position [[threadgroup_position_in_grid]]) {\n"
  ^ "  const uint col = threadgroup_position.x * 8 + simdgroup;\n"
  ^ "  if (params.m != 1 || col >= params.n) return;\n"
  ^ "  const " ^ scale_type ^ " channel_scale = scale[col];\n"
  ^ "  const uint weight_base = col * params.k;\n"
  ^ "  float acc = 0.0f;\n"
  ^ "  uint scalar_start = 0;\n"
  ^ "  if ((weight_base & 3u) == 0) {\n"
  ^ "    const uint vectorized = params.k & ~3u;\n"
  ^ "    for (uint inner = lane * 4; inner < vectorized; inner += 128) {\n"
  ^ input_vector_load
  ^ "      device const char4* weight_vector =\n"
  ^ "          reinterpret_cast<device const char4*>(weight + weight_base + inner);\n"
  ^ "      const char4 weight_values = *weight_vector;\n"
  ^ "      const " ^ vector_type ^ " dequantized_weights = " ^ vector_type
  ^ "(weight_values) * channel_scale;\n"
  ^ "      acc += dot(float4(input_values), float4(dequantized_weights));\n"
  ^ "    }\n"
  ^ "    scalar_start = vectorized;\n"
  ^ "  }\n"
  ^ "  for (uint inner = scalar_start + lane; inner < params.k; inner += 32)\n"
  ^ "    acc += float(" ^ input_scalar ^ ") *\n"
  ^ "           float(" ^ weight_cast
  ^ "(weight[weight_base + inner]) * channel_scale);\n"
  ^ "  acc = simd_sum(acc);\n"
  ^ "  if (lane == 0) {\n"
  ^ "    if (params.has_bias != 0) acc += float(bias_or_scale[col]);\n"
  ^ "    output[col] = " ^ store_value ^ ";\n"
  ^ "  }\n"
  ^ "}\n"

let q8_gemv_simd_kernel =
  q8_gemv_simd_kernel_with_input ~input_mode:Direct

let q8_gemv_pair_simd_kernel_with_input ~input_mode ~extra_argument
    ~output_buffer ~parameter_buffer ~name ~value_type ~vector_type
    ~weight_cast ~scale_type ~store_value =
  let input_signature, input_shift = q8_input_signature ~value_type input_mode in
  let input_vector_load =
    q8_input_vector_load ~value_type ~vector_type ~offset:"inner"
      ~values_name:"input_values" input_mode
  in
  let input_scalar = q8_input_scalar ~value_type "inner" input_mode in
  "kernel void " ^ name ^ "(\n"
  ^ input_signature
  ^ "    device const char* weight [[buffer(" ^ string_of_int (1 + input_shift)
  ^ ")]],\n"
  ^ "    device const half* scale [[buffer(" ^ string_of_int (2 + input_shift)
  ^ ")]],\n"
  ^ "    device const half* bias_or_scale [[buffer("
  ^ string_of_int (3 + input_shift) ^ ")]],\n"
  ^ extra_argument
  ^ "    device " ^ value_type ^ "* output [[buffer("
  ^ string_of_int output_buffer ^ ")]],\n"
  ^ "    constant Q8Params& params [[buffer(" ^ string_of_int parameter_buffer
  ^ ")]],\n"
  ^ "    uint lane [[thread_index_in_simdgroup]],\n"
  ^ "    uint simdgroup [[simdgroup_index_in_threadgroup]],\n"
  ^ "    uint3 threadgroup_position [[threadgroup_position_in_grid]]) {\n"
  ^ "  const uint col = threadgroup_position.x * 16 + simdgroup * 2;\n"
  ^ "  const uint next_col = col + 1;\n"
  ^ "  if (params.m != 1 || col >= params.n) return;\n"
  ^ "  const bool next_valid = next_col < params.n;\n"
  ^ "  const " ^ scale_type ^ " channel_scale = scale[col];\n"
  ^ "  const " ^ scale_type ^ " next_channel_scale =\n"
  ^ "      next_valid ? scale[next_col] : " ^ scale_type ^ "(0);\n"
  ^ "  const uint weight_base = col * params.k;\n"
  ^ "  const uint next_weight_base = next_col * params.k;\n"
  ^ "  float acc = 0.0f;\n"
  ^ "  float next_acc = 0.0f;\n"
  ^ "  uint scalar_start = 0;\n"
  ^ "  if ((weight_base & 3u) == 0 &&\n"
  ^ "      (!next_valid || (next_weight_base & 3u) == 0)) {\n"
  ^ "    const uint vectorized = params.k & ~3u;\n"
  ^ "    for (uint inner = lane * 4; inner < vectorized; inner += 128) {\n"
  ^ input_vector_load
  ^ "      device const char4* weight_vector =\n"
  ^ "          reinterpret_cast<device const char4*>(weight + weight_base + inner);\n"
  ^ "      const char4 weight_values = *weight_vector;\n"
  ^ "      const char4 next_weight_values = next_valid\n"
  ^ "          ? *reinterpret_cast<device const char4*>(\n"
  ^ "                weight + next_weight_base + inner)\n"
  ^ "          : char4(0);\n"
  ^ "      const " ^ vector_type ^ " dequantized_weights = " ^ vector_type
  ^ "(weight_values) * channel_scale;\n"
  ^ "      const " ^ vector_type ^ " next_dequantized_weights = " ^ vector_type
  ^ "(next_weight_values) * next_channel_scale;\n"
  ^ "      acc += dot(float4(input_values), float4(dequantized_weights));\n"
  ^ "      next_acc += dot(float4(input_values),\n"
  ^ "                      float4(next_dequantized_weights));\n"
  ^ "    }\n"
  ^ "    scalar_start = vectorized;\n"
  ^ "  }\n"
  ^ "  for (uint inner = scalar_start + lane; inner < params.k; inner += 32) {\n"
  ^ "    const float input_value = float(" ^ input_scalar ^ ");\n"
  ^ "    acc += input_value * float(" ^ weight_cast
  ^ "(weight[weight_base + inner]) * channel_scale);\n"
  ^ "    if (next_valid)\n"
  ^ "      next_acc += input_value * float(" ^ weight_cast
  ^ "(weight[next_weight_base + inner]) * next_channel_scale);\n"
  ^ "  }\n"
  ^ "  acc = simd_sum(acc);\n"
  ^ "  next_acc = simd_sum(next_acc);\n"
  ^ "  if (lane == 0) {\n"
  ^ "    if (params.has_bias != 0) acc += float(bias_or_scale[col]);\n"
  ^ "    output[col] = " ^ store_value ^ ";\n"
  ^ "    if (next_valid) {\n"
  ^ "      const uint col = next_col;\n"
  ^ "      float acc = next_acc;\n"
  ^ "      if (params.has_bias != 0) acc += float(bias_or_scale[col]);\n"
  ^ "      output[col] = " ^ store_value ^ ";\n"
  ^ "    }\n"
  ^ "  }\n"
  ^ "}\n"

let q8_gemv_pair_simd_kernel =
  q8_gemv_pair_simd_kernel_with_input ~input_mode:Direct

let q8_dequant_kernel ~name ~value_type ~weight_cast ~scale_cast =
  "kernel void " ^ name ^ "(\n"
  ^ "    device const char* weight [[buffer(0)]],\n"
  ^ "    device const half* scale [[buffer(1)]],\n"
  ^ "    device " ^ value_type ^ "* output [[buffer(2)]],\n"
  ^ "    constant Q8Params& params [[buffer(3)]],\n"
  ^ "    uint2 gid [[thread_position_in_grid]]) {\n"
  ^ "  const uint col = gid.y;\n"
  ^ "  const uint inner = gid.x;\n"
  ^ "  if (col < params.n && inner < params.k) {\n"
  ^ "    output[col * params.k + inner] = " ^ weight_cast
  ^ "(weight[col * params.k + inner]) * " ^ scale_cast ^ "(scale[col]);\n"
  ^ "  }\n"
  ^ "}\n"

let q8_dual_linear_kernel ~name ~value_type ~weight_cast ~store_value
    ~has_bias ~silu_first =
  let weight2_buffer, scale2_buffer, bias1, bias2, output1_buffer,
      output2_buffer, parameter_buffer =
    if has_bias then 4, 5, "    device const half* bias1 [[buffer(3)]],\n",
      "    device const half* bias2 [[buffer(6)]],\n", 7, 8, 9
    else 3, 4, "", "", 5, 6, 7
  in
  "kernel void " ^ name ^ "(\n"
  ^ "    device const " ^ value_type ^ "* input [[buffer(0)]],\n"
  ^ "    device const char* weight1 [[buffer(1)]],\n"
  ^ "    device const half* scale1 [[buffer(2)]],\n"
  ^ bias1
  ^ "    device const char* weight2 [[buffer(" ^ string_of_int weight2_buffer
  ^ ")]],\n"
  ^ "    device const half* scale2 [[buffer(" ^ string_of_int scale2_buffer
  ^ ")]],\n"
  ^ bias2
  ^ "    device " ^ value_type ^ "* output1 [[buffer("
  ^ string_of_int output1_buffer ^ ")]],\n"
  ^ "    device " ^ value_type ^ "* output2 [[buffer("
  ^ string_of_int output2_buffer ^ ")]],\n"
  ^ "    constant Q8DualParams& params [[buffer("
  ^ string_of_int parameter_buffer ^ ")]],\n"
  ^ "    uint2 gid [[thread_position_in_grid]]) {\n"
  ^ "  const uint row = gid.y;\n"
  ^ "  const uint col = gid.x;\n"
  ^ "  const uint width = params.n1 + params.n2;\n"
  ^ "  if (row >= params.m || col >= width) return;\n"
  ^ "  const bool second = col >= params.n1;\n"
  ^ "  const uint channel = second ? col - params.n1 : col;\n"
  ^ "  device const char* weight = second ? weight2 : weight1;\n"
  ^ "  device const half* scale = second ? scale2 : scale1;\n"
  ^ "  float accumulator = 0.0f;\n"
  ^ "  for (uint inner = 0; inner < params.k; ++inner)\n"
  ^ "    accumulator += float(input[row * params.k + inner])\n"
  ^ "        * float(" ^ weight_cast
  ^ "(weight[channel * params.k + inner])) * float(scale[channel]);\n"
  ^ (if has_bias then
       "  if (second) accumulator += float(bias2[channel]);\n"
     ^ "  else accumulator += float(bias1[channel]);\n"
     else "")
  ^ (if silu_first then
       "  if (!second) accumulator = accumulator / (1.0f + exp(-accumulator));\n"
     else "")
  ^ "  if (second) output2[row * params.n2 + channel] = " ^ store_value ^ ";\n"
  ^ "  else output1[row * params.n1 + channel] = " ^ store_value ^ ";\n"
  ^ "}\n"

let q8_dual_linear_source =
  "\nstruct Q8DualParams { uint m; uint n1; uint n2; uint k; };\n\n"
  ^ q8_dual_linear_kernel ~name:"llmopt_q8_dual_linear_f16" ~value_type:"half"
      ~weight_cast:"half" ~store_value:"half(accumulator)" ~has_bias:false
      ~silu_first:false
  ^ q8_dual_linear_kernel ~name:"llmopt_q8_dual_linear_f16_bias"
      ~value_type:"half" ~weight_cast:"half" ~store_value:"half(accumulator)"
      ~has_bias:true ~silu_first:false
  ^ q8_dual_linear_kernel ~name:"llmopt_q8_dual_linear_f32" ~value_type:"float"
      ~weight_cast:"float" ~store_value:"accumulator" ~has_bias:false
      ~silu_first:false
  ^ q8_dual_linear_kernel ~name:"llmopt_q8_dual_linear_f32_bias"
      ~value_type:"float" ~weight_cast:"float" ~store_value:"accumulator"
      ~has_bias:true ~silu_first:false
  ^ q8_dual_linear_kernel ~name:"llmopt_q8_dual_linear_silu_f16"
      ~value_type:"half" ~weight_cast:"half" ~store_value:"half(accumulator)"
      ~has_bias:false ~silu_first:true
  ^ q8_dual_linear_kernel ~name:"llmopt_q8_dual_linear_silu_f16_bias"
      ~value_type:"half" ~weight_cast:"half" ~store_value:"half(accumulator)"
      ~has_bias:true ~silu_first:true
  ^ q8_dual_linear_kernel ~name:"llmopt_q8_dual_linear_silu_f32"
      ~value_type:"float" ~weight_cast:"float" ~store_value:"accumulator"
      ~has_bias:false ~silu_first:true
  ^ q8_dual_linear_kernel ~name:"llmopt_q8_dual_linear_silu_f32_bias"
      ~value_type:"float" ~weight_cast:"float" ~store_value:"accumulator"
      ~has_bias:true ~silu_first:true

let q8_qkv_linear_kernel ~name ~value_type ~weight_cast ~store_value ~has_bias =
  let weight_k_buffer, scale_k_buffer, bias_q, bias_k, weight_v_buffer,
      scale_v_buffer, bias_v, output_q_buffer, output_k_buffer, output_v_buffer,
      parameter_buffer =
    if has_bias then
      4, 5, "    device const half* bias_q [[buffer(3)]],\n",
      "    device const half* bias_k [[buffer(6)]],\n", 7, 8,
      "    device const half* bias_v [[buffer(9)]],\n", 10, 11, 12, 13
    else 3, 4, "", "", 5, 6, "", 7, 8, 9, 10
  in
  "kernel void " ^ name ^ "(\n"
  ^ "    device const " ^ value_type ^ "* input [[buffer(0)]],\n"
  ^ "    device const char* weight_q [[buffer(1)]],\n"
  ^ "    device const half* scale_q [[buffer(2)]],\n"
  ^ bias_q
  ^ "    device const char* weight_k [[buffer(" ^ string_of_int weight_k_buffer
  ^ ")]],\n"
  ^ "    device const half* scale_k [[buffer(" ^ string_of_int scale_k_buffer
  ^ ")]],\n"
  ^ bias_k
  ^ "    device const char* weight_v [[buffer(" ^ string_of_int weight_v_buffer
  ^ ")]],\n"
  ^ "    device const half* scale_v [[buffer(" ^ string_of_int scale_v_buffer
  ^ ")]],\n"
  ^ bias_v
  ^ "    device " ^ value_type ^ "* output_q [[buffer("
  ^ string_of_int output_q_buffer ^ ")]],\n"
  ^ "    device " ^ value_type ^ "* output_k [[buffer("
  ^ string_of_int output_k_buffer ^ ")]],\n"
  ^ "    device " ^ value_type ^ "* output_v [[buffer("
  ^ string_of_int output_v_buffer ^ ")]],\n"
  ^ "    constant Q8QkvParams& params [[buffer("
  ^ string_of_int parameter_buffer ^ ")]],\n"
  ^ "    uint2 gid [[thread_position_in_grid]]) {\n"
  ^ "  const uint row = gid.y;\n"
  ^ "  const uint col = gid.x;\n"
  ^ "  const uint kv_end = params.n_q + params.n_kv;\n"
  ^ "  const uint width = params.n_q + 2 * params.n_kv;\n"
  ^ "  if (row >= params.m || col >= width) return;\n"
  ^ "  const uint group = col < params.n_q ? 0 : (col < kv_end ? 1 : 2);\n"
  ^ "  const uint channel = group == 0 ? col\n"
  ^ "      : (group == 1 ? col - params.n_q : col - kv_end);\n"
  ^ "  device const char* weight = group == 0 ? weight_q\n"
  ^ "      : (group == 1 ? weight_k : weight_v);\n"
  ^ "  device const half* scale = group == 0 ? scale_q\n"
  ^ "      : (group == 1 ? scale_k : scale_v);\n"
  ^ "  float accumulator = 0.0f;\n"
  ^ "  for (uint inner = 0; inner < params.k; ++inner)\n"
  ^ "    accumulator += float(input[row * params.k + inner])\n"
  ^ "        * float(" ^ weight_cast
  ^ "(weight[channel * params.k + inner])) * float(scale[channel]);\n"
  ^ (if has_bias then
       "  if (group == 0) accumulator += float(bias_q[channel]);\n"
     ^ "  else if (group == 1) accumulator += float(bias_k[channel]);\n"
     ^ "  else accumulator += float(bias_v[channel]);\n"
     else "")
  ^ "  if (group == 0) output_q[row * params.n_q + channel] = "
  ^ store_value ^ ";\n"
  ^ "  else if (group == 1) output_k[row * params.n_kv + channel] = "
  ^ store_value ^ ";\n"
  ^ "  else output_v[row * params.n_kv + channel] = " ^ store_value ^ ";\n"
  ^ "}\n"

let q8_qkv_linear_source =
  "\nstruct Q8QkvParams { uint m; uint n_q; uint n_kv; uint k; };\n\n"
  ^ q8_qkv_linear_kernel ~name:"llmopt_q8_qkv_linear_f16" ~value_type:"half"
      ~weight_cast:"half" ~store_value:"half(accumulator)" ~has_bias:false
  ^ q8_qkv_linear_kernel ~name:"llmopt_q8_qkv_linear_f16_bias"
      ~value_type:"half" ~weight_cast:"half" ~store_value:"half(accumulator)"
      ~has_bias:true
  ^ q8_qkv_linear_kernel ~name:"llmopt_q8_qkv_linear_f32" ~value_type:"float"
      ~weight_cast:"float" ~store_value:"accumulator" ~has_bias:false
  ^ q8_qkv_linear_kernel ~name:"llmopt_q8_qkv_linear_f32_bias"
      ~value_type:"float" ~weight_cast:"float" ~store_value:"accumulator"
      ~has_bias:true

let q8_linear_add_norm_kernel ~name ~value_type ~weight_cast ~extra_output =
  let value_name = name ^ "_value" in
  let weight_value =
    "float(" ^ weight_cast ^ "(weight[col * params.k + inner]) * "
    ^ weight_cast ^ "(scale[col]))"
  in
  "inline float " ^ value_name ^ "(\n"
  ^ "    uint row, uint col, device const " ^ value_type ^ "* input,\n"
  ^ "    device const char* weight, device const half* scale,\n"
  ^ "    device const " ^ value_type ^ "* residual,\n"
  ^ "    constant Q8LinearAddNormParams& params) {\n"
  ^ "  float accumulator = 0.0f;\n"
  ^ "  for (uint inner = 0; inner < params.k; ++inner)\n"
  ^ "    accumulator += float(input[row * params.k + inner]) * "
  ^ weight_value ^ ";\n"
  ^ "  return accumulator + float(residual[row * params.n + col]);\n"
  ^ "}\n\n"
  ^ "kernel void " ^ name ^ "(\n"
  ^ "    device const " ^ value_type ^ "* input [[buffer(0)]],\n"
  ^ "    device const char* weight [[buffer(1)]],\n"
  ^ "    device const half* scale [[buffer(2)]],\n"
  ^ "    device const " ^ value_type ^ "* residual [[buffer(3)]],\n"
  ^ "    device const half* norm_weight [[buffer(4)]],\n"
  ^ "    device half* output [[buffer(5)]],\n"
  ^ (if extra_output then
       "    device half* residual_output [[buffer(6)]],\n"
       ^ "    constant Q8LinearAddNormParams& params [[buffer(7)]],\n"
     else "    constant Q8LinearAddNormParams& params [[buffer(6)]],\n")
  ^ "    uint lane [[thread_index_in_simdgroup]],\n"
  ^ "    uint simdgroup [[simdgroup_index_in_threadgroup]],\n"
  ^ "    uint3 group_position [[threadgroup_position_in_grid]]) {\n"
  ^ "  const uint row = group_position.x;\n"
  ^ "  if (row >= params.m) return;\n"
  ^ "  const uint thread_index = simdgroup * 32 + lane;\n"
  ^ "  float square_sum = 0.0f;\n"
  ^ "  for (uint col = thread_index; col < params.n; col += 256) {\n"
  ^ "    const float value = " ^ value_name
  ^ "(row, col, input, weight, scale, residual, params);\n"
  ^ "    square_sum += value * value;\n"
  ^ "  }\n"
  ^ "  threadgroup float partial_sums[8];\n"
  ^ "  threadgroup float inverse;\n"
  ^ "  square_sum = simd_sum(square_sum);\n"
  ^ "  if (lane == 0) partial_sums[simdgroup] = square_sum;\n"
  ^ "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  if (simdgroup == 0 && lane == 0) {\n"
  ^ "    float total = 0.0f;\n"
  ^ "    for (uint group = 0; group < 8; ++group)\n"
  ^ "      total += partial_sums[group];\n"
  ^ "    inverse = rsqrt(total / float(params.n) + params.epsilon);\n"
  ^ "  }\n"
  ^ "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  for (uint col = thread_index; col < params.n; col += 256) {\n"
  ^ "    const float value = " ^ value_name
  ^ "(row, col, input, weight, scale, residual, params);\n"
  ^ (if extra_output then
       "    residual_output[row * params.n + col] = half(value);\n"
     else "")
  ^ "    output[row * params.n + col] = half(value * inverse * float(norm_weight[col]));\n"
  ^ "  }\n"
  ^ "}\n"

let q8_linear_add_norm_source =
  "\nstruct Q8LinearAddNormParams { uint m; uint n; uint k; float epsilon; };\n\n"
  ^ q8_linear_add_norm_kernel ~name:"llmopt_q8_linear_add_norm_f16"
      ~value_type:"half" ~weight_cast:"half" ~extra_output:false
  ^ q8_linear_add_norm_kernel ~name:"llmopt_q8_linear_add_norm_f32"
      ~value_type:"float" ~weight_cast:"float" ~extra_output:false
  ^ q8_linear_add_norm_kernel ~name:"llmopt_q8_linear_add_norm_extra_f16"
      ~value_type:"half" ~weight_cast:"half" ~extra_output:true
  ^ q8_linear_add_norm_kernel ~name:"llmopt_q8_linear_add_norm_extra_f32"
      ~value_type:"float" ~weight_cast:"float" ~extra_output:true

let q8_lm_head_argmax_kernel ~name ~value_type ~weight_cast =
  "kernel void " ^ name ^ "(\n"
  ^ "    device const " ^ value_type ^ "* input [[buffer(0)]],\n"
  ^ "    device const half* norm_weight [[buffer(1)]],\n"
  ^ "    device const char* weight [[buffer(2)]],\n"
  ^ "    device const half* scale [[buffer(3)]],\n"
  ^ "    device uint* token_ids [[buffer(4)]],\n"
  ^ "    constant Q8LmHeadParams& params [[buffer(5)]],\n"
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
  ^ "  threadgroup float best_values[256];\n"
  ^ "  threadgroup uint best_indices[256];\n"
  ^ "  float best_value = -3.402823466e+38f;\n"
  ^ "  uint best_index = 0;\n"
  ^ "  for (uint col = tid; col < params.n; col += 256) {\n"
  ^ "    float accumulator = 0.0f;\n"
  ^ "    for (uint inner = 0; inner < params.k; ++inner) {\n"
  ^ "      float normalized = float(input[row * params.k + inner]) *\n"
  ^ "          float(norm_weight[inner]) * inverse;\n"
  ^ "      accumulator += normalized * float(" ^ weight_cast
  ^ "(weight[col * params.k + inner])) * float(scale[col]);\n"
  ^ "    }\n"
  ^ "    if (accumulator > best_value ||\n"
  ^ "        (accumulator == best_value && col < best_index)) {\n"
  ^ "      best_value = accumulator;\n"
  ^ "      best_index = col;\n"
  ^ "    }\n"
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

let q8_lm_head_argmax_source =
  "\nstruct Q8LmHeadParams { uint m; uint n; uint k; float epsilon; };\n\n"
  ^ q8_lm_head_argmax_kernel ~name:"llmopt_q8_lm_head_argmax_f16"
      ~value_type:"half" ~weight_cast:"half"
  ^ q8_lm_head_argmax_kernel ~name:"llmopt_q8_lm_head_argmax_f32"
      ~value_type:"float" ~weight_cast:"float"

let q8_source =
  "\nconstant uint Q8_TILE = 64;\n\n"
  ^ "struct Q8Params { uint m; uint n; uint k; uint has_bias; };\n\n"
  ^ q8_kernel
      ~extra_argument:"" ~output_buffer:4 ~parameter_buffer:5
      ~name:"llmopt_q8_linear"
      ~value_type:"half"
      ~vector_type:"half4"
      ~alignment:"8"
      ~weight_cast:"half"
      ~scale_type:"half"
      ~scale_zero:"half(0.0h)"
      ~zero_value:"half(0.0h)"
      ~store_value:"half(acc)"
  ^ q8_kernel
      ~extra_argument:"" ~output_buffer:4 ~parameter_buffer:5
      ~name:"llmopt_q8_linear_f32"
      ~value_type:"float"
      ~vector_type:"float4"
      ~alignment:"16"
      ~weight_cast:"float"
      ~scale_type:"float"
      ~scale_zero:"0.0f"
      ~zero_value:"0.0f"
      ~store_value:"acc"
  ^ q8_kernel
      ~extra_argument:"" ~output_buffer:4 ~parameter_buffer:5
      ~name:"llmopt_q8_linear_silu"
      ~value_type:"half"
      ~vector_type:"half4"
      ~alignment:"8"
      ~weight_cast:"half"
      ~scale_type:"half"
      ~scale_zero:"half(0.0h)"
      ~zero_value:"half(0.0h)"
      ~store_value:
        "half(float(half(acc)) / (1.0f + exp(-float(half(acc)))))"
  ^ q8_kernel
      ~extra_argument:"" ~output_buffer:4 ~parameter_buffer:5
      ~name:"llmopt_q8_linear_silu_f32"
      ~value_type:"float"
      ~vector_type:"float4"
      ~alignment:"16"
      ~weight_cast:"float"
      ~scale_type:"float"
      ~scale_zero:"0.0f"
      ~zero_value:"0.0f"
      ~store_value:"acc / (1.0f + exp(-acc))"
  ^ q8_kernel
      ~extra_argument:"    device const half* residual [[buffer(4)]],\n"
      ~output_buffer:5 ~parameter_buffer:6
      ~name:"llmopt_q8_linear_add"
      ~value_type:"half"
      ~vector_type:"half4"
      ~alignment:"8"
      ~weight_cast:"half"
      ~scale_type:"half"
      ~scale_zero:"half(0.0h)"
      ~zero_value:"half(0.0h)"
      ~store_value:"half(half(acc) + residual[row * params.n + col])"
  ^ q8_kernel
      ~extra_argument:"    device const float* residual [[buffer(4)]],\n"
      ~output_buffer:5 ~parameter_buffer:6
      ~name:"llmopt_q8_linear_add_f32"
      ~value_type:"float"
      ~vector_type:"float4"
      ~alignment:"16"
      ~weight_cast:"float"
      ~scale_type:"float"
      ~scale_zero:"0.0f"
      ~zero_value:"0.0f"
      ~store_value:"acc + residual[row * params.n + col]"
  ^ q8_gemv_kernel
      ~extra_argument:"" ~output_buffer:4 ~parameter_buffer:5
      ~name:"llmopt_q8_gemv"
      ~value_type:"half"
      ~vector_type:"half4"
      ~weight_cast:"half"
      ~scale_type:"half"
      ~store_value:"half(acc)"
  ^ q8_gemv_kernel
      ~extra_argument:"" ~output_buffer:4 ~parameter_buffer:5
      ~name:"llmopt_q8_gemv_f32"
      ~value_type:"float"
      ~vector_type:"float4"
      ~weight_cast:"float"
      ~scale_type:"float"
      ~store_value:"acc"
  ^ q8_gemv_kernel
      ~extra_argument:"" ~output_buffer:4 ~parameter_buffer:5
      ~name:"llmopt_q8_gemv_silu"
      ~value_type:"half"
      ~vector_type:"half4"
      ~weight_cast:"half"
      ~scale_type:"half"
      ~store_value:
        "half(float(half(acc)) / (1.0f + exp(-float(half(acc)))))"
  ^ q8_gemv_kernel
      ~extra_argument:"" ~output_buffer:4 ~parameter_buffer:5
      ~name:"llmopt_q8_gemv_silu_f32"
      ~value_type:"float"
      ~vector_type:"float4"
      ~weight_cast:"float"
      ~scale_type:"float"
      ~store_value:"acc / (1.0f + exp(-acc))"
  ^ q8_gemv_kernel
      ~extra_argument:"    device const half* residual [[buffer(4)]],\n"
      ~output_buffer:5 ~parameter_buffer:6
      ~name:"llmopt_q8_gemv_add"
      ~value_type:"half"
      ~vector_type:"half4"
      ~weight_cast:"half"
      ~scale_type:"half"
      ~store_value:"half(half(acc) + residual[col])"
  ^ q8_gemv_kernel
      ~extra_argument:"    device const float* residual [[buffer(4)]],\n"
      ~output_buffer:5 ~parameter_buffer:6
      ~name:"llmopt_q8_gemv_add_f32"
      ~value_type:"float"
      ~vector_type:"float4"
      ~weight_cast:"float"
      ~scale_type:"float"
      ~store_value:"acc + residual[col]"
  ^ q8_kernel_with_input
      ~input_mode:Product
      ~extra_argument:"    device const half* residual [[buffer(5)]],\n"
      ~output_buffer:6 ~parameter_buffer:7
      ~name:"llmopt_q8_linear_mul_add"
      ~value_type:"half"
      ~vector_type:"half4"
      ~alignment:"8"
      ~weight_cast:"half"
      ~scale_type:"half"
      ~scale_zero:"half(0.0h)"
      ~zero_value:"half(0.0h)"
      ~store_value:"half(half(acc) + residual[row * params.n + col])"
  ^ q8_kernel_with_input
      ~input_mode:Product
      ~extra_argument:"    device const float* residual [[buffer(5)]],\n"
      ~output_buffer:6 ~parameter_buffer:7
      ~name:"llmopt_q8_linear_mul_add_f32"
      ~value_type:"float"
      ~vector_type:"float4"
      ~alignment:"16"
      ~weight_cast:"float"
      ~scale_type:"float"
      ~scale_zero:"0.0f"
      ~zero_value:"0.0f"
      ~store_value:"acc + residual[row * params.n + col]"
  ^ q8_gemv_kernel_with_input
      ~input_mode:Product
      ~extra_argument:"    device const half* residual [[buffer(5)]],\n"
      ~output_buffer:6 ~parameter_buffer:7
      ~name:"llmopt_q8_gemv_mul_add"
      ~value_type:"half"
      ~vector_type:"half4"
      ~weight_cast:"half"
      ~scale_type:"half"
      ~store_value:"half(half(acc) + residual[col])"
  ^ q8_gemv_kernel_with_input
      ~input_mode:Product
      ~extra_argument:"    device const float* residual [[buffer(5)]],\n"
      ~output_buffer:6 ~parameter_buffer:7
      ~name:"llmopt_q8_gemv_mul_add_f32"
      ~value_type:"float"
      ~vector_type:"float4"
      ~weight_cast:"float"
      ~scale_type:"float"
      ~store_value:"acc + residual[col]"
  ^ q8_gemv_simd_kernel
      ~extra_argument:"" ~output_buffer:4 ~parameter_buffer:5
      ~name:"llmopt_q8_gemv_simd"
      ~value_type:"half"
      ~vector_type:"half4"
      ~weight_cast:"half"
      ~scale_type:"half"
      ~store_value:"half(acc)"
  ^ q8_gemv_simd_kernel
      ~extra_argument:"" ~output_buffer:4 ~parameter_buffer:5
      ~name:"llmopt_q8_gemv_simd_f32"
      ~value_type:"float"
      ~vector_type:"float4"
      ~weight_cast:"float"
      ~scale_type:"float"
      ~store_value:"acc"
  ^ q8_gemv_simd_kernel
      ~extra_argument:"" ~output_buffer:4 ~parameter_buffer:5
      ~name:"llmopt_q8_gemv_silu_simd"
      ~value_type:"half"
      ~vector_type:"half4"
      ~weight_cast:"half"
      ~scale_type:"half"
      ~store_value:
        "half(float(half(acc)) / (1.0f + exp(-float(half(acc)))))"
  ^ q8_gemv_simd_kernel
      ~extra_argument:"" ~output_buffer:4 ~parameter_buffer:5
      ~name:"llmopt_q8_gemv_silu_simd_f32"
      ~value_type:"float"
      ~vector_type:"float4"
      ~weight_cast:"float"
      ~scale_type:"float"
      ~store_value:"acc / (1.0f + exp(-acc))"
  ^ q8_gemv_simd_kernel
      ~extra_argument:"    device const half* residual [[buffer(4)]],\n"
      ~output_buffer:5 ~parameter_buffer:6
      ~name:"llmopt_q8_gemv_add_simd"
      ~value_type:"half"
      ~vector_type:"half4"
      ~weight_cast:"half"
      ~scale_type:"half"
      ~store_value:"half(half(acc) + residual[col])"
  ^ q8_gemv_simd_kernel
      ~extra_argument:"    device const float* residual [[buffer(4)]],\n"
      ~output_buffer:5 ~parameter_buffer:6
      ~name:"llmopt_q8_gemv_add_simd_f32"
      ~value_type:"float"
      ~vector_type:"float4"
      ~weight_cast:"float"
      ~scale_type:"float"
      ~store_value:"acc + residual[col]"
  ^ q8_gemv_simd_kernel_with_input
      ~input_mode:Product
      ~extra_argument:"    device const half* residual [[buffer(5)]],\n"
      ~output_buffer:6 ~parameter_buffer:7
      ~name:"llmopt_q8_gemv_mul_add_simd"
      ~value_type:"half"
      ~vector_type:"half4"
      ~weight_cast:"half"
      ~scale_type:"half"
      ~store_value:"half(half(acc) + residual[col])"
  ^ q8_gemv_simd_kernel_with_input
      ~input_mode:Product
      ~extra_argument:"    device const float* residual [[buffer(5)]],\n"
      ~output_buffer:6 ~parameter_buffer:7
      ~name:"llmopt_q8_gemv_mul_add_simd_f32"
      ~value_type:"float"
      ~vector_type:"float4"
      ~weight_cast:"float"
      ~scale_type:"float"
      ~store_value:"acc + residual[col]"
  ^ q8_gemv_pair_simd_kernel
      ~extra_argument:"" ~output_buffer:4 ~parameter_buffer:5
      ~name:"llmopt_q8_gemv_pair_simd"
      ~value_type:"half"
      ~vector_type:"half4"
      ~weight_cast:"half"
      ~scale_type:"half"
      ~store_value:"half(acc)"
  ^ q8_gemv_pair_simd_kernel
      ~extra_argument:"" ~output_buffer:4 ~parameter_buffer:5
      ~name:"llmopt_q8_gemv_pair_simd_f32"
      ~value_type:"float"
      ~vector_type:"float4"
      ~weight_cast:"float"
      ~scale_type:"float"
      ~store_value:"acc"
  ^ q8_gemv_pair_simd_kernel
      ~extra_argument:"" ~output_buffer:4 ~parameter_buffer:5
      ~name:"llmopt_q8_gemv_silu_pair_simd"
      ~value_type:"half"
      ~vector_type:"half4"
      ~weight_cast:"half"
      ~scale_type:"half"
      ~store_value:
        "half(float(half(acc)) / (1.0f + exp(-float(half(acc)))))"
  ^ q8_gemv_pair_simd_kernel
      ~extra_argument:"" ~output_buffer:4 ~parameter_buffer:5
      ~name:"llmopt_q8_gemv_silu_pair_simd_f32"
      ~value_type:"float"
      ~vector_type:"float4"
      ~weight_cast:"float"
      ~scale_type:"float"
      ~store_value:"acc / (1.0f + exp(-acc))"
  ^ q8_gemv_pair_simd_kernel
      ~extra_argument:"    device const half* residual [[buffer(4)]],\n"
      ~output_buffer:5 ~parameter_buffer:6
      ~name:"llmopt_q8_gemv_add_pair_simd"
      ~value_type:"half"
      ~vector_type:"half4"
      ~weight_cast:"half"
      ~scale_type:"half"
      ~store_value:"half(half(acc) + residual[col])"
  ^ q8_gemv_pair_simd_kernel
      ~extra_argument:"    device const float* residual [[buffer(4)]],\n"
      ~output_buffer:5 ~parameter_buffer:6
      ~name:"llmopt_q8_gemv_add_pair_simd_f32"
      ~value_type:"float"
      ~vector_type:"float4"
      ~weight_cast:"float"
      ~scale_type:"float"
      ~store_value:"acc + residual[col]"
  ^ q8_gemv_pair_simd_kernel_with_input
      ~input_mode:Product
      ~extra_argument:"    device const half* residual [[buffer(5)]],\n"
      ~output_buffer:6 ~parameter_buffer:7
      ~name:"llmopt_q8_gemv_mul_add_pair_simd"
      ~value_type:"half"
      ~vector_type:"half4"
      ~weight_cast:"half"
      ~scale_type:"half"
      ~store_value:"half(half(acc) + residual[col])"
  ^ q8_gemv_pair_simd_kernel_with_input
      ~input_mode:Product
      ~extra_argument:"    device const float* residual [[buffer(5)]],\n"
      ~output_buffer:6 ~parameter_buffer:7
      ~name:"llmopt_q8_gemv_mul_add_pair_simd_f32"
      ~value_type:"float"
      ~vector_type:"float4"
      ~weight_cast:"float"
      ~scale_type:"float"
      ~store_value:"acc + residual[col]"
  ^ q8_dequant_kernel
      ~name:"llmopt_q8_dequantize"
      ~value_type:"half"
      ~weight_cast:"half"
      ~scale_cast:"half"
  ^ q8_dequant_kernel
      ~name:"llmopt_q8_dequantize_f32"
      ~value_type:"float"
      ~weight_cast:"float"
      ~scale_cast:"float"

let q8_parameterized_tiles =
  [ (32, 8, 64); (8, 32, 64); (32, 32, 64); (16, 16, 128);
    (32, 8, 128); (8, 32, 128); (32, 32, 128) ]

let q8_parameterized_source =
  q8_parameterized_tiles
  |> List.map (fun (tile_m, tile_n, tile_k) ->
         q8_kernel_parameterized ~tile_m ~tile_n ~tile_k
         ^ q8_kernel_parameterized_f32 ~tile_m ~tile_n ~tile_k)
  |> String.concat "\n"

let q8_source = q8_source ^ "\n" ^ q8_dual_linear_source ^ "\n"
  ^ q8_qkv_linear_source ^ "\n"
  ^ q8_linear_add_norm_source ^ "\n"
  ^ q8_lm_head_argmax_source ^ "\n"
  ^ q8_parameterized_source

let q8_parameterized_entries =
  q8_parameterized_tiles
  |> List.map (fun (tile_m, tile_n, tile_k) ->
         [ kernel_entry_with_threadgroup_and_tile
             ~tile:(tile_m, tile_n, tile_k)
             ~threadgroup:(tile_n, tile_m, 1)
             ~name:(q8_parameterized_name ~tile_m ~tile_n ~tile_k)
             ~operation:Kernel_abi.Operation.Q8_linear
             ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
           kernel_entry_with_threadgroup_and_tile
             ~tile:(tile_m, tile_n, tile_k)
             ~threadgroup:(tile_n, tile_m, 1)
             ~name:(q8_parameterized_name ~tile_m ~tile_n ~tile_k ^ "_f32")
             ~operation:Kernel_abi.Operation.Q8_linear
             ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32 ])
  |> List.flatten

let q8_all_entries =
  [ kernel_entry_with_tile ~tile:(16, 16, 64) ~name:"llmopt_q8_linear"
      ~operation:Kernel_abi.Operation.Q8_linear
      ~input_dtype:Ir.Dtype.Float16
      ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_tile ~tile:(16, 16, 64) ~name:"llmopt_q8_linear_f32"
      ~operation:Kernel_abi.Operation.Q8_linear
      ~input_dtype:Ir.Dtype.Float32
      ~output_dtype:Ir.Dtype.Float32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_gemv_simd" ~operation:Kernel_abi.Operation.Q8_linear
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_gemv_simd_f32" ~operation:Kernel_abi.Operation.Q8_linear
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_gemv_pair_simd"
      ~operation:Kernel_abi.Operation.Q8_linear ~input_dtype:Ir.Dtype.Float16
      ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_gemv_pair_simd_f32"
      ~operation:Kernel_abi.Operation.Q8_linear ~input_dtype:Ir.Dtype.Float32
      ~output_dtype:Ir.Dtype.Float32;
    kernel_entry ~name:"llmopt_q8_linear_silu"
      ~operation:Kernel_abi.Operation.Q8_linear_silu
      ~input_dtype:Ir.Dtype.Float16
      ~output_dtype:Ir.Dtype.Float16;
    kernel_entry ~name:"llmopt_q8_linear_silu_f32"
      ~operation:Kernel_abi.Operation.Q8_linear_silu
      ~input_dtype:Ir.Dtype.Float32
      ~output_dtype:Ir.Dtype.Float32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_gemv_silu_simd"
      ~operation:Kernel_abi.Operation.Q8_linear_silu
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_gemv_silu_simd_f32"
      ~operation:Kernel_abi.Operation.Q8_linear_silu
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_gemv_silu_pair_simd"
      ~operation:Kernel_abi.Operation.Q8_linear_silu
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_gemv_silu_pair_simd_f32"
      ~operation:Kernel_abi.Operation.Q8_linear_silu
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32;
    kernel_entry ~name:"llmopt_q8_linear_add"
      ~operation:Kernel_abi.Operation.Q8_linear_add
      ~input_dtype:Ir.Dtype.Float16
      ~output_dtype:Ir.Dtype.Float16;
    kernel_entry ~name:"llmopt_q8_linear_add_f32"
      ~operation:Kernel_abi.Operation.Q8_linear_add
      ~input_dtype:Ir.Dtype.Float32
      ~output_dtype:Ir.Dtype.Float32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_gemv_add_simd"
      ~operation:Kernel_abi.Operation.Q8_linear_add
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_gemv_add_simd_f32"
      ~operation:Kernel_abi.Operation.Q8_linear_add
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_gemv_add_pair_simd"
      ~operation:Kernel_abi.Operation.Q8_linear_add
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_gemv_add_pair_simd_f32"
      ~operation:Kernel_abi.Operation.Q8_linear_add
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32;
    kernel_entry ~name:"llmopt_q8_linear_mul_add"
      ~operation:Kernel_abi.Operation.Q8_linear_mul_add
      ~input_dtype:Ir.Dtype.Float16
      ~output_dtype:Ir.Dtype.Float16;
    kernel_entry ~name:"llmopt_q8_linear_mul_add_f32"
      ~operation:Kernel_abi.Operation.Q8_linear_mul_add
      ~input_dtype:Ir.Dtype.Float32
      ~output_dtype:Ir.Dtype.Float32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_gemv_mul_add_simd"
      ~operation:Kernel_abi.Operation.Q8_linear_mul_add
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_gemv_mul_add_simd_f32"
      ~operation:Kernel_abi.Operation.Q8_linear_mul_add
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_gemv_mul_add_pair_simd"
      ~operation:Kernel_abi.Operation.Q8_linear_mul_add
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_gemv_mul_add_pair_simd_f32"
      ~operation:Kernel_abi.Operation.Q8_linear_mul_add
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32;
    kernel_entry ~name:"llmopt_q8_dequantize"
      ~operation:Kernel_abi.Operation.Q8_dequantize
      ~input_dtype:Ir.Dtype.Int8
      ~output_dtype:Ir.Dtype.Float16;
    kernel_entry ~name:"llmopt_q8_dequantize_f32"
      ~operation:Kernel_abi.Operation.Q8_dequantize
      ~input_dtype:Ir.Dtype.Int8
      ~output_dtype:Ir.Dtype.Float32 ]
  @ q8_parameterized_entries

let q8_dual_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_dual_linear_f16"
      ~operation:Kernel_abi.Operation.Q8_linear
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_dual_linear_f16_bias"
      ~operation:Kernel_abi.Operation.Q8_linear
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_dual_linear_f32"
      ~operation:Kernel_abi.Operation.Q8_linear
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_dual_linear_f32_bias"
      ~operation:Kernel_abi.Operation.Q8_linear
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_dual_linear_silu_f16"
      ~operation:Kernel_abi.Operation.Q8_linear
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_dual_linear_silu_f16_bias"
      ~operation:Kernel_abi.Operation.Q8_linear
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_dual_linear_silu_f32"
      ~operation:Kernel_abi.Operation.Q8_linear
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_dual_linear_silu_f32_bias"
      ~operation:Kernel_abi.Operation.Q8_linear
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32 ]

let q8_qkv_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_qkv_linear_f16"
      ~operation:Kernel_abi.Operation.Q8_linear
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_qkv_linear_f16_bias"
      ~operation:Kernel_abi.Operation.Q8_linear
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_qkv_linear_f32"
      ~operation:Kernel_abi.Operation.Q8_linear
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_qkv_linear_f32_bias"
      ~operation:Kernel_abi.Operation.Q8_linear
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float32 ]

let q8_linear_add_norm_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_linear_add_norm_f16"
      ~operation:Kernel_abi.Operation.Q8_linear_add
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_linear_add_norm_f32"
      ~operation:Kernel_abi.Operation.Q8_linear_add
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_linear_add_norm_extra_f16"
      ~operation:Kernel_abi.Operation.Q8_linear_add
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_linear_add_norm_extra_f32"
      ~operation:Kernel_abi.Operation.Q8_linear_add
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float16 ]

let q8_lm_head_argmax_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_lm_head_argmax_f16"
      ~operation:Kernel_abi.Operation.Q8_lm_head_argmax
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Int32;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_lm_head_argmax_f32"
      ~operation:Kernel_abi.Operation.Q8_lm_head_argmax
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Int32 ]

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

kernel void llmopt_cache_pack_attention_f16(
    device const half* source [[buffer(0)]],
    device const uint* slots [[buffer(1)]],
    device uchar* pool [[buffer(2)]],
    constant AttentionCacheParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint segment_elements = params.heads * params.head_dim;
  if (gid >= params.items * segment_elements) return;
  const uint item = gid / segment_elements;
  const uint local = gid - item * segment_elements;
  const uint head = local / params.head_dim;
  const uint within_head = local - head * params.head_dim;
  const uint source_index =
      (head * params.source_items + params.source_offset + item)
      * params.head_dim + within_head;
  device half* destination = reinterpret_cast<device half*>(
      pool + slots[item] * params.token_stride);
  destination[params.segment * segment_elements + local] = source[source_index];
}

kernel void llmopt_cache_unpack_attention_f16(
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
  const uint destination_index =
      (head * params.items + item) * params.head_dim + within_head;
  device const half* source = reinterpret_cast<device const half*>(
      pool + slots[item] * params.token_stride);
  destination[destination_index] =
      source[params.segment * segment_elements + local];
}

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

kernel void llmopt_cache_pack_checkpoint_f16(
    device const half* source [[buffer(0)]],
    device uchar* pool [[buffer(1)]],
    constant CheckpointCacheParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.layer_elements) return;
  device half* destination = reinterpret_cast<device half*>(
      pool + params.checkpoint * params.checkpoint_stride);
  destination[params.layer * params.layer_elements + gid] = source[gid];
}

kernel void llmopt_cache_unpack_checkpoint_f16(
    device const uchar* pool [[buffer(0)]],
    device half* destination [[buffer(1)]],
    constant CheckpointCacheParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= params.layer_elements) return;
  device const half* source = reinterpret_cast<device const half*>(
      pool + params.checkpoint * params.checkpoint_stride);
  destination[gid] = source[params.layer * params.layer_elements + gid];
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
      + kv_head * groups_per_head + dimension / params.group_size;
  return float(half(float(values[value_index]) * float(scales[scale_index])));
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
  const uint rows = params.batches * params.query_heads;
  if (row >= rows) return;
  const uint head = row % params.query_heads;
  const uint batch = row / params.query_heads;
  const uint heads_per_kv = params.query_heads / params.kv_heads;
  const uint kv_head = head / heads_per_kv;
  const uint query_base =
      (batch * params.query_heads + head) * params.head_dim;
  const uint current_base =
      (batch * params.kv_heads + kv_head) * params.head_dim;
  const uint key_length = params.past_length + 1;
  const uint key_segment = params.cache_layer * 2;
  const uint value_segment = key_segment + 1;
  float maximum = -INFINITY;
  float denominator = 0.0f;
  float result_low = 0.0f;
  float result_high = 0.0f;
  for (uint key_position = 0; key_position < key_length; ++key_position) {
    const uint mask_batch = params.mask_batches == 1 ? 0 : batch;
    const uint mask_head = params.mask_heads == 1 ? 0 : head;
    const uint mask_index =
        (mask_batch * params.mask_heads + mask_head) * key_length + key_position;
    if (mask[mask_index] == 0) continue;
    float partial_score = 0.0f;
    for (uint dimension = lane; dimension < params.head_dim;
         dimension += Q8_PAGED_ATTENTION_SIMD_WIDTH) {
      const float key_value = key_position < params.past_length
          ? llmopt_q8_paged_value(pool, slots, params, key_position,
                key_segment, kv_head, dimension)
          : float(current_key[current_base + dimension]);
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
          : float(current_value[current_base + lane]);
      result_low = result_low * previous_scale + current_scale * value;
    }
    const uint high_dimension = lane + Q8_PAGED_ATTENTION_SIMD_WIDTH;
    if (high_dimension < params.head_dim) {
      const float value = key_position < params.past_length
          ? llmopt_q8_paged_value(pool, slots, params, key_position,
                value_segment, kv_head, high_dimension)
          : float(current_value[current_base + high_dimension]);
      result_high = result_high * previous_scale + current_scale * value;
    }
    denominator = denominator * previous_scale + current_scale;
    maximum = next_maximum;
  }
  const uint output_base = row * params.head_dim;
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

let cache_f16_entries =
  [ cache_entry "llmopt_cache_pack_attention_f16" Ir.Dtype.Float16
      Ir.Dtype.Float16;
    cache_entry "llmopt_cache_unpack_attention_f16" Ir.Dtype.Float16
      Ir.Dtype.Float16;
    cache_entry "llmopt_cache_pack_checkpoint_f16" Ir.Dtype.Float16
      Ir.Dtype.Float16;
    cache_entry "llmopt_cache_unpack_checkpoint_f16" Ir.Dtype.Float16
      Ir.Dtype.Float16 ]

let cache_q8_entries =
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

let add_cache_kernels ~formats program =
  let has_f16 = List.mem Kv_cache.Format.f16 formats in
  let has_q8 =
    List.exists
      (function Kv_cache.Format.Q8 _ -> true | Kv_cache.Format.F16 -> false)
      formats
  in
  let entries =
    (if has_f16 then cache_f16_entries else [])
    @ if has_q8 then cache_q8_entries else []
  in
  if entries = [] then program
  else
    Program.make ~source:(Program.source program ^ cache_source)
      ~kernels:(Program.kernels program @ entries)

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
  "\nstruct ShortConvPrefillParams {\n"
  ^ "  uint tokens;\n"
  ^ "  uint channels;\n"
  ^ "};\n\n"
  ^ "kernel void llmopt_short_conv_prefill_f16(\n"
  ^ "    device const half* in_proj [[buffer(0)]],\n"
  ^ "    device const half* conv_weight [[buffer(1)]],\n"
  ^ "    device half* output [[buffer(2)]],\n"
  ^ "    device half* conv_state_out [[buffer(3)]],\n"
  ^ "    constant ShortConvPrefillParams& params [[buffer(4)]],\n"
  ^ "    uint gid [[thread_position_in_grid]]) {\n"
  ^ "  if (gid >= params.channels) return;\n"
  ^ "  const uint c = gid;\n"
  ^ "  const uint tokens = params.tokens;\n"
  ^ "  const uint channels = params.channels;\n"
  ^ "  const uint weight_base = c * 3;\n"
  ^ "  const float w0 = float(conv_weight[weight_base + 0]);\n"
  ^ "  const float w1 = float(conv_weight[weight_base + 1]);\n"
  ^ "  const float w2 = float(conv_weight[weight_base + 2]);\n"
  ^ "  float g_prev3 = 0.0f;\n"
  ^ "  float g_prev2 = 0.0f;\n"
  ^ "  float g_prev1 = 0.0f;\n"
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
  ^ "  const uint state_base = c * 3;\n"
  ^ "  conv_state_out[state_base + 0] = half(g_prev3);\n"
  ^ "  conv_state_out[state_base + 1] = half(g_prev2);\n"
  ^ "  conv_state_out[state_base + 2] = half(g_prev1);\n"
  ^ "}\n"

let short_conv_prefill_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_short_conv_prefill_f16"
      ~operation:Kernel_abi.Operation.Short_conv_prefill
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16 ]

let attention_source =
  "\nconstant uint ATTENTION_SIMD_WIDTH = 32;\n"
  ^ "constant uint ATTENTION_ROWS_PER_THREADGROUP = 8;\n\n"
  ^ "struct AttentionParams {\n"
  ^ "  uint batches; uint heads; uint query_length; uint key_length;\n"
  ^ "  uint head_dimension; uint mask_batches; uint mask_heads; uint causal;\n"
  ^ "  float scale;\n"
  ^ "};\n\n"
  ^ "inline bool llmopt_attention_allowed(\n"
  ^ "    device const uchar* mask, constant AttentionParams& params,\n"
  ^ "    uint batch, uint head, uint query_position, uint key_position) {\n"
  ^ "  const uint mask_batch = params.mask_batches == 1 ? 0 : batch;\n"
  ^ "  const uint mask_head = params.mask_heads == 1 ? 0 : head;\n"
  ^ "  const uint mask_index = (((mask_batch * params.mask_heads + mask_head)\n"
  ^ "      * params.query_length + query_position) * params.key_length)\n"
  ^ "      + key_position;\n"
  ^ "  return mask[mask_index] != 0\n"
  ^ "      && (params.causal == 0 || key_position <= query_position);\n"
  ^ "}\n\n"
  ^ "inline float llmopt_attention_score(\n"
  ^ "    device const half* query, device const half* key,\n"
  ^ "    constant AttentionParams& params, uint batch, uint head,\n"
  ^ "    uint query_position, uint key_position) {\n"
  ^ "  float result = 0.0f;\n"
  ^ "  for (uint dimension = 0; dimension < params.head_dimension; ++dimension) {\n"
  ^ "    const uint query_index = (((batch * params.heads + head)\n"
  ^ "        * params.query_length + query_position) * params.head_dimension)\n"
  ^ "        + dimension;\n"
  ^ "    const uint key_index = (((batch * params.heads + head)\n"
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
  ^ "  for (uint dimension = 0; dimension < params.head_dimension; ++dimension) {\n"
  ^ "    float result = 0.0f;\n"
  ^ "    if (denominator != 0.0f) {\n"
  ^ "      for (uint key_position = 0; key_position < params.key_length; ++key_position) {\n"
  ^ "        if (llmopt_attention_allowed(mask, params, batch, head, query_position, key_position)) {\n"
  ^ "          const float probability = exp(llmopt_attention_score(query, key, params,\n"
  ^ "              batch, head, query_position, key_position) - maximum) / denominator;\n"
  ^ "          const uint value_index = (((batch * params.heads + head)\n"
  ^ "              * params.key_length + key_position) * params.head_dimension)\n"
  ^ "              + dimension;\n"
  ^ "          result += probability * float(value[value_index]);\n"
  ^ "        }\n"
  ^ "      }\n"
  ^ "    }\n"
  ^ "    output[gid * params.head_dimension + dimension] = half(result);\n"
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
  ^ "  const uint query_base = (((batch * params.heads + head)\n"
  ^ "      * params.query_length + query_position) * params.head_dimension);\n"
  ^ "  float maximum = -INFINITY;\n"
  ^ "  float denominator = 0.0f;\n"
  ^ "  float result_low = 0.0f;\n"
  ^ "  float result_high = 0.0f;\n"
  ^ "  for (uint key_position = 0; key_position < params.key_length; ++key_position) {\n"
  ^ "    if (!llmopt_attention_allowed(mask, params, batch, head,\n"
  ^ "        query_position, key_position)) continue;\n"
  ^ "    const uint key_base = (((batch * params.heads + head)\n"
  ^ "        * params.key_length + key_position) * params.head_dimension);\n"
  ^ "    float partial_score = 0.0f;\n"
  ^ "    for (uint dimension = lane; dimension < params.head_dimension;\n"
  ^ "         dimension += ATTENTION_SIMD_WIDTH)\n"
  ^ "      partial_score += float(query[query_base + dimension])\n"
  ^ "          * float(key[key_base + dimension]);\n"
  ^ "    const float score = simd_sum(partial_score) * params.scale;\n"
  ^ "    const float next_maximum = max(maximum, score);\n"
  ^ "    const float previous_scale = denominator == 0.0f ? 0.0f\n"
  ^ "        : exp(maximum - next_maximum);\n"
  ^ "    const float current_scale = exp(score - next_maximum);\n"
  ^ "    if (lane < params.head_dimension)\n"
  ^ "      result_low = result_low * previous_scale\n"
  ^ "          + current_scale * float(value[key_base + lane]);\n"
  ^ "    const uint high_dimension = lane + ATTENTION_SIMD_WIDTH;\n"
  ^ "    if (high_dimension < params.head_dimension)\n"
  ^ "      result_high = result_high * previous_scale\n"
  ^ "          + current_scale * float(value[key_base + high_dimension]);\n"
  ^ "    denominator = denominator * previous_scale + current_scale;\n"
  ^ "    maximum = next_maximum;\n"
  ^ "  }\n"
  ^ "  const uint output_base = row * params.head_dimension;\n"
  ^ "  if (lane < params.head_dimension)\n"
  ^ "    output[output_base + lane] = half(denominator == 0.0f\n"
  ^ "        ? 0.0f : result_low / denominator);\n"
  ^ "  const uint high_dimension = lane + ATTENTION_SIMD_WIDTH;\n"
  ^ "  if (high_dimension < params.head_dimension)\n"
  ^ "    output[output_base + high_dimension] = half(denominator == 0.0f\n"
  ^ "        ? 0.0f : result_high / denominator);\n"
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

let has_q8_linear graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Q8_linear _ -> true
         | _ -> false)

let has_q8_linear_silu graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Q8_linear_silu _ -> true
         | _ -> false)

let has_q8_linear_add graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Q8_linear_add _ -> true
         | _ -> false)

let has_q8_linear_mul_add graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Q8_linear_mul_add _ -> true
         | _ -> false)

let has_q8_linear_add_norm graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Q8_linear_add_norm _ -> true
         | _ -> false)

let has_q8_lm_head_argmax graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Q8_lm_head_argmax _ -> true
         | _ -> false)

let has_q8_dual_linear graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Q8_dual_linear _ -> true
         | _ -> false)

let has_q8_qkv_linear graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Q8_qkv_linear _ -> true
         | _ -> false)

let has_q8 graph =
  has_q8_linear graph || has_q8_linear_silu graph || has_q8_linear_add graph
  || has_q8_linear_mul_add graph || has_q8_linear_add_norm graph
  || has_q8_dual_linear graph || has_q8_qkv_linear graph
  || has_q8_lm_head_argmax graph

let q8_entries graph =
  q8_all_entries
  |> List.filter (fun entry ->
         match Kernel_abi.Entry.operation entry with
         | Kernel_abi.Operation.Q8_linear -> has_q8_linear graph
         | Kernel_abi.Operation.Q8_linear_silu -> has_q8_linear_silu graph
         | Kernel_abi.Operation.Q8_linear_add -> has_q8_linear_add graph
         | Kernel_abi.Operation.Q8_linear_mul_add ->
             has_q8_linear_mul_add graph
         | Kernel_abi.Operation.Q8_dequantize -> has_q8 graph
         | _ -> false)
  |> fun entries ->
  let entries =
    if has_q8_dual_linear graph then entries @ q8_dual_entries else entries
  in
  let entries =
    if has_q8_qkv_linear graph then entries @ q8_qkv_entries else entries
  in
  if has_q8_linear_add_norm graph then
    entries @ q8_linear_add_norm_entries
  else entries
  |> fun entries ->
  if has_q8_lm_head_argmax graph then
    entries @ q8_lm_head_argmax_entries
  else entries

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
          "#include <metal_stdlib>\nusing namespace metal;\n\n"
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
          ^ "    b_tile[tid.y][tid.x] = (col < params.n && base + tid.y < params.k)\n"
          ^ "        ? b[(base + tid.y) * params.n + col] : 0.0f;\n"
          ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
          ^ "    for (uint inner = 0; inner < TILE; ++inner) acc += a_tile[tid.y][inner] * b_tile[inner][tid.x];\n"
          ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
          ^ "  }\n"
          ^ "  if (row < params.m && col < params.n)\n"
          ^ "    c[row * params.n + col] = acc;\n"
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
          "#include <metal_stdlib>\nusing namespace metal;\n\n"
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
          "#include <metal_stdlib>\nusing namespace metal;\n\n"
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
    [ has_q8 graph, q8_source, q8_entries graph;
      has_f16_linear graph, linear_f16_source, linear_f16_entries;
      has_rms_norm graph, rms_norm_source, rms_norm_entries;
      has_rms_rope graph, rms_rope_source, rms_rope_entries;
      has_short_conv graph, short_conv_source, short_conv_entries;
      has_short_conv_step graph, short_conv_step_source, short_conv_step_entries;
      ( has_short_conv_step_fused graph,
        short_conv_step_fused_source,
        short_conv_step_fused_entries );
      has_short_conv_prefill graph, short_conv_prefill_source, short_conv_prefill_entries;
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
             ("#include <metal_stdlib>\nusing namespace metal;\n"
             ^ auxiliary_source)
           ~kernels:auxiliary_entries)
  | Error message -> Error message

let emit graph = Result.map Program.source (lower graph)
