type kernel =
  | Matmul of int * int * int * Ir.Value.t * Ir.Value.t * Ir.Value.t
  | Fused of int * int * int * Ir.Value.t * Ir.Value.t * Ir.Value.t * Ir.Value.t
  | Linear of int * int * int * bool * Ir.Value.t * Ir.Value.t * Ir.Value.t option * Ir.Value.t
  | Q8 of int * int * int * bool * Ir.Value.t * Ir.Value.t * Ir.Value.t * Ir.Value.t option * Ir.Value.t

module Program = struct
  type t = { source : string; kernels : Kernel_abi.Entry.t list }

  let make ~source ~kernels = { source; kernels }
  let source program = program.source
  let kernels program = program.kernels
end

let kernel_entry ~name ~operation ~input_dtype ~output_dtype =
  match
    Kernel_abi.Entry.create ~name ~operation ~input_dtype ~output_dtype
      ~threadgroup:(16, 16, 1)
  with
  | Ok entry -> entry
  | Error message -> invalid_arg ("invalid built-in Metal kernel: " ^ message)

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

let q8_kernel ~name ~value_type ~vector_type ~alignment ~weight_value_type
    ~weight_cast ~scale_type ~scale_zero ~zero_value ~store_value =
  "kernel void " ^ name ^ "(\n"
  ^ "    device const " ^ value_type ^ "* input [[buffer(0)]],\n"
  ^ "    device const char* weight [[buffer(1)]],\n"
  ^ "    device const half* scale [[buffer(2)]],\n"
  ^ "    device const half* bias_or_scale [[buffer(3)]],\n"
  ^ "    device " ^ value_type ^ "* output [[buffer(4)]],\n"
  ^ "    constant Q8Params& params [[buffer(5)]],\n"
  ^ "    uint2 gid [[thread_position_in_grid]],\n"
  ^ "    uint2 tid [[thread_position_in_threadgroup]]) {\n"
  ^ "  const uint row = gid.y;\n"
  ^ "  const uint col = gid.x;\n"
  ^ "  threadgroup " ^ value_type ^ " input_tile[16][16];\n"
  ^ "  threadgroup " ^ weight_value_type ^ " weight_tile[16][16];\n"
  ^ "  float acc = 0.0f;\n"
  ^ "  const " ^ scale_type ^ " channel_scale =\n"
  ^ "      (col < params.n) ? scale[col] : " ^ scale_zero ^ ";\n"
  ^ "  for (uint base = 0; base < params.k; base += TILE) {\n"
  ^ "    if (tid.x < 4) {\n"
  ^ "      const uint input_offset = row * params.k + base + tid.x * 4;\n"
  ^ "      if (row < params.m && base + tid.x * 4 + 3 < params.k &&\n"
  ^ "          input_offset % " ^ alignment ^ " == 0) {\n"
  ^ "        device const " ^ vector_type ^ "* input_vector =\n"
  ^ "            reinterpret_cast<device const " ^ vector_type ^ "*>(input + input_offset);\n"
  ^ "        " ^ vector_type ^ " values = *input_vector;\n"
  ^ "        input_tile[tid.y][tid.x * 4 + 0] = values[0];\n"
  ^ "        input_tile[tid.y][tid.x * 4 + 1] = values[1];\n"
  ^ "        input_tile[tid.y][tid.x * 4 + 2] = values[2];\n"
  ^ "        input_tile[tid.y][tid.x * 4 + 3] = values[3];\n"
  ^ "      } else {\n"
  ^ "        for (uint lane = 0; lane < 4; ++lane)\n"
  ^ "          input_tile[tid.y][tid.x * 4 + lane] =\n"
  ^ "              (row < params.m && base + tid.x * 4 + lane < params.k)\n"
  ^ "                  ? input[input_offset + lane]\n"
  ^ "                  : " ^ zero_value ^ ";\n"
  ^ "      }\n"
  ^ "    }\n"
  ^ "    if (tid.y < 4) {\n"
  ^ "      const uint weight_offset = col * params.k + base + tid.y * 4;\n"
  ^ "      if (col < params.n && base + tid.y * 4 + 3 < params.k &&\n"
  ^ "          weight_offset % 4 == 0) {\n"
  ^ "        device const char4* weight_vector =\n"
  ^ "            reinterpret_cast<device const char4*>(weight + weight_offset);\n"
  ^ "        char4 values = *weight_vector;\n"
  ^ "        weight_tile[tid.y * 4 + 0][tid.x] = " ^ weight_cast ^ "(values[0]) * channel_scale;\n"
  ^ "        weight_tile[tid.y * 4 + 1][tid.x] = " ^ weight_cast ^ "(values[1]) * channel_scale;\n"
  ^ "        weight_tile[tid.y * 4 + 2][tid.x] = " ^ weight_cast ^ "(values[2]) * channel_scale;\n"
  ^ "        weight_tile[tid.y * 4 + 3][tid.x] = " ^ weight_cast ^ "(values[3]) * channel_scale;\n"
  ^ "      } else {\n"
  ^ "        for (uint lane = 0; lane < 4; ++lane)\n"
  ^ "          weight_tile[tid.y * 4 + lane][tid.x] =\n"
  ^ "              (col < params.n && base + tid.y * 4 + lane < params.k)\n"
  ^ "                  ? " ^ weight_cast ^ "(weight[weight_offset + lane]) * channel_scale\n"
  ^ "                  : " ^ weight_cast ^ "(0);\n"
  ^ "      }\n"
  ^ "    }\n"
  ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "    #pragma clang loop unroll(full)\n"
  ^ "    for (uint inner = 0; inner < TILE; ++inner)\n"
  ^ "      acc += float(input_tile[tid.y][inner]) *\n"
  ^ "             float(weight_tile[inner][tid.x]);\n"
  ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
  ^ "  }\n"
  ^ "  if (row < params.m && col < params.n) {\n"
  ^ "    if (params.has_bias != 0) acc += float(bias_or_scale[col]);\n"
  ^ "    output[row * params.n + col] = " ^ store_value ^ ";\n"
  ^ "  }\n"
  ^ "}\n"

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

let lower graph =
  let kernel =
    Ir.Graph.nodes graph
    |> List.find_map (fun node ->
           match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
           | Ir.Op.Matmul { m; n; k }, [ lhs; rhs ], Some output ->
               Some (Matmul (m, n, k, lhs, rhs, output))
           | Ir.Op.Fused_matmul_bias { m; n; k }, [ lhs; rhs; bias ], Some output ->
               Some (Fused (m, n, k, lhs, rhs, bias, output))
           | Ir.Op.Q8_linear { m; n; k; bias = has_bias }, inputs, Some output ->
               (match inputs with
               | [ input; weight; scale ] ->
                   Some (Q8 (m, n, k, has_bias, input, weight, scale, None, output))
               | [ input; weight; scale; bias ] ->
                   Some (Q8 (m, n, k, has_bias, input, weight, scale, Some bias, output))
               | _ -> None)
           | Ir.Op.Linear { m; n; k; bias }, inputs, Some output ->
               (match inputs with
               | [ input; weight ] -> Some (Linear (m, n, k, bias, input, weight, None, output))
               | [ input; weight; bias_value ] ->
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
          ^ "kernel void llmopt_matmul(\n"
          ^ "    device const float* a [[buffer(0)]],\n"
          ^ "    device const float* b [[buffer(1)]],\n"
          ^ "    device float* c [[buffer(2)]],\n"
          ^ "    uint2 gid [[thread_position_in_grid]],\n"
          ^ "    uint2 tid [[thread_position_in_threadgroup]]) {\n"
          ^ "  threadgroup float a_tile[16][16];\n"
          ^ "  threadgroup float b_tile[16][16];\n"
          ^ Printf.sprintf "  const uint row = gid.y; const uint col = gid.x;\n"
          ^ Printf.sprintf "  float acc = 0.0f;\n"
          ^ Printf.sprintf "  for (uint base = 0; base < %d; base += TILE) {\n" k
          ^ Printf.sprintf
              "    a_tile[tid.y][tid.x] = (row < %d && base + tid.x < %d) ? a[row * %d + base + tid.x] : 0.0f;\n"
              m k k
          ^ Printf.sprintf
              "    b_tile[tid.y][tid.x] = (col < %d && base + tid.y < %d) ? b[(base + tid.y) * %d + col] : 0.0f;\n"
              n k n
          ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
          ^ "    for (uint inner = 0; inner < TILE; ++inner) acc += a_tile[tid.y][inner] * b_tile[inner][tid.x];\n"
          ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
          ^ "  }\n"
          ^ Printf.sprintf "  if (row < %d && col < %d) c[row * %d + col] = acc;\n" m n n
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
          ^ "kernel void llmopt_fused_linear(\n"
          ^ "    device const float* a [[buffer(0)]],\n"
          ^ "    device const float* b [[buffer(1)]],\n"
          ^ "    device const float* bias [[buffer(2)]],\n"
          ^ "    device float* c [[buffer(3)]],\n"
          ^ "    uint2 gid [[thread_position_in_grid]],\n"
          ^ "    uint2 tid [[thread_position_in_threadgroup]]) {\n"
          ^ "  threadgroup float a_tile[16][16];\n"
          ^ "  threadgroup float b_tile[16][16];\n"
          ^ "  const uint row = gid.y; const uint col = gid.x;\n"
          ^ "  float acc = 0.0f;\n"
          ^ Printf.sprintf "  for (uint base = 0; base < %d; base += TILE) {\n" k
          ^ Printf.sprintf
              "    a_tile[tid.y][tid.x] = (row < %d && base + tid.x < %d) ? a[row * %d + base + tid.x] : 0.0f;\n"
              m k k
          ^ Printf.sprintf
              "    b_tile[tid.y][tid.x] = (col < %d && base + tid.y < %d) ? b[(base + tid.y) * %d + col] : 0.0f;\n"
              n k n
          ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
          ^ "    for (uint inner = 0; inner < TILE; ++inner) acc += a_tile[tid.y][inner] * b_tile[inner][tid.x];\n"
          ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
          ^ "  }\n"
          ^ Printf.sprintf "  if (row < %d && col < %d) c[row * %d + col] = acc + bias[col];\n" m n n
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
  | Some (Q8 (_m, _n, _k, _has_bias, input, weight, scale, bias, output)) ->
      let input_symbol = input_name graph input in
      let weight_symbol = input_name graph weight in
      let scale_symbol = input_name graph scale in
      let bias_symbol = Option.map (input_name graph) bias in
      let output_symbol = input_name graph output in
      let source =
        "#include <metal_stdlib>\nusing namespace metal;\n\n"
        ^ "constant uint TILE = 16;\n\n"
        ^ "struct Q8Params { uint m; uint n; uint k; uint has_bias; };\n\n"
        ^ q8_kernel
            ~name:"llmopt_q8_linear"
            ~value_type:"half"
            ~vector_type:"half4"
            ~alignment:"8"
            ~weight_value_type:"half"
            ~weight_cast:"half"
            ~scale_type:"half"
            ~scale_zero:"half(0.0h)"
            ~zero_value:"half(0.0h)"
            ~store_value:"half(acc)"
        ^ q8_kernel
            ~name:"llmopt_q8_linear_f32"
            ~value_type:"float"
            ~vector_type:"float4"
            ~alignment:"16"
            ~weight_value_type:"float"
            ~weight_cast:"float"
            ~scale_type:"float"
            ~scale_zero:"0.0f"
            ~zero_value:"0.0f"
            ~store_value:"acc"
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
      in
      ignore (input_symbol, weight_symbol, scale_symbol, bias_symbol, output_symbol);
      Ok
        (Program.make ~source
           ~kernels:
             [ kernel_entry ~name:"llmopt_q8_linear"
                 ~operation:Kernel_abi.Operation.Q8_linear
                 ~input_dtype:Ir.Dtype.Float16
                 ~output_dtype:Ir.Dtype.Float16;
               kernel_entry ~name:"llmopt_q8_linear_f32"
                 ~operation:Kernel_abi.Operation.Q8_linear
                 ~input_dtype:Ir.Dtype.Float32
                 ~output_dtype:Ir.Dtype.Float32;
               kernel_entry ~name:"llmopt_q8_dequantize"
                 ~operation:Kernel_abi.Operation.Q8_dequantize
                 ~input_dtype:Ir.Dtype.Int8
                 ~output_dtype:Ir.Dtype.Float16;
               kernel_entry ~name:"llmopt_q8_dequantize_f32"
                 ~operation:Kernel_abi.Operation.Q8_dequantize
                 ~input_dtype:Ir.Dtype.Int8
                 ~output_dtype:Ir.Dtype.Float32 ])
  | Some (Linear (m, n, k, has_bias, input, weight, bias, output)) ->
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
          ^ "kernel void llmopt_linear(\n"
          ^ "    device const float* input [[buffer(0)]],\n"
          ^ "    device const float* weight [[buffer(1)]],\n"
          ^ bias_argument
          ^ Printf.sprintf "    device float* output [[buffer(%d)]],\n" output_buffer
          ^ "    uint2 gid [[thread_position_in_grid]],\n"
          ^ "    uint2 tid [[thread_position_in_threadgroup]]) {\n"
          ^ "  threadgroup float input_tile[16][16];\n"
          ^ "  threadgroup float weight_tile[16][16];\n"
          ^ "  const uint row = gid.y; const uint col = gid.x;\n"
          ^ "  float acc = 0.0f;\n"
          ^ Printf.sprintf "  for (uint base = 0; base < %d; base += TILE) {\n" k
          ^ Printf.sprintf
              "    input_tile[tid.y][tid.x] = (row < %d && base + tid.x < %d) ? input[row * %d + base + tid.x] : 0.0f;\n"
              m k k
          ^ Printf.sprintf
              "    weight_tile[tid.y][tid.x] = (col < %d && base + tid.y < %d) ? weight[col * %d + base + tid.y] : 0.0f;\n"
              n k k
          ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
          ^ "    for (uint inner = 0; inner < TILE; ++inner) acc += input_tile[tid.y][inner] * weight_tile[inner][tid.x];\n"
          ^ "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
          ^ "  }\n"
          ^ Printf.sprintf "  if (row < %d && col < %d) output[row * %d + col] = acc%s;\n" m n n bias_value
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

let emit graph = Result.map Program.source (lower graph)
