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
  ^ "  for (uint base = 0; base < params.k; base += Q8_TILE) {\n"
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
  ^ "    for (uint inner = 0; inner < Q8_TILE; ++inner)\n"
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

let q8_source =
  "\nconstant uint Q8_TILE = 16;\n\n"
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

let q8_entries =
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
      ~output_dtype:Ir.Dtype.Float32 ]

let rms_norm_source =
  "\nstruct RmsNormParams { uint rows; uint width; float epsilon; };\n\n"
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
  ^ "}\n"

let rms_norm_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_rms_norm_f32_f16"
      ~operation:Kernel_abi.Operation.Rms_norm
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_rms_norm_f16"
      ~operation:Kernel_abi.Operation.Rms_norm
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

let attention_source =
  "\nstruct AttentionParams {\n"
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
  ^ "}\n"

let attention_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
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

let has_rms_norm graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with Ir.Op.Rms_norm _ -> true | _ -> false)

let has_short_conv graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Primitive (Ir.Primitive.Short_conv _) -> true
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

let has_q8 graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Q8_linear _ -> true
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

let lower graph =
  let components =
    [ has_q8 graph, q8_source, q8_entries;
      has_rms_norm graph, rms_norm_source, rms_norm_entries;
      has_short_conv graph, short_conv_source, short_conv_entries;
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
        cast_entries ) ]
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
