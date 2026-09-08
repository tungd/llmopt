(** Metal recurrent/ShortConv shader catalog (part 1): short_conv and
   short_conv_silu sources, step/prefill/prefill-init variants and their
   ABI entries.  Split out of metal.ml (LOOP item 5). *)

open Metal_source_common

let replace_first = Metal_tensor_source.replace_first

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

let short_conv_silu_source = {|
struct ShortConvSiluTmParams {
  uint batches; uint tokens; uint channels; uint kernel_width;
  uint stride; uint padding; uint dilation; uint output_start;
};

kernel void llmopt_short_conv_silu_tm_f16(
    device const half* input [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant ShortConvSiluTmParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint count = params.batches * params.tokens * params.channels;
  if (gid >= count) return;
  const uint channel = gid % params.channels;
  const uint token = (gid / params.channels) % params.tokens;
  const uint batch = gid / (params.tokens * params.channels);
  float accumulator = 0.0f;
  for (uint tap = 0; tap < params.kernel_width; ++tap) {
    const int source_token = int((token + params.output_start) * params.stride)
        - int(params.padding) + int(tap * params.dilation);
    if (source_token >= 0 && source_token < int(params.tokens)) {
      const uint input_index = ((batch * params.tokens + uint(source_token))
          * params.channels) + channel;
      const uint weight_index = channel * params.kernel_width + tap;
      accumulator += float(input[input_index]) * float(weight[weight_index]);
    }
  }
  const half convolved = half(accumulator);
  const float value = float(convolved);
  output[gid] = half(value / (1.0f + exp(-value)));
}

kernel void llmopt_short_conv_silu_tm_f32(
    device const half* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant ShortConvSiluTmParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint count = params.batches * params.tokens * params.channels;
  if (gid >= count) return;
  const uint channel = gid % params.channels;
  const uint token = (gid / params.channels) % params.tokens;
  const uint batch = gid / (params.tokens * params.channels);
  float accumulator = 0.0f;
  for (uint tap = 0; tap < params.kernel_width; ++tap) {
    const int source_token = int((token + params.output_start) * params.stride)
        - int(params.padding) + int(tap * params.dilation);
    if (source_token >= 0 && source_token < int(params.tokens)) {
      const uint input_index = ((batch * params.tokens + uint(source_token))
          * params.channels) + channel;
      const uint weight_index = channel * params.kernel_width + tap;
      accumulator += float(input[input_index]) * weight[weight_index];
    }
  }
  const half convolved = half(accumulator);
  const float value = float(convolved);
  output[gid] = half(value / (1.0f + exp(-value)));
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

let short_conv_silu_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_short_conv_silu_tm_f16"
      ~operation:Kernel_abi.Operation.Short_conv
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_short_conv_silu_tm_f32"
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
