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
