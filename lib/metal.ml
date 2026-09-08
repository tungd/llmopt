type kernel =
  | Matmul of int * int * int * Ir.Value.t * Ir.Value.t * Ir.Value.t
  | Fused of int * int * int * Ir.Value.t * Ir.Value.t * Ir.Value.t * Ir.Value.t
  | Linear of int * int * int * bool * Ir.Value.t * Ir.Value.t * Ir.Value.t option * Ir.Value.t

let ( let* ) = Result.bind

module Program = struct
  type t = { source : string; kernels : Kernel_abi.Entry.t list }

  let make ~source ~kernels = { source; kernels }
  let source program = program.source
  let kernels program = program.kernels
end

module Tactic = Metal_selection.Tactic

let kernel_entry_with_threadgroup = Metal_source_common.kernel_entry_with_threadgroup
let kernel_entry = Metal_source_common.kernel_entry
let kernel_entry_with_threadgroup_and_tile = Metal_source_common.kernel_entry_with_threadgroup_and_tile
let kernel_entry_with_tile = Metal_source_common.kernel_entry_with_tile

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

let w4a16_source = Metal_w4a16_source.w4a16_source
let w4a16_entries = Metal_w4a16_source.w4a16_entries
let w4a16_qkv_linear_source = Metal_w4a16_source.w4a16_qkv_linear_source
let w4a16_qkv_linear_entries = Metal_w4a16_source.w4a16_qkv_linear_entries
let w4a16_swiglu_ffn_source = Metal_w4a16_source.w4a16_swiglu_ffn_source
let w4a16_swiglu_ffn_entries = Metal_w4a16_source.w4a16_swiglu_ffn_entries
let w4a16_lm_head_argmax_source = Metal_w4a16_source.w4a16_lm_head_argmax_source
let w4a16_lm_head_argmax_entries = Metal_w4a16_source.w4a16_lm_head_argmax_entries

let cache_source = Metal_cache_source.cache_source

let cache_entries = Metal_cache_source.cache_entries
let add_cache_kernels program =
  Program.make ~source:(Program.source program ^ cache_source)
    ~kernels:(Program.kernels program @ cache_entries)

let quant_common_source = Metal_quant_source.quant_common_source

let q8_0_source = Metal_q8_0_source.q8_0_source

let q5_0_source = Metal_q5_0_source.q5_0_source

let q4_0_source = Metal_q4_0_source.q4_0_source

let quant_qkv_params_source = Metal_quant_qkv_source.quant_qkv_params_source
let q4_0_qkv_source = Metal_quant_qkv_source.q4_0_qkv_source
let q5_0_qkv_source = Metal_quant_qkv_source.q5_0_qkv_source
let q4_0_qkv_entries = Metal_quant_qkv_source.q4_0_qkv_entries
let q5_0_qkv_entries = Metal_quant_qkv_source.q5_0_qkv_entries
let quant_qkv_entries = Metal_quant_qkv_source.quant_qkv_entries
let quant_qkv_source = Metal_quant_qkv_source.quant_qkv_source

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
    kernel_entry_with_threadgroup ~threadgroup:(128, 1, 1)
      ~name:"llmopt_q8_0_linear_f16_m1_r2_sg4"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q8_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(128, 1, 1)
      ~name:"llmopt_q8_0_linear_f16_llama_mv"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q8_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_0_linear_f16_m2"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q8_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_0_linear_f16_splitk"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q8_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q8_0_linear_f16_splitk_m2"
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
      ~name:"llmopt_q5_0_gated_linear_f16"
      ~operation:Kernel_abi.Operation.Gated_linear
      ~input_dtype:(Ir.Dtype.Quant Q5_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_q5_0_linear_f16_llama_mv_add"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q5_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_q5_0_gated_linear_f16_llama_mv"
      ~operation:Kernel_abi.Operation.Gated_linear
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
      ~name:"llmopt_q4_0_linear_f16_m4"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_linear_f16_m4_add"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_linear_f16_m16"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_linear_f16_m16_add"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_linear_f16_m32"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_linear_f16_m32_add"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_linear_f16_x4"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_linear_f16_x8"
      ~operation:Kernel_abi.Operation.Linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_gated_linear_f16_x8"
      ~operation:Kernel_abi.Operation.Gated_linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_gated_linear_f16_m4"
      ~operation:Kernel_abi.Operation.Gated_linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_gated_linear_f16_m16"
      ~operation:Kernel_abi.Operation.Gated_linear
      ~input_dtype:(Ir.Dtype.Quant Q4_0) ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_q4_0_gated_linear_f16_m32"
      ~operation:Kernel_abi.Operation.Gated_linear
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

let metal_header = Metal_source_common.metal_header

let emit_dequant_q8_0 = Metal_quant_source.emit_dequant_q8_0
let emit_dequant_q5_0 = Metal_quant_source.emit_dequant_q5_0
let emit_dequant_q4_0 = Metal_quant_source.emit_dequant_q4_0
let q4_k_source = Metal_q4_k_source.q4_k_source

let q5_k_source = Metal_q5_k_source.q5_k_source

let q6_k_source = Metal_q6_k_source.q6_k_source

let iq4_xs_source = Metal_iq4_xs_source.iq4_xs_source

let extract_kernel = Metal_quant_source.extract_kernel
let legacy_quant_source = Metal_quant_source.legacy_quant_source
let quant_source = Metal_quant_source.quant_source
let quant_linear_add_source = Metal_quant_source.quant_linear_add_source
let kquant_entries = Metal_quant_source.kquant_entries

let add_kquant_kernels program =
  Program.make
    ~source:
      (Program.source program ^ quant_common_source ^ q4_k_source ^ q5_k_source ^ q6_k_source)
    ~kernels:(Program.kernels program @ kquant_entries)

let emit_dequant_q4_k = Metal_quant_source.emit_dequant_q4_k
let emit_dequant_q5_k = Metal_quant_source.emit_dequant_q5_k
let emit_dequant_q6_k = Metal_quant_source.emit_dequant_q6_k
let linear_f16_source = Metal_tensor_source.linear_f16_source
let linear_f16_entries = Metal_tensor_source.linear_f16_entries
let linear_f16_bf16_source = Metal_tensor_source.linear_f16_bf16_source
let linear_f16_bf16_entries = Metal_tensor_source.linear_f16_bf16_entries
let linear_f16_f32_source = Metal_tensor_source.linear_f16_f32_source
let linear_f16_f32_entries = Metal_tensor_source.linear_f16_f32_entries
let replace_first = Metal_tensor_source.replace_first
let replace_all = Metal_tensor_source.replace_all

let rms_norm_source = Metal_normalization_source.rms_norm_source
let rms_norm_entries = Metal_normalization_source.rms_norm_entries
let l2_norm_source = Metal_normalization_source.l2_norm_source
let l2_norm_entries = Metal_normalization_source.l2_norm_entries
let l2_norm_slice_source = Metal_normalization_source.l2_norm_slice_source
let l2_norm_slice_entries = Metal_normalization_source.l2_norm_slice_entries
let rms_rope_source = Metal_normalization_source.rms_rope_source
let rms_rope_entries = Metal_normalization_source.rms_rope_entries
let rms_rope_qk_source = Metal_normalization_source.rms_rope_qk_source
let rms_rope_qk_entries = Metal_normalization_source.rms_rope_qk_entries
let rotary_qk_source = Metal_normalization_source.rotary_qk_source
let rotary_qk_entries = Metal_normalization_source.rotary_qk_entries

let short_conv_source = Metal_recurrent_source.short_conv_source
let short_conv_f32_weight_source = Metal_recurrent_source.short_conv_f32_weight_source
let short_conv_silu_source = Metal_recurrent_source.short_conv_silu_source
let short_conv_entries = Metal_recurrent_source.short_conv_entries
let short_conv_silu_entries = Metal_recurrent_source.short_conv_silu_entries
let short_conv_step_source = Metal_recurrent_source.short_conv_step_source
let short_conv_step_entries = Metal_recurrent_source.short_conv_step_entries
let short_conv_step_fused_source = Metal_recurrent_source.short_conv_step_fused_source
let short_conv_step_fused_entries = Metal_recurrent_source.short_conv_step_fused_entries
let short_conv_prefill_source = Metal_recurrent_source.short_conv_prefill_source
let short_conv_prefill_entries = Metal_recurrent_source.short_conv_prefill_entries
let short_conv_prefill_init_source = Metal_recurrent_source.short_conv_prefill_init_source
let short_conv_prefill_init_entries = Metal_recurrent_source.short_conv_prefill_init_entries

let attention_source =
  "\nconstant uint ATTENTION_SIMD_WIDTH = 32;\n"
  ^ "constant uint ATTENTION_ROWS_PER_THREADGROUP = 8;\n\n"
  ^ "struct AttentionParams {\n"
  ^ "  uint batches; uint heads; uint query_length; uint key_length;\n"
  ^ "  uint head_dimension; uint mask_batches; uint mask_heads; uint causal;\n"
  ^ "  float scale; uint kv_heads; uint token_first_output;\n"
  ^ "  uint past_length; uint token_first_value; uint mask_dtype;\n"
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
  ^ "  if (params.mask_dtype == 2)\n"
  ^ "    return float(reinterpret_cast<device const half*>(mask)[mask_index]) >= 0.0f;\n"
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
  ^ "          const uint value_index = params.token_first_value != 0\n"
  ^ "              ? (((batch * params.key_length + key_position)\n"
  ^ "                  * effective_kv_heads + kv_head) * params.head_dimension) + dimension\n"
  ^ "              : (((batch * effective_kv_heads + kv_head)\n"
  ^ "                  * params.key_length + key_position) * params.head_dimension) + dimension;\n"
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
  ^ "    const uint value_base = params.token_first_value != 0\n"
  ^ "        ? (((batch * params.key_length + key_position) * params.kv_heads + kv_head) * 64)\n"
  ^ "        : key_base;\n"
  ^ "    const float partial_score = query_low * float(key[key_base + lane])\n"
  ^ "        + query_high * float(key[key_base + lane + ATTENTION_SIMD_WIDTH]);\n"
  ^ "    const float score = simd_sum(partial_score) * params.scale;\n"
  ^ "    const float next_maximum = max(maximum, score);\n"
  ^ "    const float previous_scale = denominator == 0.0f ? 0.0f\n"
  ^ "        : exp(maximum - next_maximum);\n"
  ^ "    const float current_scale = exp(score - next_maximum);\n"
  ^ "    result_low = result_low * previous_scale\n"
  ^ "        + current_scale * float(value[value_base + lane]);\n"
  ^ "    result_high = result_high * previous_scale\n"
  ^ "        + current_scale * float(value[value_base + lane + ATTENTION_SIMD_WIDTH]);\n"
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
    const uint value_base = params.token_first_value != 0
        ? (((batch * params.key_length + key_position) * effective_kv_heads
            + kv_head) * __ATTENTION_HEAD_DIMENSION__)
        : key_base;
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
          + current_scale * float(value[value_base + dimension]);
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

let attention_tree_speculative_source = {|
struct SpeculativeAttentionParams {
  uint batches; uint heads; uint query_length; uint key_length;
  uint head_dimension; uint kv_heads; uint past_length;
  float scale;
  uint tree_mask_0; uint tree_mask_1; uint tree_mask_2; uint tree_mask_3;
  uint tree_mask_4; uint tree_mask_5; uint tree_mask_6; uint tree_mask_7;
};

inline bool llmopt_speculative_tree_allowed(
    constant SpeculativeAttentionParams& params,
    uint query_position, uint key_position) {
  if (key_position < params.past_length) {
    return true;
  }
  const uint draft_key_idx = key_position - params.past_length;
  if (draft_key_idx >= params.query_length) return false;
  uint mask = 0;
  switch (query_position) {
    case 0: mask = params.tree_mask_0; break;
    case 1: mask = params.tree_mask_1; break;
    case 2: mask = params.tree_mask_2; break;
    case 3: mask = params.tree_mask_3; break;
    case 4: mask = params.tree_mask_4; break;
    case 5: mask = params.tree_mask_5; break;
    case 6: mask = params.tree_mask_6; break;
    case 7: mask = params.tree_mask_7; break;
    default: mask = 0; break;
  }
  return (mask & (1u << draft_key_idx)) != 0u;
}

kernel void llmopt_attention_tree_speculative_f16(
    device const half* query [[buffer(0)]],
    device const half* key [[buffer(1)]],
    device const half* value [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant SpeculativeAttentionParams& params [[buffer(4)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = threadgroup_position.x * ATTENTION_ROWS_PER_THREADGROUP + simdgroup;
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
  const uint kv_head_base = (batch * effective_kv_heads + kv_head) * params.key_length;

  float maximum = -INFINITY;
  float denominator = 0.0f;

  for (uint key_position = 0; key_position < params.key_length; ++key_position) {
    if (!llmopt_speculative_tree_allowed(params, query_position, key_position)) continue;
    const uint key_base = (kv_head_base + key_position) * params.head_dimension;
    float partial_score = 0.0f;
    for (uint d = lane; d < params.head_dimension; d += 32u) {
      partial_score += float(query[query_base + d]) * float(key[key_base + d]);
    }
    const float score = simd_sum(partial_score) * params.scale;
    const float next_maximum = max(maximum, score);
    const float prev_scale = (denominator == 0.0f) ? 0.0f : exp(maximum - next_maximum);
    const float curr_scale = exp(score - next_maximum);
    denominator = denominator * prev_scale + curr_scale;
    maximum = next_maximum;
  }

  for (uint d = lane; d < params.head_dimension; d += 32u) {
    float result = 0.0f;
    if (denominator != 0.0f) {
      for (uint key_position = 0; key_position < params.key_length; ++key_position) {
        if (!llmopt_speculative_tree_allowed(params, query_position, key_position)) continue;
        const uint key_base = (kv_head_base + key_position) * params.head_dimension;
        float partial_score = 0.0f;
        for (uint kd = 0; kd < params.head_dimension; ++kd) {
          partial_score += float(query[query_base + kd]) * float(key[key_base + kd]);
        }
        const float score = partial_score * params.scale;
        const float prob = exp(score - maximum) / denominator;
        const uint value_base = (((batch * params.key_length + key_position) * effective_kv_heads
            + kv_head) * params.head_dimension);
        result += prob * float(value[value_base + d]);
      }
    }
    const uint output_base = (((batch * params.query_length + query_position) * params.heads + head)
        * params.head_dimension);
    output[output_base + d] = half(result);
  }
}

kernel void llmopt_attention_tree_speculative_f16_simd_h64(
    device const half* query [[buffer(0)]],
    device const half* key [[buffer(1)]],
    device const half* value [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant SpeculativeAttentionParams& params [[buffer(4)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = threadgroup_position.x * ATTENTION_ROWS_PER_THREADGROUP + simdgroup;
  const uint count = params.batches * params.heads * params.query_length;
  if (row >= count) return;
  const uint query_position = row % params.query_length;
  const uint head = (row / params.query_length) % params.heads;
  const uint batch = row / (params.query_length * params.heads);
  const uint effective_kv_heads = params.kv_heads > 0 ? params.kv_heads : params.heads;
  const uint kv_head = effective_kv_heads < params.heads
      ? head / (params.heads / effective_kv_heads) : head;

  const uint query_base = (((batch * params.heads + head)
      * params.query_length + query_position) * 64u);
  const float query_low = float(query[query_base + lane]);
  const float query_high = float(query[query_base + lane + 32u]);
  const uint kv_head_base = (batch * effective_kv_heads + kv_head) * params.key_length;

  float maximum = -INFINITY;
  float denominator = 0.0f;
  float result_low = 0.0f;
  float result_high = 0.0f;

  for (uint key_position = 0; key_position < params.key_length; ++key_position) {
    if (!llmopt_speculative_tree_allowed(params, query_position, key_position)) continue;

    const uint key_base = (kv_head_base + key_position) * 64u;
    const uint value_base = (((batch * params.key_length + key_position) * effective_kv_heads
        + kv_head) * 64u);

    const float partial_score = query_low * float(key[key_base + lane])
        + query_high * float(key[key_base + lane + 32u]);
    const float score = simd_sum(partial_score) * params.scale;
    const float next_maximum = max(maximum, score);
    const float prev_scale = (denominator == 0.0f) ? 0.0f : exp(maximum - next_maximum);
    const float curr_scale = exp(score - next_maximum);

    result_low = result_low * prev_scale + curr_scale * float(value[value_base + lane]);
    result_high = result_high * prev_scale + curr_scale * float(value[value_base + lane + 32u]);
    denominator = denominator * prev_scale + curr_scale;
    maximum = next_maximum;
  }

  const uint output_base = (((batch * params.query_length + query_position) * params.heads + head) * 64u);
  output[output_base + lane] = half(denominator == 0.0f ? 0.0f : result_low / denominator);
  output[output_base + lane + 32u] = half(denominator == 0.0f ? 0.0f : result_high / denominator);
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
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_attention_tree_speculative_f16_simd_h64"
      ~operation:Kernel_abi.Operation.Attention
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_attention_tree_speculative_f16"
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

let has_l2_norm_slice graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Primitive (Ir.Primitive.L2_norm_slice _) -> true
         | _ -> false)

let has_rms_rope graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with Ir.Op.Rms_rope _ -> true | _ -> false)

let has_rms_rope_qk graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with Ir.Op.Rms_rope_qk _ -> true | _ -> false)

let has_rotary_qk graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with Ir.Op.Rotary_qk _ -> true | _ -> false)

let has_short_conv graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Primitive (Ir.Primitive.Short_conv _) -> true
         | _ -> false)

let has_short_conv_silu graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Primitive (Ir.Primitive.Short_conv_silu _) -> true
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

let has_quant_qkv_linear graph =
  Ir.Graph.nodes graph
  |> List.exists (fun node ->
         match Ir.node_op node with
         | Ir.Op.Quant_qkv_linear _ -> true
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
         | ( Ir.Op.Linear_add _,
             [ _input; weight; _residual ],
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
                 (Ir.Dtype.Q4_K | Ir.Dtype.Q5_K | Ir.Dtype.Q5_0
                 | Ir.Dtype.Q4_0 | Ir.Dtype.IQ4_XS) ->
                 true
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
                 let quant =
                   match Ir.Value.dtype weight with
                   | Ir.Dtype.Quant quant -> Some quant
                   | _ -> None
                 in
                 (match
                    quant,
                    Tactic.select_linear ~target ~m ~n ~k
                      ~input_dtype:(Ir.Value.dtype input)
                      ~storage:storage.layout
                      ~output_dtype:(Ir.Value.dtype output)
                  with
                 | Some quant, Some tactic -> Some (quant, tactic)
                 | _ -> None))
         | _ -> None)
  |> List.sort_uniq (fun (left_quant, left) (right_quant, right) ->
         match compare left_quant right_quant with
         | 0 -> String.compare (Tactic.name left) (Tactic.name right)
         | order -> order)

let quant_tactic_entries ~quant_types tactics =
  let selected_names = List.map (fun (_, tactic) -> Tactic.name tactic) tactics in
  let generic_name = function
    | Ir.Dtype.Q8_0 -> "llmopt_q8_0_linear_f16"
    | Ir.Dtype.Q4_K -> "llmopt_q4_k_linear_f16"
    | Ir.Dtype.Q5_K -> "llmopt_q5_k_linear_f16"
    | Ir.Dtype.Q6_K -> "llmopt_q6_k_linear_f16"
    | Ir.Dtype.Q5_0 -> "llmopt_q5_0_linear_f16"
    | Ir.Dtype.Q4_0 -> "llmopt_q4_0_linear_f16"
    | Ir.Dtype.IQ4_XS -> "llmopt_iq4_xs_linear_f16"
  in
  let generic_names =
    tactics |> List.map (fun (quant, _) -> generic_name quant)
    |> List.sort_uniq String.compare
  in
  block32_entries @ kquant_entries
  |> List.filter (fun entry ->
         match Kernel_abi.Entry.input_dtype entry with
         | Ir.Dtype.Quant quant when List.mem quant quant_types ->
             Kernel_abi.Entry.operation entry <> Kernel_abi.Operation.Linear
             || List.mem (Kernel_abi.Entry.name entry) selected_names
             || List.mem (Kernel_abi.Entry.name entry) generic_names
             || String.ends_with ~suffix:"_m4" (Kernel_abi.Entry.name entry)
             || String.ends_with ~suffix:"_m4_add" (Kernel_abi.Entry.name entry)
             || String.ends_with ~suffix:"_m16" (Kernel_abi.Entry.name entry)
             || String.ends_with ~suffix:"_m16_add" (Kernel_abi.Entry.name entry)
             || String.ends_with ~suffix:"_m32" (Kernel_abi.Entry.name entry)
             || String.ends_with ~suffix:"_m32_add" (Kernel_abi.Entry.name entry)
             || List.mem (Kernel_abi.Entry.name entry)
                  [ "llmopt_q5_0_linear_f16_llama_mv_add";
                    "llmopt_q8_0_linear_f16_llama_mv";
                    "llmopt_q4_k_linear_f16_llama_mv_add";
                    "llmopt_q6_k_linear_f16_llama_mv_add" ]
         | _ -> false)

let graph_quant_types graph =
  Ir.Graph.nodes graph
  |> List.concat_map Ir.node_inputs
  |> List.filter_map (fun value ->
         match Ir.Value.dtype value with
         | Ir.Dtype.Quant quant -> Some quant
         | _ -> None)
  |> List.sort_uniq Stdlib.compare

let quant_linear_add_tactics ~target graph =
  Ir.Graph.nodes graph
  |> List.filter_map (fun node ->
         match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
         | ( Ir.Op.Linear_add { m; n; k },
             [ input; weight; _residual ],
             Some output ) ->
             (match Ir.Value.dtype weight with
             | Ir.Dtype.Quant quant ->
                 Tactic.select_linear ~target ~m ~n ~k
                   ~input_dtype:(Ir.Value.dtype input)
                   ~storage:(Ir.Linear_storage.Block_quantized quant)
                   ~output_dtype:(Ir.Value.dtype output)
                 |> Option.map (fun tactic -> quant, tactic)
             | _ -> None)
         | _ -> None)
  |> List.sort_uniq (fun (_, left) (_, right) ->
         String.compare (Tactic.name left) (Tactic.name right))

let quant_linear_add_entry (quant, tactic) =
  kernel_entry_with_threadgroup ~threadgroup:(Tactic.threadgroup tactic)
    ~name:(Tactic.name tactic ^ "_add")
    ~operation:Kernel_abi.Operation.Linear
    ~input_dtype:(Ir.Dtype.Quant quant) ~output_dtype:Ir.Dtype.Float16

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

module Region_node_set = Set.Make (Int)

let region_node_ids regions =
  regions
  |> List.concat_map Kernel_region.member_node_ids
  |> Region_node_set.of_list

let without_region_nodes graph regions =
  let removed = region_node_ids regions in
  Ir.Graph.with_nodes graph
    (Ir.Graph.nodes graph
    |> List.filter (fun node ->
           not (Region_node_set.mem (Ir.node_id node) removed)))

let quant_add_twin_servable (storage : Ir.Dtype.quant_type)
    (tactic : Tactic.t) =
  (* A folded region's linear is served by the legacy textual _add twin only
     when the static kernel zoo (legacy_quant_source) really contains the base
     kernel the textual rename rewrites.  The Q4_0 x8/x4 decode kernels exist
     only in the family source (quant_source), not in the static zoo, so
     regions whose costed tactic is one of them have no legacy twin and must
     instead be served by a digest-keyed emitter kernel; every other
     quantization keeps its pre-existing twin handling untouched. *)
  match storage with
  | Ir.Dtype.Q4_0 ->
      let name = Tactic.name tactic in
      not
        (String.ends_with ~suffix:"_x8" name
        || String.ends_with ~suffix:"_x4" name)
  | _ -> true

let is_residual_add_epilogue epilogue =
  match epilogue with
  | Epilogue.Residual_add _ -> true
  | _ -> false

let region_tactics ~target regions =
  let select_linear linear output =
    Tactic.select_linear ~target ~m:linear.Kernel_ir.Fused_tactic.m
      ~n:linear.n ~k:linear.k ~input_dtype:(Ir.Value.dtype linear.input)
      ~storage:linear.storage ~output_dtype:(Ir.Value.dtype output)
  in
  regions
  |> List.fold_left
       (fun (quant, quant_add, quant_digest, has_w4, has_rms, has_rms_rope,
             has_short_conv_silu, gated_delta) region ->
         (* G4P12-02: route single-accumulator select through the
            elaboration - the legacy recognizer wins first and the widening
            only adds single-slot branch shapes ([y*y] -> pure monoid
            [Expr] epilogues served by the digest emitter). *)
         match Contraction_group.select region with
         | Some
             (Kernel_ir.Fused_tactic.Linear_epilogue
               { linear; epilogue; output }) ->
             (match linear.storage, select_linear linear output with
             | Ir.Linear_storage.Block_quantized storage, Some tactic ->
                 let quant = (storage, tactic) :: quant in
                 if
                   quant_add_twin_servable storage tactic
                   && is_residual_add_epilogue epilogue
                 then
                   quant, (storage, tactic) :: quant_add, quant_digest,
                   has_w4, has_rms, has_rms_rope, has_short_conv_silu,
                   gated_delta
                 else
                   quant, quant_add, (storage, tactic, epilogue) :: quant_digest,
                   has_w4, has_rms, has_rms_rope, has_short_conv_silu,
                   gated_delta
             | Ir.Linear_storage.Groupwise_packed _, _ ->
                 quant, quant_add, quant_digest, true, has_rms, has_rms_rope,
                 has_short_conv_silu, gated_delta
             | _ -> quant, quant_add, quant_digest, has_w4, has_rms,
                 has_rms_rope, has_short_conv_silu, gated_delta)
         | Some
             (Kernel_ir.Fused_tactic.Gated_product
               { gate; up; output; _ }) ->
             let add linear quant =
               match linear.Kernel_ir.Fused_tactic.storage,
                 select_linear linear output
               with
               | Ir.Linear_storage.Block_quantized storage, Some tactic ->
                   (storage, tactic) :: quant
               | _ -> quant
             in
             let quant = quant |> add gate |> add up in
             let has_w4 =
               has_w4
               || (match gate.storage, up.storage with
                  | ( Ir.Linear_storage.Groupwise_packed _,
                      Ir.Linear_storage.Groupwise_packed _ ) -> true
                  | _ -> false)
             in
             quant, quant_add, quant_digest, has_w4, has_rms, has_rms_rope,
             has_short_conv_silu, gated_delta
         | Some (Kernel_ir.Fused_tactic.Rms_normalize _) ->
             quant, quant_add, quant_digest, has_w4, true, has_rms_rope,
             has_short_conv_silu, gated_delta
         | Some (Kernel_ir.Fused_tactic.Rms_rope _) ->
             quant, quant_add, quant_digest, has_w4, has_rms, true,
             has_short_conv_silu, gated_delta
         | Some (Kernel_ir.Fused_tactic.Short_conv_silu _) ->
             quant, quant_add, quant_digest, has_w4, has_rms, has_rms_rope,
             true, gated_delta
         | Some (Kernel_ir.Fused_tactic.Gated_delta { query; _ }) ->
             let gated_delta =
               match
                 Ir.Value.dtype query,
                 Tensor_shape.dimensions (Ir.Value.logical_shape query)
               with
               | Ir.Dtype.Float32, [ _; _; _; width ] ->
                   Gated_delta_head_major_f32 width :: gated_delta
               | _ -> gated_delta
             in
             quant, quant_add, quant_digest, has_w4, has_rms, has_rms_rope,
             has_short_conv_silu, gated_delta
         | None ->
             quant, quant_add, quant_digest, has_w4, has_rms, has_rms_rope,
             has_short_conv_silu, gated_delta)
       ([], [], [], false, false, false, false, [])
  |> fun (quant, quant_add, quant_digest, has_w4, has_rms, has_rms_rope,
          has_short_conv_silu, gated_delta) ->
  let compare_digest (storage, tactic, epilogue)
      (storage', tactic', epilogue') =
    let c = Stdlib.compare storage storage' in
    if c <> 0 then c
    else
      let c = String.compare (Tactic.name tactic) (Tactic.name tactic') in
      if c <> 0 then c
      else
        String.compare
          (Epilogue.algebraic_kernel_name ~base:(Tactic.name tactic) epilogue)
          (Epilogue.algebraic_kernel_name ~base:(Tactic.name tactic') epilogue')
  in
  ( List.sort_uniq Stdlib.compare quant,
    List.sort_uniq Stdlib.compare quant_add,
    List.sort_uniq compare_digest quant_digest,
    has_w4, has_rms, has_rms_rope, has_short_conv_silu,
    List.sort_uniq Stdlib.compare gated_delta )

let lower ?(target = Target_hardware.default) ?(fusion_regions = []) graph =
  let graph = without_region_nodes graph fusion_regions in
  let ( region_quant_tactics, region_quant_add_tactics,
        region_quant_digest_tactics, regions_have_w4, regions_have_rms,
        regions_have_rms_rope, regions_have_short_conv_silu,
        region_gated_delta_tactics ) =
    region_tactics ~target fusion_regions
  in
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
  let quant_tactics =
    quant_linear_tactics ~target graph @ region_quant_tactics
    |> List.sort_uniq Stdlib.compare
  in
  let quant_types =
    graph_quant_types graph
    @ List.map fst region_quant_tactics
    @ List.map fst region_quant_add_tactics
    |> List.sort_uniq Stdlib.compare
  in
  let selected_quant_entries =
    quant_tactic_entries ~quant_types quant_tactics
  in
  let quant_qkv_types =
    Ir.Graph.nodes graph
    |> List.filter_map (fun node ->
           match Ir.node_op node, Ir.node_inputs node with
           | Ir.Op.Quant_qkv_linear _, _ :: q_weight :: _ ->
               (match Ir.Value.dtype q_weight with
               | Ir.Dtype.Quant quant -> Some quant
               | _ -> None)
           | _ -> None)
    |> List.sort_uniq Stdlib.compare
  in
  let quant_qkv = quant_qkv_types <> [] in
  let quant_qkv_source =
    if not quant_qkv then ""
    else
      quant_qkv_params_source
      ^ (if List.mem Ir.Dtype.Q4_0 quant_qkv_types then q4_0_qkv_source else "")
      ^ (if List.mem Ir.Dtype.Q5_0 quant_qkv_types then q5_0_qkv_source else "")
  in
  let quant_qkv_entries =
    (if List.mem Ir.Dtype.Q4_0 quant_qkv_types then q4_0_qkv_entries else [])
    @ (if List.mem Ir.Dtype.Q5_0 quant_qkv_types then q5_0_qkv_entries else [])
  in
  let quant_add_tactics =
    quant_linear_add_tactics ~target graph @ region_quant_add_tactics
    |> List.sort_uniq Stdlib.compare
  in
  let quant_add_entries = List.map quant_linear_add_entry quant_add_tactics in
  let q4_0_digest_schema =
    match
      Block_schema.of_storage (Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_0)
    with
    | Ok schema -> schema
    | Error err -> failwith (Block_schema.Error.to_string err)
  in
  let q8_0_twin_schema =
    match
      Block_schema.of_storage (Ir.Linear_storage.Block_quantized Ir.Dtype.Q8_0)
    with
    | Ok schema -> schema
    | Error err -> failwith (Block_schema.Error.to_string err)
  in
  let q5_0_twin_schema =
    match
      Block_schema.of_storage (Ir.Linear_storage.Block_quantized Ir.Dtype.Q5_0)
    with
    | Ok schema -> schema
    | Error err -> failwith (Block_schema.Error.to_string err)
  in
  let q4_k_twin_schema =
    match
      Block_schema.of_storage (Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_K)
    with
    | Ok schema -> schema
    | Error err -> failwith (Block_schema.Error.to_string err)
  in
  let q5_k_twin_schema =
    match
      Block_schema.of_storage (Ir.Linear_storage.Block_quantized Ir.Dtype.Q5_K)
    with
    | Ok schema -> schema
    | Error err -> failwith (Block_schema.Error.to_string err)
  in
  let q6_k_twin_schema =
    match
      Block_schema.of_storage (Ir.Linear_storage.Block_quantized Ir.Dtype.Q6_K)
    with
    | Ok schema -> schema
    | Error err -> failwith (Block_schema.Error.to_string err)
  in
  let iq4_xs_twin_schema =
    match
      Block_schema.of_storage
        (Ir.Linear_storage.Block_quantized Ir.Dtype.IQ4_XS)
    with
    | Ok schema -> schema
    | Error err -> failwith (Block_schema.Error.to_string err)
  in
  (* G4P12-04d-A2 + G4P12-05: the _add twins are all structural.  Q4_0/
     Q8_0/Q5_0 and the superblock families' generic/_m2 twins (and IQ4_XS
     _m2) plus the K families' m2_x4 twins (05-2) are emitted by
     Metal_block_codegen (parametric-style, double-rounding store); the Q4_0
     m4/m16/m32 twins ship in the unconditional q4_0_linear family text.
     The superblock families serve the generic tile for every m <> 2
     Linear-add shape and the m2_x4 four-column tile at m = 2 (parametric
     since 05-2; llama m2_x4 bodies deleted); IQ4_XS serves the plain _m2
     twin at m = 2.  Q4_0 x8/x4 bases never had twins and have no digest
     route for graph-level Linear_add nodes, so they fail loudly with a
     clear Error (the retired text-patcher raised Invalid_argument inside
     extract_kernel instead).  The legacy shim below is the defensive
     fallback only. *)
  let quant_add_source_result =
    quant_add_tactics
    |> List.fold_left
         (fun result (quant, tactic) ->
           let* source = result in
           match quant, Tactic.name tactic with
           | Ir.Dtype.Q4_0, "llmopt_q4_0_linear_f16" ->
               let* kernel =
                 Metal_block_codegen.q4_0_linear_add_kernel
                   ~name:"llmopt_q4_0_linear_f16_add" ~target q4_0_digest_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Q4_0, "llmopt_q4_0_linear_f16_m2" ->
               let* kernel =
                 Metal_block_codegen.q4_0_m2_add_kernel
                   ~name:"llmopt_q4_0_linear_f16_m2_add" ~target
                   q4_0_digest_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Q8_0, "llmopt_q8_0_linear_f16" ->
               let* kernel =
                 Metal_block_codegen.q8_0_linear_add_kernel
                   ~name:"llmopt_q8_0_linear_f16_add" ~target q8_0_twin_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Q8_0, "llmopt_q8_0_linear_f16_m2" ->
               let* kernel =
                 Metal_block_codegen.q8_0_m2_add_kernel
                   ~name:"llmopt_q8_0_linear_f16_m2_add" ~target
                   q8_0_twin_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Q5_0, "llmopt_q5_0_linear_f16" ->
               let* kernel =
                 Metal_block_codegen.q5_0_linear_add_kernel
                   ~name:"llmopt_q5_0_linear_f16_add" ~target q5_0_twin_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Q5_0, "llmopt_q5_0_linear_f16_m2" ->
               let* kernel =
                 Metal_block_codegen.q5_0_m2_add_kernel
                   ~name:"llmopt_q5_0_linear_f16_m2_add" ~target
                   q5_0_twin_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Q4_K, "llmopt_q4_k_linear_f16" ->
               let* kernel =
                 Metal_block_codegen.q4_k_linear_add_kernel
                   ~name:"llmopt_q4_k_linear_f16_add" ~target q4_k_twin_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Q5_K, "llmopt_q5_k_linear_f16" ->
               let* kernel =
                 Metal_block_codegen.q5_k_linear_add_kernel
                   ~name:"llmopt_q5_k_linear_f16_add" ~target q5_k_twin_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Q6_K, "llmopt_q6_k_linear_f16" ->
               let* kernel =
                 Metal_block_codegen.q6_k_linear_add_kernel
                   ~name:"llmopt_q6_k_linear_f16_add" ~target q6_k_twin_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.IQ4_XS, "llmopt_iq4_xs_linear_f16" ->
               let* kernel =
                 Metal_block_codegen.iq4_xs_linear_add_kernel
                   ~name:"llmopt_iq4_xs_linear_f16_add" ~target
                   iq4_xs_twin_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.IQ4_XS, "llmopt_iq4_xs_linear_f16_m2" ->
               let* kernel =
                 Metal_block_codegen.iq4_xs_m2_add_kernel
                   ~name:"llmopt_iq4_xs_linear_f16_m2_add" ~target
                   iq4_xs_twin_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Q4_K, "llmopt_q4_k_linear_f16_m2_x4" ->
               let* kernel =
                 Metal_block_codegen.q4_k_m2_x4_add_kernel
                   ~name:"llmopt_q4_k_linear_f16_m2_x4_add" ~target
                   q4_k_twin_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Q5_K, "llmopt_q5_k_linear_f16_m2_x4" ->
               let* kernel =
                 Metal_block_codegen.q5_k_m2_x4_add_kernel
                   ~name:"llmopt_q5_k_linear_f16_m2_x4_add" ~target
                   q5_k_twin_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Q6_K, "llmopt_q6_k_linear_f16_m2_x4" ->
               let* kernel =
                 Metal_block_codegen.q6_k_m2_x4_add_kernel
                   ~name:"llmopt_q6_k_linear_f16_m2_x4_add" ~target
                   q6_k_twin_schema
               in
               Ok (source ^ kernel)
           | Ir.Dtype.Q4_0, base
             when String.ends_with ~suffix:"_m4" base
                  || String.ends_with ~suffix:"_m16" base
                  || String.ends_with ~suffix:"_m32" base ->
               (* the q4_0_linear zoo text already carries these twins
                  unconditionally for every Q4_0 program *)
               Ok source
           | Ir.Dtype.Q4_0, base ->
               Error
                 (Printf.sprintf
                    "Metal lower: Q4_0 graph-level Linear-add on the %s tile \
                     has no legacy _add twin (x8/x4 twins never existed); \
                     lower it as a fused Linear_epilogue region so the digest \
                     route serves it"
                    base)
           | _ -> Ok (source ^ quant_linear_add_source ~target (quant, tactic)))
         (Ok "")
  in
  let region_quant_digest_source_result =
    region_quant_digest_tactics
    |> List.fold_left
         (fun result (quant, tactic, epilogue) ->
           let* source = result in
           let base = Tactic.name tactic in
           let kernel_name = Epilogue.algebraic_kernel_name ~base epilogue in
           let q4_0_digest_emitter base =
             if String.ends_with ~suffix:"_x8" base then
               Some Metal_block_codegen.q4_0_x8_epilogue_kernel
             else if String.ends_with ~suffix:"_x4" base then
               Some Metal_block_codegen.q4_0_x4_epilogue_kernel
             else if String.ends_with ~suffix:"_m16" base then
               Some Metal_block_codegen.q4_0_m16_epilogue_kernel
             else if String.ends_with ~suffix:"_m32" base then
               Some Metal_block_codegen.q4_0_m32_epilogue_kernel
             else None
           in
           match quant, q4_0_digest_emitter base with
           | Ir.Dtype.Q4_0, Some emit ->
               let* kernel =
                 (* The schema-level emitters, like the other parametric
                    emitters, expect an already-canonicalized epilogue; the
                    retired splice canonicalized internally, so canonicalize
                    here to keep byte-identical output.  kernel_name is still
                    derived from the tactic epilogue so the ABI entry name
                    registered below agrees with the emitted kernel text. *)
                 emit ~name:kernel_name
                   ~epilogue:(Epilogue.canonicalize epilogue) ~target
                   q4_0_digest_schema
               in
               Ok (source ^ kernel)
           | _ ->
               Error
                 (Printf.sprintf
                    "Metal lower: digest epilogue emission is defined only for \
                     Q4_0 x8/x4/m16/m32 decode regions (got %s serving %s)"
                    (Ir.Dtype.quant_to_string quant) base))
         (Ok "")
  in
  let region_quant_digest_entries =
    region_quant_digest_tactics
    |> List.map (fun (quant, tactic, epilogue) ->
           kernel_entry_with_threadgroup ~threadgroup:(Tactic.threadgroup tactic)
             ~name:(Epilogue.algebraic_kernel_name ~base:(Tactic.name tactic) epilogue)
             ~operation:Kernel_abi.Operation.Linear
             ~input_dtype:(Ir.Dtype.Quant quant)
             ~output_dtype:Ir.Dtype.Float16)
  in
  let gated_delta_tactics =
    gated_delta_tactics graph @ region_gated_delta_tactics
    |> List.sort_uniq Stdlib.compare
  in
  let gated_delta_source =
    gated_delta_tactics |> List.map gated_delta_source |> String.concat "\n"
  in
  let gated_delta_entries = List.map gated_delta_entry gated_delta_tactics in
  let* quant_format_source =
    quant_types
    |> List.fold_left
         (fun result quant ->
           let* source = result in
           let* component = quant_source ~target quant in
           Ok (source ^ component))
         (Ok "")
  in
  let* region_quant_digest_source =
    region_quant_digest_source_result
  in
  let* quant_add_source = quant_add_source_result in
  let components =
    [ ( has_w4a16_linear graph || regions_have_w4,
        w4a16_source,
        w4a16_entries );
      ( has_w4a16_qkv_linear graph,
        w4a16_qkv_linear_source,
        w4a16_qkv_linear_entries );
      ( has_w4a16_swiglu_ffn graph,
        w4a16_swiglu_ffn_source,
        w4a16_swiglu_ffn_entries );
      ( has_w4a16_lm_head_argmax graph,
        w4a16_lm_head_argmax_source,
        w4a16_lm_head_argmax_entries );
      ( has_quant_linear graph || region_quant_tactics <> [] || quant_qkv,
        quant_common_source ^ quant_format_source ^ quant_qkv_source
        ^ quant_add_source ^ region_quant_digest_source,
        selected_quant_entries @ quant_qkv_entries @ quant_add_entries
        @ region_quant_digest_entries );
      has_f16_linear graph, linear_f16_source, linear_f16_entries;
      ( has_f16_bf16_linear graph,
        linear_f16_bf16_source,
        linear_f16_bf16_entries );
      ( has_f16_f32_linear graph,
        linear_f16_f32_source,
        linear_f16_f32_entries );
      (has_rms_norm graph || regions_have_rms), rms_norm_source, rms_norm_entries;
      has_l2_norm graph, l2_norm_source, l2_norm_entries;
      has_l2_norm_slice graph, l2_norm_slice_source, l2_norm_slice_entries;
      (has_rms_rope graph || regions_have_rms_rope), rms_rope_source,
      rms_rope_entries;
      has_rms_rope_qk graph, rms_rope_qk_source, rms_rope_qk_entries;
      has_rotary_qk graph, rotary_qk_source, rotary_qk_entries;
      ( has_short_conv graph,
        short_conv_source ^ short_conv_f32_weight_source,
        short_conv_entries );
      ( has_short_conv_silu graph || regions_have_short_conv_silu,
        short_conv_silu_source,
        short_conv_silu_entries );
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

let emit_parametric_w4a16_linear ~name ~epilogue =
  let epilogue_code = Epilogue.emit_msl_writeback ~acc:"acc" ~idx:"col" epilogue in
  let epilogue_buffers = Epilogue.msl_buffer_decls ~start_index:6 epilogue in
  Printf.sprintf "kernel void %s(\n\
   \    device const half* input [[buffer(0)]],\n\
   \    device const uchar* packed_weight [[buffer(1)]],\n\
   \    device const half* scale [[buffer(2)]],\n\
   \    device const half* bias_or_scale [[buffer(3)]],\n\
   \    device half* output [[buffer(4)]],\n\
   \    constant W4A16Params& params [[buffer(5)]]%s,\n\
   \    uint gid [[thread_position_in_grid]],\n\
   \    uint lane [[thread_index_in_simdgroup]]) {\n\
   \  const uint simd_idx = gid >> 5;\n\
   \  const uint total_elements = params.m * params.n;\n\
   \  if (simd_idx >= total_elements) return;\n\
   \  const uint row = simd_idx / params.n;\n\
   \  const uint col = simd_idx - row * params.n;\n\
   \  const uint groups = params.k >> 6;\n\
   \  const uint packed_base = col * (params.k >> 1);\n\
   \  const uint scale_base = col * groups;\n\
   \  const uint input_base = row * params.k;\n\
   \  float acc = 0.0f;\n\
   \  for (uint g = 0; g < groups; ++g) {\n\
   \    const float s = float(scale[scale_base + g]);\n\
   \    const uchar packed = packed_weight[packed_base + (g << 5) + lane];\n\
   \    const int q0 = int(packed & 15u) - ((packed & 8u) ? 16 : 0);\n\
   \    const int q1 = int(packed >> 4) - ((packed & 128u) ? 16 : 0);\n\
   \    const uint in_idx = input_base + (g << 6) + (lane << 1);\n\
   \    const half2 in_val = *reinterpret_cast<device const half2*>(input + in_idx);\n\
   \    acc += (float(in_val.x) * float(q0) + float(in_val.y) * float(q1)) * s;\n\
   \  }\n\
   \  acc = simd_sum(acc);\n\
   \  if (lane == 0) {\n\
   \    output[row * params.n + col] = half(%s);\n\
   \  }\n\
   }\n" name epilogue_buffers epilogue_code

let emit_parametric_q4_k_linear ~name ~epilogue =
  let epilogue_code = Epilogue.emit_msl_writeback ~acc:"acc" ~idx:"col" epilogue in
  let epilogue_buffers = Epilogue.msl_buffer_decls ~start_index:6 epilogue in
  Printf.sprintf "kernel void %s(\n\
   \    device const half* input [[buffer(0)]],\n\
   \    device const block_q4_K* weight [[buffer(1)]],\n\
   \    device half* output [[buffer(2)]],\n\
   \    constant uint& num_superblocks [[buffer(3)]],\n\
   \    constant uint& total_elements [[buffer(4)]],\n\
   \    constant uint& has_bias [[buffer(5)]]%s,\n\
   \    uint gid [[thread_position_in_grid]],\n\
   \    uint lane [[thread_index_in_simdgroup]]) {\n\
   \  const uint simd_idx = gid >> 5;\n\
   \  if (simd_idx >= total_elements) return;\n\
   \  const uint row = simd_idx / total_elements;\n\
   \  const uint col = simd_idx - row * total_elements;\n\
   \  device const block_q4_K* row_blocks = weight + col * num_superblocks;\n\
   \  float acc = 0.0f;\n\
   \  for (uint sb = 0; sb < num_superblocks; ++sb) {\n\
   \    device const block_q4_K& b = row_blocks[sb];\n\
   \    const float d = float(b.d);\n\
   \    const float dmin = float(b.dmin);\n\
   \    device const uchar* scales = b.scales;\n\
   \    const uint sc_idx = lane >> 2;\n\
   \    const uint sc_byte = scales[sc_idx];\n\
   \    const float scale_val = float(sc_byte & 63u) * d;\n\
   \    const float min_val = float(scales[sc_idx + 4] & 63u) * dmin;\n\
   \    const uint q_idx = lane << 2;\n\
   \    device const uchar* q = b.quants + q_idx;\n\
   \    device const half* in_ptr = input + (sb << 8) + q_idx;\n\
   \    for (uint i = 0; i < 4; ++i) {\n\
   \      const uchar byte_val = q[i];\n\
   \      const float q0 = float(byte_val & 15u);\n\
   \      const float q1 = float(byte_val >> 4);\n\
   \      acc += (q0 * scale_val - min_val) * float(in_ptr[i * 2]);\n\
   \      acc += (q1 * scale_val - min_val) * float(in_ptr[i * 2 + 1]);\n\
   \    }\n\
   \  }\n\
   \  acc = simd_sum(acc);\n\
   \  if (lane == 0) {\n\
   \    output[row * total_elements + col] = half(%s);\n\
   \  }\n\
   }\n" name epilogue_buffers epilogue_code

let emit_simdgroup_matrix_gemm ~name ~epilogue =
  let epilogue_code = Epilogue.emit_msl_writeback ~acc:"val" ~idx:"c" epilogue in
  let epilogue_buffers = Epilogue.msl_buffer_decls ~start_index:4 epilogue in
  Printf.sprintf "struct GEMMParams { uint m; uint n; uint k; };\n\n\
   kernel void %s(\n\
   \    device const half* lhs [[buffer(0)]],\n\
   \    device const half* rhs [[buffer(1)]],\n\
   \    device half* output [[buffer(2)]],\n\
   \    constant GEMMParams& params [[buffer(3)]]%s,\n\
   \    uint3 group_position [[threadgroup_position_in_grid]],\n\
   \    uint lane [[thread_index_in_simdgroup]]) {\n\
   \  const uint row_base = group_position.y * 8;\n\
   \  const uint col_base = group_position.x * 8;\n\
   \  // Swizzle padded stride to eliminate Apple Silicon threadgroup bank conflicts\n\
   \  threadgroup half lhs_tile[8][12];\n\
   \  threadgroup half rhs_tile[8][12];\n\
   \  simdgroup_half8x8 accumulator = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);\n\
   \  const uint element0 = lane * 2;\n\
   \  const uint element1 = element0 + 1;\n\
   \  const uint row0 = element0 >> 3;\n\
   \  const uint col0 = element0 & 7;\n\
   \  const uint row1 = element1 >> 3;\n\
   \  const uint col1 = element1 & 7;\n\
   \  for (uint inner_base = 0; inner_base < params.k; inner_base += 8) {\n\
   \    const uint lhs_row0 = row_base + row0;\n\
   \    const uint lhs_inner0 = inner_base + col0;\n\
   \    lhs_tile[row0][col0] = (lhs_row0 < params.m && lhs_inner0 < params.k)\n\
   \        ? lhs[lhs_row0 * params.k + lhs_inner0] : 0.0h;\n\
   \    const uint lhs_row1 = row_base + row1;\n\
   \    const uint lhs_inner1 = inner_base + col1;\n\
   \    lhs_tile[row1][col1] = (lhs_row1 < params.m && lhs_inner1 < params.k)\n\
   \        ? lhs[lhs_row1 * params.k + lhs_inner1] : 0.0h;\n\
   \    const uint rhs_inner0 = inner_base + row0;\n\
   \    const uint rhs_col0 = col_base + col0;\n\
   \    rhs_tile[row0][col0] = (rhs_inner0 < params.k && rhs_col0 < params.n)\n\
   \        ? rhs[rhs_inner0 * params.n + rhs_col0] : 0.0h;\n\
   \    const uint rhs_inner1 = inner_base + row1;\n\
   \    const uint rhs_col1 = col_base + col1;\n\
   \    rhs_tile[row1][col1] = (rhs_inner1 < params.k && rhs_col1 < params.n)\n\
   \        ? rhs[rhs_inner1 * params.n + rhs_col1] : 0.0h;\n\
   \    simdgroup_barrier(mem_flags::mem_threadgroup);\n\
   \    simdgroup_half8x8 lhs_matrix;\n\
   \    simdgroup_half8x8 rhs_matrix;\n\
   \    simdgroup_load(lhs_matrix, &lhs_tile[0][0], 12);\n\
   \    simdgroup_load(rhs_matrix, &rhs_tile[0][0], 12);\n\
   \    simdgroup_multiply_accumulate(accumulator, lhs_matrix, rhs_matrix, accumulator);\n\
   \    simdgroup_barrier(mem_flags::mem_threadgroup);\n\
   \  }\n\
   \  threadgroup half output_tile[8][12];\n\
   \  simdgroup_store(accumulator, &output_tile[0][0], 12);\n\
   \  simdgroup_barrier(mem_flags::mem_threadgroup);\n\
   \  const uint out_row0 = row_base + row0;\n\
   \  const uint out_col0 = col_base + col0;\n\
   \  if (out_row0 < params.m && out_col0 < params.n) {\n\
   \    const float val = float(output_tile[row0][col0]);\n\
   \    const uint c = out_col0;\n\
   \    output[out_row0 * params.n + out_col0] = half(%s);\n\
   \  }\n\
   \  const uint out_row1 = row_base + row1;\n\
   \  const uint out_col1 = col_base + col1;\n\
   \  if (out_row1 < params.m && out_col1 < params.n) {\n\
   \    const float val = float(output_tile[row1][col1]);\n\
   \    const uint c = out_col1;\n\
   \    output[out_row1 * params.n + out_col1] = half(%s);\n\
   \  }\n\
   }\n" name epilogue_buffers epilogue_code epilogue_code
