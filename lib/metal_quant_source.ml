(** Metal quant source selection and entry assembly for the canonical
    Tile32_Affine{4,8} runtime family and Q4_0 baseline. *)

open Metal_source_common

let ( let* ) = Result.bind

module Tactic = Metal_selection.Tactic

let q4_0_source = Metal_q4_0_source.q4_0_source

let quant_common_source = {|
struct QuantBlock32Params {
    uint total_blocks;
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

let find_substring_from source pattern offset =
  let source_length = String.length source in
  let pattern_length = String.length pattern in
  let rec search index =
    if index + pattern_length > source_length then None
    else if String.sub source index pattern_length = pattern then Some index
    else search (index + 1)
  in
  search offset

let replace_all source ~pattern ~replacement =
  let pattern_length = String.length pattern in
  let buffer = Buffer.create (String.length source) in
  let rec copy offset =
    match find_substring_from source pattern offset with
    | None ->
        Buffer.add_substring buffer source offset (String.length source - offset)
    | Some index ->
        Buffer.add_substring buffer source offset (index - offset);
        Buffer.add_string buffer replacement;
        copy (index + pattern_length)
  in
  copy 0;
  Buffer.contents buffer

let extract_kernel source name =
  let signature = "kernel void " ^ name ^ "(" in
  match find_substring_from source signature 0 with
  | None -> invalid_arg ("missing quantized Linear kernel source: " ^ name)
  | Some start ->
      let stop =
        match find_substring_from source "\nkernel void " (start + 1) with
        | Some stop -> stop
        | None -> String.length source
      in
      String.sub source start (stop - start)

let remove_kernel name source =
  let signature = "kernel void " ^ name ^ "(" in
  match find_substring_from source signature 0 with
  | None -> source
  | Some start ->
      let stop =
        match find_substring_from source "\nkernel void " (start + 1) with
        | Some stop -> stop
        | None -> String.length source
      in
      String.sub source 0 start
      ^ String.sub source stop (String.length source - stop)

let quant_source ~target = function
  | (Ir.Dtype.Tile32_Affine4 | Ir.Dtype.Tile32_Affine8) as quant ->
      let* schema =
        Block_schema.of_storage
          (Ir.Linear_storage.Block_quantized quant)
        |> Result.map_error Block_schema.Error.to_string
      in
      Metal_block_codegen.linear_kernels ~target schema
  | Ir.Dtype.Q4_0 ->
      let* schema =
        Block_schema.of_storage
          (Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_0)
        |> Result.map_error Block_schema.Error.to_string
      in
      let* generated = Metal_block_codegen.linear_kernels ~target schema in
      Ok
        (generated
        ^ (q4_0_source
          |> remove_kernel "llmopt_q4_0_linear_f16"
          |> remove_kernel "llmopt_q4_0_linear_f16_m2"))
  | ( Ir.Dtype.Q8_0
    | Ir.Dtype.Q5_0
    | Ir.Dtype.Q4_K
    | Ir.Dtype.Q5_K
    | Ir.Dtype.Q6_K
    | Ir.Dtype.IQ4_XS ) as quant ->
      Error
        (Printf.sprintf
           "legacy GGUF format %s must be transcoded to Tile32_Affine{4,8} before Metal codegen"
           (Ir.Dtype.quant_to_string quant))

let emit_dequant_q4_0 () = metal_header ^ quant_common_source ^ q4_0_source

let emit_tile32_affine4 ?(target = Target_hardware.default) () =
  match quant_source ~target Ir.Dtype.Tile32_Affine4 with
  | Ok src -> metal_header ^ quant_common_source ^ src
  | Error msg -> invalid_arg msg

let emit_tile32_affine8 ?(target = Target_hardware.default) () =
  match quant_source ~target Ir.Dtype.Tile32_Affine8 with
  | Ok src -> metal_header ^ quant_common_source ^ src
  | Error msg -> invalid_arg msg

let moe_source = {|
struct MoeTopkParams {
  uint tokens;
  uint num_experts;
  uint top_k;
  uint norm_topk_prob;
};

struct MoeGateUpParams {
  uint tokens;
  uint top_k;
  uint hidden_dim;
  uint intermediate_dim;
};

struct MoeDownParams {
  uint tokens;
  uint top_k;
  uint hidden_dim;
  uint intermediate_dim;
};

kernel void llmopt_moe_topk_softmax_f16(
    device const half* logits [[buffer(0)]],
    device uint* out_indices [[buffer(1)]],
    device half* out_weights [[buffer(2)]],
    constant MoeTopkParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint token = gid >> 5u;
  if (token >= params.tokens) return;
  const uint num_experts = params.num_experts;
  const uint top_k = min(params.top_k, 8u);
  device const half* token_logits = logits + token * num_experts;

  float best_val[8];
  uint best_idx[8];

  for (uint k = 0; k < top_k; ++k) {
    float my_val = -1e30f;
    uint my_idx = 0;
    for (uint e = lane; e < num_experts; e += 32u) {
      bool already_selected = false;
      for (uint prev = 0; prev < k; ++prev) {
        if (best_idx[prev] == e) { already_selected = true; break; }
      }
      if (!already_selected) {
        float v = float(token_logits[e]);
        if (v > my_val) {
          my_val = v;
          my_idx = e;
        }
      }
    }
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
      float other_val = simd_shuffle_down(my_val, offset);
      uint other_idx = simd_shuffle_down(my_idx, offset);
      if (other_val > my_val) {
        my_val = other_val;
        my_idx = other_idx;
      }
    }
    best_val[k] = simd_broadcast(my_val, 0u);
    best_idx[k] = simd_broadcast(my_idx, 0u);
  }

  if (lane == 0u) {
    float max_s = best_val[0];
    for (uint k = 1; k < top_k; ++k) {
      if (best_val[k] > max_s) max_s = best_val[k];
    }
    float sum_exp = 0.0f;
    float exp_val[8];
    for (uint k = 0; k < top_k; ++k) {
      exp_val[k] = metal::exp(best_val[k] - max_s);
      sum_exp += exp_val[k];
    }
    const float inv_sum = 1.0f / (sum_exp > 0.0f ? sum_exp : 1.0f);
    device uint* dst_indices = out_indices + token * params.top_k;
    device half* dst_weights = out_weights + token * params.top_k;
    for (uint k = 0; k < top_k; ++k) {
      dst_indices[k] = best_idx[k];
      dst_weights[k] = params.norm_topk_prob != 0u ? half(exp_val[k] * inv_sum) : half(exp_val[k]);
    }
  }
}

kernel void llmopt_moe_topk_softmax_f32(
    device const float* logits [[buffer(0)]],
    device uint* out_indices [[buffer(1)]],
    device half* out_weights [[buffer(2)]],
    constant MoeTopkParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint token = gid >> 5u;
  if (token >= params.tokens) return;
  const uint num_experts = params.num_experts;
  const uint top_k = min(params.top_k, 8u);
  device const float* token_logits = logits + token * num_experts;

  float best_val[8];
  uint best_idx[8];

  for (uint k = 0; k < top_k; ++k) {
    float my_val = -1e30f;
    uint my_idx = 0;
    for (uint e = lane; e < num_experts; e += 32u) {
      bool already_selected = false;
      for (uint prev = 0; prev < k; ++prev) {
        if (best_idx[prev] == e) { already_selected = true; break; }
      }
      if (!already_selected) {
        float v = token_logits[e];
        if (v > my_val) {
          my_val = v;
          my_idx = e;
        }
      }
    }
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
      float other_val = simd_shuffle_down(my_val, offset);
      uint other_idx = simd_shuffle_down(my_idx, offset);
      if (other_val > my_val) {
        my_val = other_val;
        my_idx = other_idx;
      }
    }
    best_val[k] = simd_broadcast(my_val, 0u);
    best_idx[k] = simd_broadcast(my_idx, 0u);
  }

  if (lane == 0u) {
    float max_s = best_val[0];
    for (uint k = 1; k < top_k; ++k) {
      if (best_val[k] > max_s) max_s = best_val[k];
    }
    float sum_exp = 0.0f;
    float exp_val[8];
    for (uint k = 0; k < top_k; ++k) {
      exp_val[k] = metal::exp(best_val[k] - max_s);
      sum_exp += exp_val[k];
    }
    const float inv_sum = 1.0f / (sum_exp > 0.0f ? sum_exp : 1.0f);
    device uint* dst_indices = out_indices + token * params.top_k;
    device half* dst_weights = out_weights + token * params.top_k;
    for (uint k = 0; k < top_k; ++k) {
      dst_indices[k] = best_idx[k];
      dst_weights[k] = params.norm_topk_prob != 0u ? half(exp_val[k] * inv_sum) : half(exp_val[k]);
    }
  }
}

kernel void llmopt_tile32_affine4_moe_gate_up_swiglu_f16_x8(
    device const half* hidden_states [[buffer(0)]],
    device const uint* topk_indices [[buffer(1)]],
    device const uchar* gate_weights [[buffer(2)]],
    device const uchar* up_weights [[buffer(3)]],
    device half* intermediate [[buffer(4)]],
    constant MoeGateUpParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint col_groups = (params.intermediate_dim + 7u) >> 3u;
  const uint P = params.tokens * params.top_k;
  if (simd_idx >= P * col_groups) return;

  const uint p = simd_idx / col_groups;
  const uint group_idx = simd_idx - p * col_groups;
  const uint base_col = group_idx << 3u;
  if (base_col >= params.intermediate_dim) return;

  const uint token = p / params.top_k;
  const uint expert = topk_indices[p];

  const uint blocks = params.hidden_dim >> 5u;
  const uint col_stride = blocks * 20u;
  const uint expert_stride = params.intermediate_dim * col_stride;

  device const uchar* gate_base = gate_weights + expert * expert_stride;
  device const uchar* up_base = up_weights + expert * expert_stride;

  device const uchar* gate_row0 = gate_base + min(base_col, params.intermediate_dim - 1u) * col_stride;
  device const uchar* gate_row1 = gate_base + min(base_col + 1u, params.intermediate_dim - 1u) * col_stride;
  device const uchar* gate_row2 = gate_base + min(base_col + 2u, params.intermediate_dim - 1u) * col_stride;
  device const uchar* gate_row3 = gate_base + min(base_col + 3u, params.intermediate_dim - 1u) * col_stride;
  device const uchar* gate_row4 = gate_base + min(base_col + 4u, params.intermediate_dim - 1u) * col_stride;
  device const uchar* gate_row5 = gate_base + min(base_col + 5u, params.intermediate_dim - 1u) * col_stride;
  device const uchar* gate_row6 = gate_base + min(base_col + 6u, params.intermediate_dim - 1u) * col_stride;
  device const uchar* gate_row7 = gate_base + min(base_col + 7u, params.intermediate_dim - 1u) * col_stride;

  device const uchar* up_row0 = up_base + min(base_col, params.intermediate_dim - 1u) * col_stride;
  device const uchar* up_row1 = up_base + min(base_col + 1u, params.intermediate_dim - 1u) * col_stride;
  device const uchar* up_row2 = up_base + min(base_col + 2u, params.intermediate_dim - 1u) * col_stride;
  device const uchar* up_row3 = up_base + min(base_col + 3u, params.intermediate_dim - 1u) * col_stride;
  device const uchar* up_row4 = up_base + min(base_col + 4u, params.intermediate_dim - 1u) * col_stride;
  device const uchar* up_row5 = up_base + min(base_col + 5u, params.intermediate_dim - 1u) * col_stride;
  device const uchar* up_row6 = up_base + min(base_col + 6u, params.intermediate_dim - 1u) * col_stride;
  device const uchar* up_row7 = up_base + min(base_col + 7u, params.intermediate_dim - 1u) * col_stride;

  device const half* activation = hidden_states + token * params.hidden_dim;

  float acc_g0 = 0.0f, acc_g1 = 0.0f, acc_g2 = 0.0f, acc_g3 = 0.0f;
  float acc_g4 = 0.0f, acc_g5 = 0.0f, acc_g6 = 0.0f, acc_g7 = 0.0f;
  float acc_u0 = 0.0f, acc_u1 = 0.0f, acc_u2 = 0.0f, acc_u3 = 0.0f;
  float acc_u4 = 0.0f, acc_u5 = 0.0f, acc_u6 = 0.0f, acc_u7 = 0.0f;

  const uint cluster_blocks = (blocks + 7u) >> 3u;
  const uint b_sub = lane >> 2u;
  const uint t_in_b = lane & 3u;
  const uint byte_offset = 4u + (t_in_b << 2u);

  for (uint c = 0u; c < cluster_blocks; ++c) {
    const uint block_idx = (c << 3u) + b_sub;
    if (block_idx < blocks) {
      const uint blk_byte_off = block_idx * 20u;
      const uint l0 = t_in_b << 2u;
      const uint k_idx = block_idx * 32u + l0;
      const float a0_lo = float(activation[k_idx]);
      const float a0_hi = float(activation[k_idx + 16u]);
      const float a1_lo = float(activation[k_idx + 1u]);
      const float a1_hi = float(activation[k_idx + 17u]);
      const float a2_lo = float(activation[k_idx + 2u]);
      const float a2_hi = float(activation[k_idx + 18u]);
      const float a3_lo = float(activation[k_idx + 3u]);
      const float a3_hi = float(activation[k_idx + 19u]);
      const float a_sum = a0_lo + a0_hi + a1_lo + a1_hi + a2_lo + a2_hi + a3_lo + a3_hi;

      #define DECODE_G(row_v, acc_v) { \
        device const uchar* blk = row_v + blk_byte_off; \
        const float sc = float(*reinterpret_cast<device const half*>(blk)); \
        const float mn = float(*reinterpret_cast<device const half*>(blk + 2u)); \
        const ushort p_lo = *reinterpret_cast<device const ushort*>(blk + byte_offset); \
        const ushort p_hi = *reinterpret_cast<device const ushort*>(blk + byte_offset + 2u); \
        acc_v += sc * ( \
            a0_lo * float(p_lo & 15u) + \
            a0_hi * float((p_lo >> 4u) & 15u) + \
            a1_lo * float((p_lo >> 8u) & 15u) + \
            a1_hi * float((p_lo >> 12u) & 15u) + \
            a2_lo * float(p_hi & 15u) + \
            a2_hi * float((p_hi >> 4u) & 15u) + \
            a3_lo * float((p_hi >> 8u) & 15u) + \
            a3_hi * float((p_hi >> 12u) & 15u)) + mn * a_sum; \
      }

      DECODE_G(gate_row0, acc_g0);
      DECODE_G(gate_row1, acc_g1);
      DECODE_G(gate_row2, acc_g2);
      DECODE_G(gate_row3, acc_g3);
      DECODE_G(gate_row4, acc_g4);
      DECODE_G(gate_row5, acc_g5);
      DECODE_G(gate_row6, acc_g6);
      DECODE_G(gate_row7, acc_g7);

      DECODE_G(up_row0, acc_u0);
      DECODE_G(up_row1, acc_u1);
      DECODE_G(up_row2, acc_u2);
      DECODE_G(up_row3, acc_u3);
      DECODE_G(up_row4, acc_u4);
      DECODE_G(up_row5, acc_u5);
      DECODE_G(up_row6, acc_u6);
      DECODE_G(up_row7, acc_u7);
      #undef DECODE_G
    }
  }

  acc_g0 = simd_sum(acc_g0); acc_g1 = simd_sum(acc_g1);
  acc_g2 = simd_sum(acc_g2); acc_g3 = simd_sum(acc_g3);
  acc_g4 = simd_sum(acc_g4); acc_g5 = simd_sum(acc_g5);
  acc_g6 = simd_sum(acc_g6); acc_g7 = simd_sum(acc_g7);

  acc_u0 = simd_sum(acc_u0); acc_u1 = simd_sum(acc_u1);
  acc_u2 = simd_sum(acc_u2); acc_u3 = simd_sum(acc_u3);
  acc_u4 = simd_sum(acc_u4); acc_u5 = simd_sum(acc_u5);
  acc_u6 = simd_sum(acc_u6); acc_u7 = simd_sum(acc_u7);

  if (lane == 0u && base_col < params.intermediate_dim) {
    device half* out_p = intermediate + p * params.intermediate_dim + base_col;
    #define SWIGLU(g, u) ((g * (1.0f / (1.0f + metal::exp(-g)))) * u)
    out_p[0] = half(SWIGLU(acc_g0, acc_u0));
    if (base_col + 1u < params.intermediate_dim) out_p[1] = half(SWIGLU(acc_g1, acc_u1));
    if (base_col + 2u < params.intermediate_dim) out_p[2] = half(SWIGLU(acc_g2, acc_u2));
    if (base_col + 3u < params.intermediate_dim) out_p[3] = half(SWIGLU(acc_g3, acc_u3));
    if (base_col + 4u < params.intermediate_dim) out_p[4] = half(SWIGLU(acc_g4, acc_u4));
    if (base_col + 5u < params.intermediate_dim) out_p[5] = half(SWIGLU(acc_g5, acc_u5));
    if (base_col + 6u < params.intermediate_dim) out_p[6] = half(SWIGLU(acc_g6, acc_u6));
    if (base_col + 7u < params.intermediate_dim) out_p[7] = half(SWIGLU(acc_g7, acc_u7));
    #undef SWIGLU
  }
}

kernel void llmopt_tile32_affine8_moe_down_reduce_f16_x8(
    device const half* intermediate [[buffer(0)]],
    device const uint* topk_indices [[buffer(1)]],
    device const half* topk_weights [[buffer(2)]],
    device const uchar* down_weights [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant MoeDownParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint col_groups = (params.hidden_dim + 7u) >> 3u;
  if (simd_idx >= params.tokens * col_groups) return;

  const uint token = simd_idx / col_groups;
  const uint group_idx = simd_idx - token * col_groups;
  const uint base_col = group_idx << 3u;
  if (base_col >= params.hidden_dim) return;

  const uint blocks = params.intermediate_dim >> 5u;
  const uint col_stride = blocks * 36u;
  const uint expert_stride = params.hidden_dim * col_stride;

  const uint cluster_blocks = (blocks + 3u) >> 2u;
  const uint b_sub = lane >> 3u;
  const uint t_in_b = lane & 7u;
  const uint byte_offset = 4u + (t_in_b << 2u);

  float tot0 = 0.0f, tot1 = 0.0f, tot2 = 0.0f, tot3 = 0.0f;
  float tot4 = 0.0f, tot5 = 0.0f, tot6 = 0.0f, tot7 = 0.0f;

  for (uint k = 0; k < params.top_k; ++k) {
    const uint p = token * params.top_k + k;
    const uint expert = topk_indices[p];
    const float w = float(topk_weights[p]);
    if (w <= 0.0f) continue;

    device const uchar* down_base = down_weights + expert * expert_stride;
    device const uchar* d_row0 = down_base + min(base_col, params.hidden_dim - 1u) * col_stride;
    device const uchar* d_row1 = down_base + min(base_col + 1u, params.hidden_dim - 1u) * col_stride;
    device const uchar* d_row2 = down_base + min(base_col + 2u, params.hidden_dim - 1u) * col_stride;
    device const uchar* d_row3 = down_base + min(base_col + 3u, params.hidden_dim - 1u) * col_stride;
    device const uchar* d_row4 = down_base + min(base_col + 4u, params.hidden_dim - 1u) * col_stride;
    device const uchar* d_row5 = down_base + min(base_col + 5u, params.hidden_dim - 1u) * col_stride;
    device const uchar* d_row6 = down_base + min(base_col + 6u, params.hidden_dim - 1u) * col_stride;
    device const uchar* d_row7 = down_base + min(base_col + 7u, params.hidden_dim - 1u) * col_stride;

    device const half* in_act = intermediate + p * params.intermediate_dim;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

    for (uint c = 0u; c < cluster_blocks; ++c) {
      const uint block_idx = (c << 2u) + b_sub;
      if (block_idx < blocks) {
        const uint blk_byte_off = block_idx * 36u;
        const uint l0 = t_in_b << 2u;
        const uint k_idx = block_idx * 32u + l0;
        const float a0 = float(in_act[k_idx]);
        const float a1 = float(in_act[k_idx + 1u]);
        const float a2 = float(in_act[k_idx + 2u]);
        const float a3 = float(in_act[k_idx + 3u]);
        const float a_sum = a0 + a1 + a2 + a3;

        #define DECODE_D(row_v, acc_v) { \
          device const uchar* blk = row_v + blk_byte_off; \
          const float sc = float(*reinterpret_cast<device const half*>(blk)); \
          const float mn = float(*reinterpret_cast<device const half*>(blk + 2u)); \
          const uchar4 q4 = *reinterpret_cast<device const uchar4*>(blk + byte_offset); \
          acc_v += sc * (a0 * float(q4.x) + a1 * float(q4.y) + a2 * float(q4.z) + a3 * float(q4.w)) + mn * a_sum; \
        }

        DECODE_D(d_row0, acc0);
        DECODE_D(d_row1, acc1);
        DECODE_D(d_row2, acc2);
        DECODE_D(d_row3, acc3);
        DECODE_D(d_row4, acc4);
        DECODE_D(d_row5, acc5);
        DECODE_D(d_row6, acc6);
        DECODE_D(d_row7, acc7);
        #undef DECODE_D
      }
    }

    acc0 = simd_sum(acc0); acc1 = simd_sum(acc1);
    acc2 = simd_sum(acc2); acc3 = simd_sum(acc3);
    acc4 = simd_sum(acc4); acc5 = simd_sum(acc5);
    acc6 = simd_sum(acc6); acc7 = simd_sum(acc7);

    tot0 += w * acc0; tot1 += w * acc1;
    tot2 += w * acc2; tot3 += w * acc3;
    tot4 += w * acc4; tot5 += w * acc5;
    tot6 += w * acc6; tot7 += w * acc7;
  }

  if (lane == 0u && base_col < params.hidden_dim) {
    device half* out_p = output + token * params.hidden_dim + base_col;
    out_p[0] = half(tot0);
    if (base_col + 1u < params.hidden_dim) out_p[1] = half(tot1);
    if (base_col + 2u < params.hidden_dim) out_p[2] = half(tot2);
    if (base_col + 3u < params.hidden_dim) out_p[3] = half(tot3);
    if (base_col + 4u < params.hidden_dim) out_p[4] = half(tot4);
    if (base_col + 5u < params.hidden_dim) out_p[5] = half(tot5);
    if (base_col + 6u < params.hidden_dim) out_p[6] = half(tot6);
    if (base_col + 7u < params.hidden_dim) out_p[7] = half(tot7);
  }
}
|}

let moe_entries =
  [
    kernel_entry_with_threadgroup ~threadgroup:(32, 1, 1)
      ~name:"llmopt_moe_topk_softmax_f16"
      ~operation:Kernel_abi.Operation.Moe_swiglu_ffn
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(32, 1, 1)
      ~name:"llmopt_moe_topk_softmax_f32"
      ~operation:Kernel_abi.Operation.Moe_swiglu_ffn
      ~input_dtype:Ir.Dtype.Float32 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_tile32_affine4_moe_gate_up_swiglu_f16_x8"
      ~operation:Kernel_abi.Operation.Moe_swiglu_ffn
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_tile32_affine8_moe_down_reduce_f16_x8"
      ~operation:Kernel_abi.Operation.Moe_swiglu_ffn
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
  ]
