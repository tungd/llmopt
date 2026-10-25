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
struct MoeTileInfo {
  uint expert;
  uint slot_start;
  uint valid_rows;
  uint _pad;
};

struct GatherF16Params {
  uint rows;
  uint cols_div8;
};

struct MoeTopkParams {
  uint tokens;
  uint num_experts;
  uint top_k;
  uint norm_topk_prob;
};

struct MoeSortParams {
  uint tokens;
  uint num_experts;
  uint top_k;
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

kernel void llmopt_gather_f16(
    device const half* source [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant GatherF16Params& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  const uint col_group = gid % params.cols_div8;
  const uint row = gid / params.cols_div8;
  if (row >= params.rows) return;

  const uint src_row = indices[row];
  const uint col_offset = col_group * 8u;
  device const half4* s_ptr = reinterpret_cast<device const half4*>(source + src_row * (params.cols_div8 * 8u) + col_offset);
  device half4* d_ptr = reinterpret_cast<device half4*>(output + row * (params.cols_div8 * 8u) + col_offset);
  d_ptr[0] = s_ptr[0];
  d_ptr[1] = s_ptr[1];
}

kernel void llmopt_moe_sort_and_bin(
    device const uint* topk_indices [[buffer(0)]],
    device uint* sorted_tokens [[buffer(1)]],
    device uint* sorted_p [[buffer(2)]],
    device uint* p_to_sorted [[buffer(3)]],
    device MoeTileInfo* tile_info [[buffer(4)]],
    device uint* num_tiles [[buffer(5)]],
    constant MoeSortParams& params [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint counts[64];
  threadgroup uint offsets[65];
  threadgroup atomic_uint write_pos[64];
  threadgroup uint expert_tiles[64];
  threadgroup uint tile_offsets[65];

  const uint P = params.tokens * params.top_k;
  const uint E = params.num_experts;

  if (tid < 64) {
    atomic_store_explicit(&counts[tid], 0, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < P; i += 64) {
    uint exp = topk_indices[i];
    if (exp < E && exp < 64) {
      atomic_fetch_add_explicit(&counts[exp], 1, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < 32) {
    uint c0 = atomic_load_explicit(&counts[tid], memory_order_relaxed);
    uint c1 = atomic_load_explicit(&counts[tid + 32], memory_order_relaxed);
    uint s0 = simd_prefix_exclusive_sum(c0);
    uint tot0 = simd_sum(c0);
    uint s1 = simd_prefix_exclusive_sum(c1) + tot0;
    offsets[tid] = s0;
    offsets[tid + 32] = s1;
    if (tid == 31) offsets[64] = s1 + c1;

    uint t0 = (c0 + 15u) / 16u;
    uint t1 = (c1 + 15u) / 16u;
    expert_tiles[tid] = t0;
    expert_tiles[tid + 32] = t1;
    uint ts0 = simd_prefix_exclusive_sum(t0);
    uint t_tot0 = simd_sum(t0);
    uint ts1 = simd_prefix_exclusive_sum(t1) + t_tot0;
    tile_offsets[tid] = ts0;
    tile_offsets[tid + 32] = ts1;
    if (tid == 31) {
      tile_offsets[64] = ts1 + t1;
      num_tiles[0] = ts1 + t1;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < 64) {
    atomic_store_explicit(&write_pos[tid], offsets[tid], memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint p = tid; p < P; p += 64) {
    uint exp = topk_indices[p];
    uint token = p / params.top_k;
    if (exp < E && exp < 64) {
      uint dst = atomic_fetch_add_explicit(&write_pos[exp], 1, memory_order_relaxed);
      sorted_tokens[dst] = token;
      sorted_p[dst] = p;
      p_to_sorted[p] = dst;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid < 64) {
    uint n_tiles_e = expert_tiles[tid];
    uint base_t = tile_offsets[tid];
    uint base_slot = offsets[tid];
    uint count_e = atomic_load_explicit(&counts[tid], memory_order_relaxed);
    for (uint i = 0u; i < n_tiles_e; ++i) {
      uint t_idx = base_t + i;
      uint rem = count_e - i * 16u;
      tile_info[t_idx].expert = tid;
      tile_info[t_idx].slot_start = base_slot + i * 16u;
      tile_info[t_idx].valid_rows = min(16u, rem);
      tile_info[t_idx]._pad = 0;
    }
  }
}

kernel void llmopt_moe_scatter_reduce_f16(
    device const half* down_out [[buffer(0)]],
    device const uint* p_to_sorted [[buffer(1)]],
    device const half* topk_weights [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant MoeDownParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  const uint col_groups = (params.hidden_dim + 7u) >> 3u;
  const uint total_groups = params.tokens * col_groups;
  if (gid >= total_groups) return;

  const uint token = gid / col_groups;
  const uint group_idx = gid - token * col_groups;
  const uint base_col = group_idx << 3u;

  half4 sum0 = half4(0.0h);
  half4 sum1 = half4(0.0h);
  const uint base_p = token * params.top_k;

  for (uint k = 0u; k < params.top_k; ++k) {
    const uint orig_p = base_p + k;
    const half w = topk_weights[orig_p];
    if (w > 0.0h) {
      const uint sorted_idx = p_to_sorted[orig_p];
      if (base_col + 7u < params.hidden_dim) {
        device const half4* p_ptr4 = reinterpret_cast<device const half4*>(
            down_out + sorted_idx * params.hidden_dim + base_col);
        sum0 += p_ptr4[0] * w;
        sum1 += p_ptr4[1] * w;
      } else {
        device const half* p_ptr = down_out + sorted_idx * params.hidden_dim + base_col;
        #pragma unroll
        for (uint i = 0u; i < 8u; ++i) {
          if (base_col + i < params.hidden_dim) {
            if (i < 4u) sum0[i] += p_ptr[i] * w;
            else sum1[i - 4u] += p_ptr[i] * w;
          }
        }
      }
    }
  }

  if (base_col + 7u < params.hidden_dim) {
    device half4* out_p4 = reinterpret_cast<device half4*>(
        output + token * params.hidden_dim + base_col);
    out_p4[0] = sum0;
    out_p4[1] = sum1;
  } else {
    device half* out_p = output + token * params.hidden_dim + base_col;
    #pragma unroll
    for (uint i = 0u; i < 8u; ++i) {
      if (base_col + i < params.hidden_dim) {
        out_p[i] = (i < 4u) ? sum0[i] : sum1[i - 4u];
      }
    }
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
    kernel_entry_with_threadgroup ~threadgroup:(64, 1, 1)
      ~name:"llmopt_moe_sort_and_bin"
      ~operation:Kernel_abi.Operation.Moe_swiglu_ffn
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_gather_f16"
      ~operation:Kernel_abi.Operation.Moe_swiglu_ffn
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_moe_scatter_reduce_f16"
      ~operation:Kernel_abi.Operation.Moe_swiglu_ffn
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
  ]
