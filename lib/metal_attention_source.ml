(** Metal attention shader catalog: attention_source, simd template,
   tree-speculative source, per-head simd helpers, and attention_entries.
   Split out of metal.ml (LOOP item 5). *)

open Metal_source_common

let replace_all = Metal_tensor_source.replace_all

let attention_simd_kernel_name head_dimension =
  Printf.sprintf "llmopt_attention_f16_simd_h%d" head_dimension

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
  if (params.query_length == 0u) return;
  const uint q_tiles_per_head = (params.query_length + 31u) / 32u;
  const uint total_tiles = params.batches * params.heads * q_tiles_per_head;
  if (threadgroup_position.x >= total_tiles) return;

  const uint q_tile = threadgroup_position.x % q_tiles_per_head;
  const uint head = (threadgroup_position.x / q_tiles_per_head) % params.heads;
  const uint batch = threadgroup_position.x / (q_tiles_per_head * params.heads);
  const uint effective_kv_heads = params.kv_heads > 0 ? params.kv_heads : params.heads;
  const uint kv_head = effective_kv_heads < params.heads
      ? head / (params.heads / effective_kv_heads) : head;
  const uint kv_head_base = (batch * effective_kv_heads + kv_head) * params.key_length;
  const uint query_head_base = (batch * params.heads + head) * params.query_length;

  const uint q_tg_base = q_tile * 32u;
  const uint q_base = q_tg_base + simdgroup * 8u;
  const uint my_row = ((lane & 16u) >> 2u) | ((lane & 6u) >> 1u);
  const uint q_idx = q_base + my_row;

  const uint col_base = ((lane & 8u) >> 1u) | ((lane & 1u) << 1u);
  const uint col0 = col_base;
  const uint col1 = col_base + 1u;

  threadgroup half q_tile_mem[4][8][__ATTENTION_HEAD_DIMENSION__];
  threadgroup half k_tile_mem[16][__ATTENTION_HEAD_DIMENSION__];
  threadgroup half v_tile_mem[16][__ATTENTION_HEAD_DIMENSION__];
  threadgroup float o_tile_mem[4][8][__ATTENTION_HEAD_DIMENSION__];

  const uint tid = simdgroup * 32u + lane;
  const uint vecs_per_row = __ATTENTION_HEAD_DIMENSION__ / 4u;
  const uint q_total_vec4 = 8u * __ATTENTION_HEAD_DIMENSION__;
  const uint kv_total_vec4 = 4u * __ATTENTION_HEAD_DIMENSION__;

  // 128 threads cooperatively load Q (32 rows) into threadgroup memory
  threadgroup half4* q_mem4 = reinterpret_cast<threadgroup half4*>(q_tile_mem);
  for (uint i = tid; i < q_total_vec4; i += 128u) {
    const uint r = i / vecs_per_row;
    const uint d4 = i % vecs_per_row;
    const uint curr_q = q_tg_base + r;
    if (curr_q < params.query_length) {
      const uint q_off = (query_head_base + curr_q) * __ATTENTION_HEAD_DIMENSION__ + d4 * 4u;
      q_mem4[i] = *reinterpret_cast<device const half4*>(query + q_off);
    } else {
      q_mem4[i] = half4(0.0h);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Initialize output accumulators in registers
  simdgroup_float8x8 o_acc[__ATTENTION_SIMD_CHUNKS__];
  #pragma unroll
  for (uint c = 0; c < __ATTENTION_SIMD_CHUNKS__; ++c) {
    o_acc[c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  }

  float row_max = -INFINITY;
  float row_denom = 0.0f;

  threadgroup half4* k_mem4 = reinterpret_cast<threadgroup half4*>(k_tile_mem);
  threadgroup half4* v_mem4 = reinterpret_cast<threadgroup half4*>(v_tile_mem);

  const uint max_q = min(params.query_length - 1u, q_tg_base + 31u);
  const uint max_allowed_k = params.past_length + max_q;

  for (uint k_start = 0u; k_start < params.key_length; k_start += 16u) {
    if (params.causal != 0 && k_start > max_allowed_k) break;

    // 128 threads cooperatively load 16 keys and values
    for (uint i = tid; i < kv_total_vec4; i += 128u) {
      const uint r = i / vecs_per_row;
      const uint d4 = i % vecs_per_row;
      const uint curr_k = k_start + r;
      if (curr_k < params.key_length) {
        const uint k_off = (kv_head_base + curr_k) * __ATTENTION_HEAD_DIMENSION__ + d4 * 4u;
        const uint v_off = params.token_first_value != 0
            ? (((batch * params.key_length + curr_k) * effective_kv_heads + kv_head)
                * __ATTENTION_HEAD_DIMENSION__ + d4 * 4u)
            : k_off;
        k_mem4[i] = *reinterpret_cast<device const half4*>(key + k_off);
        v_mem4[i] = *reinterpret_cast<device const half4*>(value + v_off);
      } else {
        k_mem4[i] = half4(0.0h);
        v_mem4[i] = half4(0.0h);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute S0 = Q * K0^T and S1 = Q * K1^T
    simdgroup_float8x8 s_mat0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 s_mat1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    #pragma unroll
    for (uint c = 0; c < __ATTENTION_SIMD_CHUNKS__; ++c) {
      simdgroup_half8x8 q_mat;
      simdgroup_load(q_mat, &q_tile_mem[simdgroup][0][c * 8u], __ATTENTION_HEAD_DIMENSION__);

      simdgroup_half8x8 k_mat0;
      simdgroup_load(k_mat0, &k_tile_mem[0][c * 8u], __ATTENTION_HEAD_DIMENSION__, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(s_mat0, q_mat, k_mat0, s_mat0);

      simdgroup_half8x8 k_mat1;
      simdgroup_load(k_mat1, &k_tile_mem[8][c * 8u], __ATTENTION_HEAD_DIMENSION__, ulong2(0, 0), true);
      simdgroup_multiply_accumulate(s_mat1, q_mat, k_mat1, s_mat1);
    }

    const uint k0_pos0 = k_start + col0;
    const uint k0_pos1 = k_start + col1;
    const bool valid0_0 = (q_idx < params.query_length) && (k0_pos0 < params.key_length)
        && (params.causal == 0 || k0_pos0 <= (params.past_length + q_idx))
        && llmopt_attention_allowed(mask, params, batch, head, q_idx, k0_pos0);
    const bool valid0_1 = (q_idx < params.query_length) && (k0_pos1 < params.key_length)
        && (params.causal == 0 || k0_pos1 <= (params.past_length + q_idx))
        && llmopt_attention_allowed(mask, params, batch, head, q_idx, k0_pos1);

    const uint k1_pos0 = k_start + 8u + col0;
    const uint k1_pos1 = k_start + 8u + col1;
    const bool valid1_0 = (q_idx < params.query_length) && (k1_pos0 < params.key_length)
        && (params.causal == 0 || k1_pos0 <= (params.past_length + q_idx))
        && llmopt_attention_allowed(mask, params, batch, head, q_idx, k1_pos0);
    const bool valid1_1 = (q_idx < params.query_length) && (k1_pos1 < params.key_length)
        && (params.causal == 0 || k1_pos1 <= (params.past_length + q_idx))
        && llmopt_attention_allowed(mask, params, batch, head, q_idx, k1_pos1);

    const float s0_0 = valid0_0 ? (s_mat0.thread_elements()[0] * params.scale) : -INFINITY;
    const float s0_1 = valid0_1 ? (s_mat0.thread_elements()[1] * params.scale) : -INFINITY;
    const float s1_0 = valid1_0 ? (s_mat1.thread_elements()[0] * params.scale) : -INFINITY;
    const float s1_1 = valid1_1 ? (s_mat1.thread_elements()[1] * params.scale) : -INFINITY;

    // Intra-row reduction across the 4 lanes sharing my_row (16 keys total)
    float thread_max = max(max(s0_0, s0_1), max(s1_0, s1_1));
    thread_max = max(thread_max, simd_shuffle_xor(thread_max, 1u));
    thread_max = max(thread_max, simd_shuffle_xor(thread_max, 8u));
    const float block_max = thread_max;

    const bool has_valid = (block_max > -INFINITY);
    const float next_max = has_valid ? max(row_max, block_max) : row_max;
    const float alpha = (row_denom == 0.0f || row_max == -INFINITY || !has_valid)
        ? (has_valid ? 0.0f : 1.0f)
        : exp(row_max - next_max);

    const float p0_0 = (valid0_0 && has_valid) ? exp(s0_0 - next_max) : 0.0f;
    const float p0_1 = (valid0_1 && has_valid) ? exp(s0_1 - next_max) : 0.0f;
    const float p1_0 = (valid1_0 && has_valid) ? exp(s1_0 - next_max) : 0.0f;
    const float p1_1 = (valid1_1 && has_valid) ? exp(s1_1 - next_max) : 0.0f;

    float thread_sum = (p0_0 + p0_1) + (p1_0 + p1_1);
    thread_sum += simd_shuffle_xor(thread_sum, 1u);
    thread_sum += simd_shuffle_xor(thread_sum, 8u);
    const float block_denom = thread_sum;

    row_denom = row_denom * alpha + block_denom;
    row_max = next_max;

    // In-register scaling of accumulator
    #pragma unroll
    for (uint c = 0; c < __ATTENTION_SIMD_CHUNKS__; ++c) {
      o_acc[c].thread_elements()[0] *= alpha;
      o_acc[c].thread_elements()[1] *= alpha;
    }

    // Hardware MMA: Compute P * V and accumulate into o_acc
    simdgroup_half8x8 p_mat0;
    p_mat0.thread_elements()[0] = half(p0_0);
    p_mat0.thread_elements()[1] = half(p0_1);

    simdgroup_half8x8 p_mat1;
    p_mat1.thread_elements()[0] = half(p1_0);
    p_mat1.thread_elements()[1] = half(p1_1);

    #pragma unroll
    for (uint c = 0; c < __ATTENTION_SIMD_CHUNKS__; ++c) {
      simdgroup_half8x8 v_mat0;
      simdgroup_load(v_mat0, &v_tile_mem[0][c * 8u], __ATTENTION_HEAD_DIMENSION__);
      simdgroup_multiply_accumulate(o_acc[c], p_mat0, v_mat0, o_acc[c]);

      simdgroup_half8x8 v_mat1;
      simdgroup_load(v_mat1, &v_tile_mem[8][c * 8u], __ATTENTION_HEAD_DIMENSION__);
      simdgroup_multiply_accumulate(o_acc[c], p_mat1, v_mat1, o_acc[c]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Final softmax normalization
  const float inv_denom = row_denom > 0.0f ? (1.0f / row_denom) : 0.0f;
  #pragma unroll
  for (uint c = 0; c < __ATTENTION_SIMD_CHUNKS__; ++c) {
    o_acc[c].thread_elements()[0] *= inv_denom;
    o_acc[c].thread_elements()[1] *= inv_denom;
    simdgroup_store(o_acc[c], &o_tile_mem[simdgroup][0][c * 8u], __ATTENTION_HEAD_DIMENSION__);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Cooperative writeout of all 32 queries
  for (uint i = tid; i < q_total_vec4; i += 128u) {
    const uint r = i / vecs_per_row;
    const uint d4 = i % vecs_per_row;
    const uint curr_q = q_tg_base + r;
    if (curr_q < params.query_length) {
      const uint sg = r / 8u;
      const uint row_in_sg = r % 8u;
      const uint out_off = params.token_first_output != 0
          ? (((batch * params.query_length + curr_q) * params.heads + head)
              * __ATTENTION_HEAD_DIMENSION__ + d4 * 4u)
          : (((batch * params.heads + head) * params.query_length + curr_q)
              * __ATTENTION_HEAD_DIMENSION__ + d4 * 4u);
      *reinterpret_cast<device half4*>(output + out_off) = half4(
          half(o_tile_mem[sg][row_in_sg][d4 * 4u + 0u]),
          half(o_tile_mem[sg][row_in_sg][d4 * 4u + 1u]),
          half(o_tile_mem[sg][row_in_sg][d4 * 4u + 2u]),
          half(o_tile_mem[sg][row_in_sg][d4 * 4u + 3u])
      );
    }
  }
}
|}

let attention_simd_source head_dimension =
  attention_simd_template
  |> replace_all ~needle:"__ATTENTION_KERNEL__"
       ~replacement:(attention_simd_kernel_name head_dimension)
  |> replace_all ~needle:"__ATTENTION_HEAD_DIMENSION__"
       ~replacement:(string_of_int head_dimension)
  |> replace_all ~needle:"__ATTENTION_SIMD_CHUNKS__"
       ~replacement:(string_of_int (head_dimension / 8))

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
  ^ attention_simd_source 64


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

let attention_simd_entry head_dimension =
  kernel_entry_with_threadgroup ~threadgroup:(128, 1, 1)
    ~name:(attention_simd_kernel_name head_dimension)
    ~operation:Kernel_abi.Operation.Attention ~input_dtype:Ir.Dtype.Float16
    ~output_dtype:Ir.Dtype.Float16

let attention_entries =
  [ kernel_entry_with_threadgroup ~threadgroup:(128, 1, 1)
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
