(** Metal cache shader catalog: cache_source text, its entry builder, and the
   cache/attention-q8 entry list.  Split out of metal.ml (LOOP item 5). *)

open Metal_source_common

let cache_source =
  {|

struct AttentionCacheParams {
  uint items;
  uint segment_offset;
  uint heads;
  uint head_dim;
  uint group_size;
  uint token_elements;
  uint token_groups;
  uint token_stride;
  uint source_items;
  uint source_offset;
};

struct AttentionCachePairParams {
  uint items;
  uint key_segment_offset;
  uint value_segment_offset;
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
      params.segment_offset
      + head * params.head_dim + group_in_head * params.group_size;
  for (uint index = 0; index < params.group_size; ++index) {
    const int quantized = clamp(
        int(rint(float(source[source_base + index]) / scale)), -127, 127);
    values[value_base + index] = char(quantized);
  }
  scales[params.segment_offset / params.group_size + local_group] = stored_scale;
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
      params.segment_offset
      + head * params.head_dim + group_in_head * params.group_size;
  for (uint index = lane; index < params.group_size; index += 32) {
    const int quantized = clamp(
        int(rint(float(source[source_base + index]) / scale)), -127, 127);
    values[value_base + index] = char(quantized);
  }
  if (lane == 0)
    scales[params.segment_offset / params.group_size + local_group] = stored_scale;
}

kernel void llmopt_cache_pack_attention_q8_pair_simd(
    device const half* key_source [[buffer(0)]],
    device const half* value_source [[buffer(1)]],
    device const uint* slots [[buffer(2)]],
    device uchar* pool [[buffer(3)]],
    constant AttentionCachePairParams& params [[buffer(4)]],
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
  float key_maximum = 0.0f;
  float value_maximum = 0.0f;
  for (uint index = lane; index < params.group_size; index += 32) {
    key_maximum = max(
        key_maximum, abs(float(key_source[source_base + index])));
    value_maximum = max(
        value_maximum, abs(float(value_source[source_base + index])));
  }
  key_maximum = simd_max(key_maximum);
  value_maximum = simd_max(value_maximum);
  const half key_stored_scale = half(
      key_maximum == 0.0f ? 1.0f : key_maximum / 127.0f);
  const half value_stored_scale = half(
      value_maximum == 0.0f ? 1.0f : value_maximum / 127.0f);
  const float key_scale = float(key_stored_scale);
  const float value_scale = float(value_stored_scale);
  const uint slot_base = slots[item] * params.token_stride;
  device char* values = reinterpret_cast<device char*>(pool + slot_base);
  device half* scales = reinterpret_cast<device half*>(
      pool + slot_base + params.token_elements);
  const uint local_offset =
      head * params.head_dim + group_in_head * params.group_size;
  const uint key_value_base = params.key_segment_offset + local_offset;
  const uint value_value_base = params.value_segment_offset + local_offset;
  for (uint index = lane; index < params.group_size; index += 32) {
    const int key_quantized = clamp(
        int(rint(float(key_source[source_base + index]) / key_scale)), -127, 127);
    const int value_quantized = clamp(
        int(rint(float(value_source[source_base + index]) / value_scale)),
        -127, 127);
    values[key_value_base + index] = char(key_quantized);
    values[value_value_base + index] = char(value_quantized);
  }
  if (lane == 0) {
    scales[params.key_segment_offset / params.group_size + local_group] =
        key_stored_scale;
    scales[params.value_segment_offset / params.group_size + local_group] =
        value_stored_scale;
  }
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
  const uint local_group = head * groups_per_head
      + within_head / params.group_size;
  const uint slot_base = slots[item] * params.token_stride;
  device const char* values =
      reinterpret_cast<device const char*>(pool + slot_base);
  device const half* scales = reinterpret_cast<device const half*>(
      pool + slot_base + params.token_elements);
  const uint value_index = params.segment_offset + local;
  const uint destination_index =
      (head * params.items + item) * params.head_dim + within_head;
  destination[destination_index] = half(
      float(values[value_index])
      * float(scales[params.segment_offset / params.group_size + local_group]));
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
  const uint local_group = head * groups_per_head
      + within_head / params.group_size;
  const uint slot_base = slots[item] * params.token_stride;
  device const char* values =
      reinterpret_cast<device const char*>(pool + slot_base);
  device const half* scales = reinterpret_cast<device const half*>(
      pool + slot_base + params.token_elements);
  const uint value_index = params.segment_offset + local;
  const uint destination_index =
      (head * params.items + item) * params.head_dim + within_head;
  const char4 quantized =
      *reinterpret_cast<device const char4*>(values + value_index);
  const float scale =
      float(scales[params.segment_offset / params.group_size + local_group]);
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
  uint token_elements;
  uint key_offset;
  uint value_offset;
  uint group_size;
  uint token_stride;
  float scale;
  uint query_length;
  uint mask_dtype;
  uint window;
};

inline bool llmopt_q8_paged_mask_active(
    device const uchar* mask,
    constant Q8PagedAttentionParams& params,
    uint index) {
  if (mask == nullptr || params.mask_batches == 0 || params.mask_dtype == 0)
    return true;
  if (params.mask_dtype == 2)
    return float(reinterpret_cast<device const half*>(mask)[index]) >= 0.0f;
  return mask[index] != 0;
}

inline float llmopt_q8_paged_value(
    device const uchar* pool,
    device const uint* slots,
    constant Q8PagedAttentionParams& params,
    uint key_position,
    uint segment_offset,
    uint kv_head,
    uint dimension) {
  const uint groups_per_head = params.head_dim / params.group_size;
  const uint slot_base = slots[key_position] * params.token_stride;
  device const char* values =
      reinterpret_cast<device const char*>(pool + slot_base);
  device const half* scales = reinterpret_cast<device const half*>(
      pool + slot_base + params.token_elements);
  const uint value_index = segment_offset
      + kv_head * params.head_dim + dimension;
  const uint scale_index = segment_offset / params.group_size
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
  const uint first_key = params.window == 0 || key_length <= params.window
      ? 0 : key_length - params.window;
  const uint key_segment = params.key_offset;
  const uint value_segment = params.value_offset;
  float maximum = -INFINITY;
  float denominator = 0.0f;
  float result_low = 0.0f;
  float result_high = 0.0f;
  for (uint key_position = first_key; key_position < key_length; ++key_position) {
    if (mask != nullptr && params.mask_batches > 0) {
      const uint mask_batch = params.mask_batches == 1 ? 0 : batch;
      const uint mask_head = params.mask_heads == 1 ? 0 : head;
      const uint mask_key_stride = params.past_length + q_len;
      const uint mask_index =
          ((mask_batch * params.mask_heads + mask_head) * q_len + q_pos)
              * mask_key_stride + key_position;
      if (!llmopt_q8_paged_mask_active(mask, params, mask_index)) continue;
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
      ((batch * q_len + q_pos) * params.query_heads + head) * params.head_dim;
  if (lane < params.head_dim)
    output[output_base + lane] = half(
        denominator == 0.0f ? 0.0f : result_low / denominator);
  const uint high_dimension = lane + Q8_PAGED_ATTENTION_SIMD_WIDTH;
  if (high_dimension < params.head_dim)
    output[output_base + high_dimension] = half(
        denominator == 0.0f ? 0.0f : result_high / denominator);
}

#define Q8_PAGED_ATTENTION_MAX_FRAGMENTS 16

kernel void llmopt_attention_q8_paged_simd_generic(
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
  const uint first_key = params.window == 0 || key_length <= params.window
      ? 0 : key_length - params.window;
  const uint fragments =
      (params.head_dim + Q8_PAGED_ATTENTION_SIMD_WIDTH - 1u)
      / Q8_PAGED_ATTENTION_SIMD_WIDTH;
  float maximum = -INFINITY;
  float denominator = 0.0f;
  float results[Q8_PAGED_ATTENTION_MAX_FRAGMENTS];
  for (uint component = 0; component < Q8_PAGED_ATTENTION_MAX_FRAGMENTS;
       ++component)
    results[component] = 0.0f;
  for (uint key_position = first_key; key_position < key_length; ++key_position) {
    if (mask != nullptr && params.mask_batches > 0) {
      const uint mask_batch = params.mask_batches == 1 ? 0 : batch;
      const uint mask_head = params.mask_heads == 1 ? 0 : head;
      const uint mask_key_stride = params.past_length + q_len;
      const uint mask_index =
          ((mask_batch * params.mask_heads + mask_head) * q_len + q_pos)
              * mask_key_stride + key_position;
      if (!llmopt_q8_paged_mask_active(mask, params, mask_index)) continue;
    }
    float partial_score = 0.0f;
    for (uint dimension = lane; dimension < params.head_dim;
         dimension += Q8_PAGED_ATTENTION_SIMD_WIDTH) {
      const float key_value = key_position < params.past_length
          ? llmopt_q8_paged_value(pool, slots, params, key_position,
                params.key_offset, kv_head, dimension)
          : float(current_key[current_base
                + (key_position - params.past_length) * params.head_dim
                + dimension]);
      partial_score += float(query[query_base + dimension]) * key_value;
    }
    const float score = simd_sum(partial_score) * params.scale;
    const float next_maximum = max(maximum, score);
    const float previous_scale = denominator == 0.0f ? 0.0f
        : exp(maximum - next_maximum);
    const float current_scale = exp(score - next_maximum);
    for (uint component = 0; component < fragments; ++component) {
      const uint dimension =
          lane + component * Q8_PAGED_ATTENTION_SIMD_WIDTH;
      if (dimension < params.head_dim) {
        const float value = key_position < params.past_length
            ? llmopt_q8_paged_value(pool, slots, params, key_position,
                  params.value_offset, kv_head, dimension)
            : float(current_value[current_base
                  + (key_position - params.past_length) * params.head_dim
                  + dimension]);
        results[component] = results[component] * previous_scale
            + current_scale * value;
      }
    }
    denominator = denominator * previous_scale + current_scale;
    maximum = next_maximum;
  }
  const uint output_base =
      ((batch * q_len + q_pos) * params.query_heads + head) * params.head_dim;
  for (uint component = 0; component < fragments; ++component) {
    const uint dimension =
        lane + component * Q8_PAGED_ATTENTION_SIMD_WIDTH;
    if (dimension < params.head_dim)
      output[output_base + dimension] = half(
          denominator == 0.0f ? 0.0f : results[component] / denominator);
  }
}

kernel void llmopt_attention_q8_paged_simd_h256(
    device const half* query [[buffer(0)]],
    device const half* current_key [[buffer(1)]],
    device const half* current_value [[buffer(2)]],
    device const uchar* pool [[buffer(3)]],
    device const uint* slots [[buffer(4)]],
    device const uchar* mask [[buffer(5)]],
    device half* output [[buffer(6)]],
    constant Q8PagedAttentionParams& params [[buffer(7)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = threadgroup_position.x;
  const uint q_len = params.query_length == 0 ? 1 : params.query_length;
  const uint rows = params.batches * params.query_heads * q_len;
  if (row >= rows) return;
  const uint q_pos = row % q_len;
  const uint head = (row / q_len) % params.query_heads;
  const uint batch = row / (q_len * params.query_heads);
  const uint heads_per_kv = params.query_heads / params.kv_heads;
  const uint kv_head = head / heads_per_kv;
  const uint query_base =
      ((batch * params.query_heads + head) * q_len + q_pos) * 256u;
  const uint current_base =
      (batch * params.kv_heads + kv_head) * q_len * 256u;
  const uint key_length = params.past_length + q_pos + 1u;
  const uint first_key = params.window == 0u || key_length <= params.window
      ? 0u : key_length - params.window;

  float q[8];
  #pragma unroll
  for (uint c = 0; c < 8u; ++c) {
    q[c] = float(query[query_base + lane + c * 32u]);
  }

  float maximum = -INFINITY;
  float denominator = 0.0f;
  float results[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (uint key_position = first_key; key_position < key_length; ++key_position) {
    if (mask != nullptr && params.mask_batches > 0) {
      const uint mask_batch = params.mask_batches == 1 ? 0 : batch;
      const uint mask_head = params.mask_heads == 1 ? 0 : head;
      const uint mask_key_stride = params.past_length + q_len;
      const uint mask_index =
          ((mask_batch * params.mask_heads + mask_head) * q_len + q_pos)
              * mask_key_stride + key_position;
      if (!llmopt_q8_paged_mask_active(mask, params, mask_index)) continue;
    }

    float partial_score = 0.0f;
    if (key_position < params.past_length) {
      const uint slot_base = slots[key_position] * params.token_stride;
      device const char* k_values =
          reinterpret_cast<device const char*>(pool + slot_base)
          + params.key_offset + kv_head * 256u;
      device const half* k_scales =
          reinterpret_cast<device const half*>(pool + slot_base + params.token_elements)
          + (params.key_offset / 64u + kv_head * 4u);

      const float k0 = float(k_values[lane + 0u]);
      const float k1 = float(k_values[lane + 32u]);
      const float sc0 = float(k_scales[0]);

      const float k2 = float(k_values[lane + 64u]);
      const float k3 = float(k_values[lane + 96u]);
      const float sc1 = float(k_scales[1]);

      const float k4 = float(k_values[lane + 128u]);
      const float k5 = float(k_values[lane + 160u]);
      const float sc2 = float(k_scales[2]);

      const float k6 = float(k_values[lane + 192u]);
      const float k7 = float(k_values[lane + 224u]);
      const float sc3 = float(k_scales[3]);

      partial_score = (k0 * q[0] + k1 * q[1]) * sc0
                    + (k2 * q[2] + k3 * q[3]) * sc1
                    + (k4 * q[4] + k5 * q[5]) * sc2
                    + (k6 * q[6] + k7 * q[7]) * sc3;
    } else {
      const uint offset = current_base + (key_position - params.past_length) * 256u;
      #pragma unroll
      for (uint c = 0; c < 8u; ++c) {
        partial_score += float(current_key[offset + lane + c * 32u]) * q[c];
      }
    }

    const float score = simd_sum(partial_score) * params.scale;
    const float next_maximum = max(maximum, score);
    const float previous_scale = denominator == 0.0f ? 0.0f : exp(maximum - next_maximum);
    const float current_scale = exp(score - next_maximum);

    if (key_position < params.past_length) {
      const uint slot_base = slots[key_position] * params.token_stride;
      device const char* v_values =
          reinterpret_cast<device const char*>(pool + slot_base)
          + params.value_offset + kv_head * 256u;
      device const half* v_scales =
          reinterpret_cast<device const half*>(pool + slot_base + params.token_elements)
          + (params.value_offset / 64u + kv_head * 4u);

      const float cs0 = current_scale * float(v_scales[0]);
      const float cs1 = current_scale * float(v_scales[1]);
      const float cs2 = current_scale * float(v_scales[2]);
      const float cs3 = current_scale * float(v_scales[3]);

      results[0] = results[0] * previous_scale + cs0 * float(v_values[lane + 0u]);
      results[1] = results[1] * previous_scale + cs0 * float(v_values[lane + 32u]);
      results[2] = results[2] * previous_scale + cs1 * float(v_values[lane + 64u]);
      results[3] = results[3] * previous_scale + cs1 * float(v_values[lane + 96u]);
      results[4] = results[4] * previous_scale + cs2 * float(v_values[lane + 128u]);
      results[5] = results[5] * previous_scale + cs2 * float(v_values[lane + 160u]);
      results[6] = results[6] * previous_scale + cs3 * float(v_values[lane + 192u]);
      results[7] = results[7] * previous_scale + cs3 * float(v_values[lane + 224u]);
    } else {
      const uint offset = current_base + (key_position - params.past_length) * 256u;
      #pragma unroll
      for (uint c = 0; c < 8u; ++c) {
        results[c] = results[c] * previous_scale + current_scale * float(current_value[offset + lane + c * 32u]);
      }
    }

    denominator = denominator * previous_scale + current_scale;
    maximum = next_maximum;
  }

  const uint output_base =
      ((batch * q_len + q_pos) * params.query_heads + head) * 256u;
  const float inv_den = denominator == 0.0f ? 0.0f : (1.0f / denominator);
  #pragma unroll
  for (uint c = 0; c < 8u; ++c) {
    output[output_base + lane + c * 32u] = half(results[c] * inv_den);
  }
}

kernel void llmopt_attention_q8_paged_simd_h512(
    device const half* query [[buffer(0)]],
    device const half* current_key [[buffer(1)]],
    device const half* current_value [[buffer(2)]],
    device const uchar* pool [[buffer(3)]],
    device const uint* slots [[buffer(4)]],
    device const uchar* mask [[buffer(5)]],
    device half* output [[buffer(6)]],
    constant Q8PagedAttentionParams& params [[buffer(7)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = threadgroup_position.x;
  const uint q_len = params.query_length == 0 ? 1 : params.query_length;
  const uint rows = params.batches * params.query_heads * q_len;
  if (row >= rows) return;
  const uint q_pos = row % q_len;
  const uint head = (row / q_len) % params.query_heads;
  const uint batch = row / (q_len * params.query_heads);
  const uint heads_per_kv = params.query_heads / params.kv_heads;
  const uint kv_head = head / heads_per_kv;
  const uint query_base =
      ((batch * params.query_heads + head) * q_len + q_pos) * 512u;
  const uint current_base =
      (batch * params.kv_heads + kv_head) * q_len * 512u;
  const uint key_length = params.past_length + q_pos + 1u;
  const uint first_key = params.window == 0u || key_length <= params.window
      ? 0u : key_length - params.window;

  float q[16];
  #pragma unroll
  for (uint c = 0; c < 16u; ++c) {
    q[c] = float(query[query_base + lane + c * 32u]);
  }

  float maximum = -INFINITY;
  float denominator = 0.0f;
  float results[16] = {0.0f};

  for (uint key_position = first_key; key_position < key_length; ++key_position) {
    if (mask != nullptr && params.mask_batches > 0) {
      const uint mask_batch = params.mask_batches == 1 ? 0 : batch;
      const uint mask_head = params.mask_heads == 1 ? 0 : head;
      const uint mask_key_stride = params.past_length + q_len;
      const uint mask_index =
          ((mask_batch * params.mask_heads + mask_head) * q_len + q_pos)
              * mask_key_stride + key_position;
      if (!llmopt_q8_paged_mask_active(mask, params, mask_index)) continue;
    }

    float partial_score = 0.0f;
    if (key_position < params.past_length) {
      const uint slot_base = slots[key_position] * params.token_stride;
      device const char* k_values =
          reinterpret_cast<device const char*>(pool + slot_base)
          + params.key_offset + kv_head * 512u;
      device const half* k_scales =
          reinterpret_cast<device const half*>(pool + slot_base + params.token_elements)
          + (params.key_offset / 64u + kv_head * 8u);

      #pragma unroll
      for (uint g = 0; g < 8u; ++g) {
        const float k0 = float(k_values[lane + (g * 2u + 0u) * 32u]);
        const float k1 = float(k_values[lane + (g * 2u + 1u) * 32u]);
        const float sc = float(k_scales[g]);
        partial_score += (k0 * q[g * 2u + 0u] + k1 * q[g * 2u + 1u]) * sc;
      }
    } else {
      const uint offset = current_base + (key_position - params.past_length) * 512u;
      #pragma unroll
      for (uint c = 0; c < 16u; ++c) {
        partial_score += float(current_key[offset + lane + c * 32u]) * q[c];
      }
    }

    const float score = simd_sum(partial_score) * params.scale;
    const float next_maximum = max(maximum, score);
    const float previous_scale = denominator == 0.0f ? 0.0f : exp(maximum - next_maximum);
    const float current_scale = exp(score - next_maximum);

    if (key_position < params.past_length) {
      const uint slot_base = slots[key_position] * params.token_stride;
      device const char* v_values =
          reinterpret_cast<device const char*>(pool + slot_base)
          + params.value_offset + kv_head * 512u;
      device const half* v_scales =
          reinterpret_cast<device const half*>(pool + slot_base + params.token_elements)
          + (params.value_offset / 64u + kv_head * 8u);

      #pragma unroll
      for (uint g = 0; g < 8u; ++g) {
        const float cs = current_scale * float(v_scales[g]);
        results[g * 2u + 0u] = results[g * 2u + 0u] * previous_scale + cs * float(v_values[lane + (g * 2u + 0u) * 32u]);
        results[g * 2u + 1u] = results[g * 2u + 1u] * previous_scale + cs * float(v_values[lane + (g * 2u + 1u) * 32u]);
      }
    } else {
      const uint offset = current_base + (key_position - params.past_length) * 512u;
      #pragma unroll
      for (uint c = 0; c < 16u; ++c) {
        results[c] = results[c] * previous_scale + current_scale * float(current_value[offset + lane + c * 32u]);
      }
    }

    denominator = denominator * previous_scale + current_scale;
    maximum = next_maximum;
  }

  const uint output_base =
      ((batch * q_len + q_pos) * params.query_heads + head) * 512u;
  const float inv_den = denominator == 0.0f ? 0.0f : (1.0f / denominator);
  #pragma unroll
  for (uint c = 0; c < 16u; ++c) {
    output[output_base + lane + c * 32u] = half(results[c] * inv_den);
  }
}

kernel void llmopt_attention_q8_paged_simd_h64_gqa3(
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
  const uint rows = params.batches * params.kv_heads * q_len;
  if (row >= rows) return;
  const uint q_pos = row % q_len;
  const uint kv_head = (row / q_len) % params.kv_heads;
  const uint batch = row / (q_len * params.kv_heads);
  const uint head0 = kv_head * 3;
  const uint head1 = head0 + 1;
  const uint head2 = head0 + 2;
  const uint query_base0 =
      ((batch * params.query_heads + head0) * q_len + q_pos) * params.head_dim;
  const uint query_base1 =
      ((batch * params.query_heads + head1) * q_len + q_pos) * params.head_dim;
  const uint query_base2 =
      ((batch * params.query_heads + head2) * q_len + q_pos) * params.head_dim;
  const uint current_base =
      (batch * params.kv_heads + kv_head) * q_len * params.head_dim;
  const uint key_length = params.past_length + q_pos + 1;
  const uint first_key = params.window == 0 || key_length <= params.window
      ? 0 : key_length - params.window;
  float maximum0 = -INFINITY;
  float maximum1 = -INFINITY;
  float maximum2 = -INFINITY;
  float denominator0 = 0.0f;
  float denominator1 = 0.0f;
  float denominator2 = 0.0f;
  float result_low0 = 0.0f;
  float result_low1 = 0.0f;
  float result_low2 = 0.0f;
  float result_high0 = 0.0f;
  float result_high1 = 0.0f;
  float result_high2 = 0.0f;
  for (uint key_position = first_key; key_position < key_length; ++key_position) {
    bool active0 = true;
    bool active1 = true;
    bool active2 = true;
    if (mask != nullptr && params.mask_batches > 0) {
      const uint mask_batch = params.mask_batches == 1 ? 0 : batch;
      const uint mask_key_stride = params.past_length + q_len;
      const uint mask_base =
          (mask_batch * params.mask_heads * q_len + q_pos)
              * mask_key_stride + key_position;
      active0 = llmopt_q8_paged_mask_active(mask, params, mask_base
          + (params.mask_heads == 1 ? 0 : head0 * q_len * mask_key_stride));
      active1 = llmopt_q8_paged_mask_active(mask, params, mask_base
          + (params.mask_heads == 1 ? 0 : head1 * q_len * mask_key_stride));
      active2 = llmopt_q8_paged_mask_active(mask, params, mask_base
          + (params.mask_heads == 1 ? 0 : head2 * q_len * mask_key_stride));
      if (!active0 && !active1 && !active2) continue;
    }
    float partial_score0 = 0.0f;
    float partial_score1 = 0.0f;
    float partial_score2 = 0.0f;
    for (uint dimension = lane; dimension < params.head_dim;
         dimension += Q8_PAGED_ATTENTION_SIMD_WIDTH) {
      const float key_value = key_position < params.past_length
          ? llmopt_q8_paged_value(pool, slots, params, key_position,
                params.key_offset, kv_head, dimension)
          : float(current_key[current_base
                + (key_position - params.past_length) * params.head_dim
                + dimension]);
      partial_score0 += float(query[query_base0 + dimension]) * key_value;
      partial_score1 += float(query[query_base1 + dimension]) * key_value;
      partial_score2 += float(query[query_base2 + dimension]) * key_value;
    }
    const float score0 = simd_sum(partial_score0) * params.scale;
    const float score1 = simd_sum(partial_score1) * params.scale;
    const float score2 = simd_sum(partial_score2) * params.scale;
    const uint high_dimension = lane + Q8_PAGED_ATTENTION_SIMD_WIDTH;
    const float value_low = key_position < params.past_length
        ? llmopt_q8_paged_value(pool, slots, params, key_position,
              params.value_offset, kv_head, lane)
        : float(current_value[current_base
              + (key_position - params.past_length) * params.head_dim + lane]);
    const float value_high = key_position < params.past_length
        ? llmopt_q8_paged_value(pool, slots, params, key_position,
              params.value_offset, kv_head, high_dimension)
        : float(current_value[current_base
              + (key_position - params.past_length) * params.head_dim
              + high_dimension]);
#define LLMOPT_GQA3_UPDATE(N) \
    if (active##N) { \
      const float next_maximum = max(maximum##N, score##N); \
      const float previous_scale = denominator##N == 0.0f ? 0.0f \
          : exp(maximum##N - next_maximum); \
      const float current_scale = exp(score##N - next_maximum); \
      result_low##N = result_low##N * previous_scale \
          + current_scale * value_low; \
      result_high##N = result_high##N * previous_scale \
          + current_scale * value_high; \
      denominator##N = denominator##N * previous_scale + current_scale; \
      maximum##N = next_maximum; \
    }
    LLMOPT_GQA3_UPDATE(0)
    LLMOPT_GQA3_UPDATE(1)
    LLMOPT_GQA3_UPDATE(2)
#undef LLMOPT_GQA3_UPDATE
  }
  const uint output_base =
      ((batch * q_len + q_pos) * params.query_heads + head0) * params.head_dim;
#define LLMOPT_GQA3_WRITE(N) \
  output[output_base + N * params.head_dim + lane] = half( \
      denominator##N == 0.0f ? 0.0f : result_low##N / denominator##N); \
  output[output_base + N * params.head_dim + lane \
      + Q8_PAGED_ATTENTION_SIMD_WIDTH] = half( \
      denominator##N == 0.0f ? 0.0f : result_high##N / denominator##N);
  LLMOPT_GQA3_WRITE(0)
  LLMOPT_GQA3_WRITE(1)
  LLMOPT_GQA3_WRITE(2)
#undef LLMOPT_GQA3_WRITE
}

kernel void llmopt_attention_q8_paged_simd_h64_split8(
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
  const uint row = threadgroup_position.x;
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
  const uint first_key = params.window == 0 || key_length <= params.window
      ? 0 : key_length - params.window;
  float maximum = -INFINITY;
  float denominator = 0.0f;
  float result_low = 0.0f;
  float result_high = 0.0f;
  const uint chunk_length =
      (key_length + Q8_PAGED_ATTENTION_ROWS_PER_THREADGROUP - 1u)
      / Q8_PAGED_ATTENTION_ROWS_PER_THREADGROUP;
  const uint chunk_start = max(simdgroup * chunk_length, first_key);
  const uint chunk_stop = min(chunk_start + chunk_length, key_length);
  for (uint key_position = chunk_start; key_position < chunk_stop;
       ++key_position) {
    if (mask != nullptr && params.mask_batches > 0) {
      const uint mask_batch = params.mask_batches == 1 ? 0 : batch;
      const uint mask_head = params.mask_heads == 1 ? 0 : head;
      const uint mask_key_stride = params.past_length + q_len;
      const uint mask_index =
          ((mask_batch * params.mask_heads + mask_head) * q_len + q_pos)
              * mask_key_stride + key_position;
      if (!llmopt_q8_paged_mask_active(mask, params, mask_index)) continue;
    }
    float partial_score = 0.0f;
    for (uint dimension = lane; dimension < params.head_dim;
         dimension += Q8_PAGED_ATTENTION_SIMD_WIDTH) {
      const float key_value = key_position < params.past_length
          ? llmopt_q8_paged_value(pool, slots, params, key_position,
                params.key_offset, kv_head, dimension)
          : float(current_key[current_base
                + (key_position - params.past_length) * params.head_dim
                + dimension]);
      partial_score += float(query[query_base + dimension]) * key_value;
    }
    const float score = simd_sum(partial_score) * params.scale;
    const float next_maximum = max(maximum, score);
    const float previous_scale = denominator == 0.0f ? 0.0f
        : exp(maximum - next_maximum);
    const float current_scale = exp(score - next_maximum);
    const float value_low = key_position < params.past_length
        ? llmopt_q8_paged_value(pool, slots, params, key_position,
              params.value_offset, kv_head, lane)
        : float(current_value[current_base
              + (key_position - params.past_length) * params.head_dim + lane]);
    const uint high_dimension = lane + Q8_PAGED_ATTENTION_SIMD_WIDTH;
    const float value_high = key_position < params.past_length
        ? llmopt_q8_paged_value(pool, slots, params, key_position,
              params.value_offset, kv_head, high_dimension)
        : float(current_value[current_base
              + (key_position - params.past_length) * params.head_dim
              + high_dimension]);
    result_low = result_low * previous_scale + current_scale * value_low;
    result_high = result_high * previous_scale + current_scale * value_high;
    denominator = denominator * previous_scale + current_scale;
    maximum = next_maximum;
  }
  threadgroup float partial_maximum[8];
  threadgroup float partial_denominator[8];
  threadgroup float partial_values[8][64];
  if (lane == 0) {
    partial_maximum[simdgroup] = maximum;
    partial_denominator[simdgroup] = denominator;
  }
  partial_values[simdgroup][lane] = result_low;
  partial_values[simdgroup][lane + Q8_PAGED_ATTENTION_SIMD_WIDTH] = result_high;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup == 0) {
    float merged_maximum = -INFINITY;
    float merged_denominator = 0.0f;
    float merged_low = 0.0f;
    float merged_high = 0.0f;
    for (uint split = 0; split < 8; ++split) {
      if (partial_denominator[split] > 0.0f) {
        const float next_maximum =
            max(merged_maximum, partial_maximum[split]);
        const float previous_scale = merged_denominator == 0.0f ? 0.0f
            : exp(merged_maximum - next_maximum);
        const float current_scale =
            exp(partial_maximum[split] - next_maximum);
        merged_low = merged_low * previous_scale
            + partial_values[split][lane] * current_scale;
        merged_high = merged_high * previous_scale
            + partial_values[split]
                [lane + Q8_PAGED_ATTENTION_SIMD_WIDTH] * current_scale;
        merged_denominator = merged_denominator * previous_scale
            + partial_denominator[split] * current_scale;
        merged_maximum = next_maximum;
      }
    }
    const uint output_base =
        ((batch * q_len + q_pos) * params.query_heads + head) * params.head_dim;
    output[output_base + lane] = half(
        merged_denominator == 0.0f ? 0.0f : merged_low / merged_denominator);
    output[output_base + lane + Q8_PAGED_ATTENTION_SIMD_WIDTH] = half(
        merged_denominator == 0.0f ? 0.0f : merged_high / merged_denominator);
  }
}

kernel void llmopt_attention_q8_paged_simd_h64_gqa3_split8(
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
  const uint row = threadgroup_position.x;
  const uint q_len = params.query_length == 0 ? 1 : params.query_length;
  const uint rows = params.batches * params.kv_heads * q_len;
  if (row >= rows) return;
  const uint q_pos = row % q_len;
  const uint kv_head = (row / q_len) % params.kv_heads;
  const uint batch = row / (q_len * params.kv_heads);
  const uint head0 = kv_head * 3;
  const uint query_base0 =
      ((batch * params.query_heads + head0) * q_len + q_pos) * params.head_dim;
  const uint query_base1 = query_base0 + q_len * params.head_dim;
  const uint query_base2 = query_base1 + q_len * params.head_dim;
  const uint current_base =
      (batch * params.kv_heads + kv_head) * q_len * params.head_dim;
  const uint key_length = params.past_length + q_pos + 1;
  const uint first_key = params.window == 0 || key_length <= params.window
      ? 0 : key_length - params.window;
  float maximum0 = -INFINITY;
  float maximum1 = -INFINITY;
  float maximum2 = -INFINITY;
  float denominator0 = 0.0f;
  float denominator1 = 0.0f;
  float denominator2 = 0.0f;
  float result_low0 = 0.0f;
  float result_low1 = 0.0f;
  float result_low2 = 0.0f;
  float result_high0 = 0.0f;
  float result_high1 = 0.0f;
  float result_high2 = 0.0f;
  const uint chunk_length =
      (key_length + Q8_PAGED_ATTENTION_ROWS_PER_THREADGROUP - 1u)
      / Q8_PAGED_ATTENTION_ROWS_PER_THREADGROUP;
  const uint chunk_start = max(simdgroup * chunk_length, first_key);
  const uint chunk_stop = min(chunk_start + chunk_length, key_length);
  for (uint key_position = chunk_start; key_position < chunk_stop;
       ++key_position) {
    bool active0 = true;
    bool active1 = true;
    bool active2 = true;
    if (mask != nullptr && params.mask_batches > 0) {
      const uint mask_batch = params.mask_batches == 1 ? 0 : batch;
      const uint mask_key_stride = params.past_length + q_len;
      const uint mask_base =
          (mask_batch * params.mask_heads * q_len + q_pos)
              * mask_key_stride + key_position;
      active0 = llmopt_q8_paged_mask_active(mask, params, mask_base
          + (params.mask_heads == 1 ? 0 : head0 * q_len * mask_key_stride));
      active1 = llmopt_q8_paged_mask_active(mask, params, mask_base
          + (params.mask_heads == 1 ? 0 : (head0 + 1) * q_len * mask_key_stride));
      active2 = llmopt_q8_paged_mask_active(mask, params, mask_base
          + (params.mask_heads == 1 ? 0 : (head0 + 2) * q_len * mask_key_stride));
      if (!active0 && !active1 && !active2) continue;
    }
    float partial_score0 = 0.0f;
    float partial_score1 = 0.0f;
    float partial_score2 = 0.0f;
    for (uint dimension = lane; dimension < params.head_dim;
         dimension += Q8_PAGED_ATTENTION_SIMD_WIDTH) {
      const float key_value = key_position < params.past_length
          ? llmopt_q8_paged_value(pool, slots, params, key_position,
                params.key_offset, kv_head, dimension)
          : float(current_key[current_base
                + (key_position - params.past_length) * params.head_dim
                + dimension]);
      partial_score0 += float(query[query_base0 + dimension]) * key_value;
      partial_score1 += float(query[query_base1 + dimension]) * key_value;
      partial_score2 += float(query[query_base2 + dimension]) * key_value;
    }
    const float score0 = simd_sum(partial_score0) * params.scale;
    const float score1 = simd_sum(partial_score1) * params.scale;
    const float score2 = simd_sum(partial_score2) * params.scale;
    const float value_low = key_position < params.past_length
        ? llmopt_q8_paged_value(pool, slots, params, key_position,
              params.value_offset, kv_head, lane)
        : float(current_value[current_base
              + (key_position - params.past_length) * params.head_dim + lane]);
    const uint high_dimension = lane + Q8_PAGED_ATTENTION_SIMD_WIDTH;
    const float value_high = key_position < params.past_length
        ? llmopt_q8_paged_value(pool, slots, params, key_position,
              params.value_offset, kv_head, high_dimension)
        : float(current_value[current_base
              + (key_position - params.past_length) * params.head_dim
              + high_dimension]);
#define LLMOPT_GQA3_SPLIT_UPDATE(N) \
    if (active##N) { \
      const float next_maximum = max(maximum##N, score##N); \
      const float previous_scale = denominator##N == 0.0f ? 0.0f \
          : exp(maximum##N - next_maximum); \
      const float current_scale = exp(score##N - next_maximum); \
      result_low##N = result_low##N * previous_scale \
          + current_scale * value_low; \
      result_high##N = result_high##N * previous_scale \
          + current_scale * value_high; \
      denominator##N = denominator##N * previous_scale + current_scale; \
      maximum##N = next_maximum; \
    }
    LLMOPT_GQA3_SPLIT_UPDATE(0)
    LLMOPT_GQA3_SPLIT_UPDATE(1)
    LLMOPT_GQA3_SPLIT_UPDATE(2)
#undef LLMOPT_GQA3_SPLIT_UPDATE
  }
  threadgroup float partial_maximum[3][8];
  threadgroup float partial_denominator[3][8];
  threadgroup float partial_values[3][8][64];
#define LLMOPT_GQA3_SPLIT_STORE(N) \
  if (lane == 0) { \
    partial_maximum[N][simdgroup] = maximum##N; \
    partial_denominator[N][simdgroup] = denominator##N; \
  } \
  partial_values[N][simdgroup][lane] = result_low##N; \
  partial_values[N][simdgroup] \
      [lane + Q8_PAGED_ATTENTION_SIMD_WIDTH] = result_high##N;
  LLMOPT_GQA3_SPLIT_STORE(0)
  LLMOPT_GQA3_SPLIT_STORE(1)
  LLMOPT_GQA3_SPLIT_STORE(2)
#undef LLMOPT_GQA3_SPLIT_STORE
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simdgroup == 0) {
    for (uint grouped_head = 0; grouped_head < 3; ++grouped_head) {
      float merged_maximum = -INFINITY;
      float merged_denominator = 0.0f;
      float merged_low = 0.0f;
      float merged_high = 0.0f;
      for (uint split = 0; split < 8; ++split) {
        if (partial_denominator[grouped_head][split] > 0.0f) {
          const float next_maximum =
              max(merged_maximum, partial_maximum[grouped_head][split]);
          const float previous_scale = merged_denominator == 0.0f ? 0.0f
              : exp(merged_maximum - next_maximum);
          const float current_scale =
              exp(partial_maximum[grouped_head][split] - next_maximum);
          merged_low = merged_low * previous_scale
              + partial_values[grouped_head][split][lane] * current_scale;
          merged_high = merged_high * previous_scale
              + partial_values[grouped_head][split]
                  [lane + Q8_PAGED_ATTENTION_SIMD_WIDTH] * current_scale;
          merged_denominator = merged_denominator * previous_scale
              + partial_denominator[grouped_head][split] * current_scale;
          merged_maximum = next_maximum;
        }
      }
      const uint output_base =
          ((batch * q_len + q_pos) * params.query_heads
              + head0 + grouped_head) * params.head_dim;
      output[output_base + lane] = half(merged_denominator == 0.0f
          ? 0.0f : merged_low / merged_denominator);
      output[output_base + lane + Q8_PAGED_ATTENTION_SIMD_WIDTH] = half(
          merged_denominator == 0.0f ? 0.0f
              : merged_high / merged_denominator);
    }
  }
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
    cache_entry "llmopt_cache_pack_attention_q8_pair_simd" Ir.Dtype.Float16
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
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_attention_q8_paged_simd_generic"
      ~operation:Kernel_abi.Operation.Attention
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(32, 1, 1)
      ~name:"llmopt_attention_q8_paged_simd_h256"
      ~operation:Kernel_abi.Operation.Attention
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(32, 1, 1)
      ~name:"llmopt_attention_q8_paged_simd_h512"
      ~operation:Kernel_abi.Operation.Attention
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_attention_q8_paged_simd_h64_gqa3"
      ~operation:Kernel_abi.Operation.Attention
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_attention_q8_paged_simd_h64_split8"
      ~operation:Kernel_abi.Operation.Attention
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16;
    kernel_entry_with_threadgroup ~threadgroup:(256, 1, 1)
      ~name:"llmopt_attention_q8_paged_simd_h64_gqa3_split8"
      ~operation:Kernel_abi.Operation.Attention
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16 ]
