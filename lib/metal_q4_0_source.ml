(** Metal Q4_0 dequant shader text.  Split out of metal.ml (LOOP item 5). *)

let q4_0_source = {|
struct block_q4_0 {
    half d;
    uint8_t qs[16];
};

kernel void llmopt_dequant_q4_0(
    device const block_q4_0* blocks [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant QuantBlock32Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint block_idx = gid >> 5;
  if (block_idx >= params.total_blocks) return;
  device const block_q4_0& b = blocks[block_idx];
  const half d = b.d;
  const uint8_t byte_val = (lane < 16u) ? b.qs[lane] : b.qs[lane - 16u];
  const uint8_t nib = (lane < 16u) ? (byte_val & 0x0Fu) : (byte_val >> 4);
  const int q = int(nib) - 8;
  output[block_idx * 32 + lane] = half(float(d) * float(q));
}


kernel void llmopt_q4_0_gated_linear_f16_m4(
    device const half* input [[buffer(0)]],
    device const block_q4_0* gate_weight [[buffer(1)]],
    device const block_q4_0* up_weight [[buffer(2)]],
    device half* product [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint m_tiles = (params.m + 3u) >> 2u;
  const uint total_simd = m_tiles * params.n;
  if (simd_idx >= total_simd) return;
  const uint m_tile = simd_idx / params.n;
  const uint col = simd_idx - m_tile * params.n;
  const uint row0 = m_tile << 2u;
  const uint row1 = row0 + 1u;
  const uint row2 = row0 + 2u;
  const uint row3 = row0 + 3u;
  const bool valid1 = row1 < params.m;
  const bool valid2 = row2 < params.m;
  const bool valid3 = row3 < params.m;

  const uint blocks = params.k >> 5u;
  device const uchar* gate_blocks = reinterpret_cast<device const uchar*>(gate_weight) + col * blocks * 18u;
  device const uchar* up_blocks = reinterpret_cast<device const uchar*>(up_weight) + col * blocks * 18u;

  device const half* in0 = input + row0 * params.k;
  device const half* in1 = valid1 ? input + row1 * params.k : in0;
  device const half* in2 = valid2 ? input + row2 * params.k : in0;
  device const half* in3 = valid3 ? input + row3 * params.k : in0;

  float gate_acc0 = 0.0f, up_acc0 = 0.0f;
  float gate_acc1 = 0.0f, up_acc1 = 0.0f;
  float gate_acc2 = 0.0f, up_acc2 = 0.0f;
  float gate_acc3 = 0.0f, up_acc3 = 0.0f;

  const uint half_blocks = blocks >> 1u;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;

  for (uint b = 0u; b < half_blocks; ++b) {
    const uint block_idx = (b << 1u) + b_offset;
    const uint blk_byte_off = block_idx * 18u;
    device const uchar* g_blk = gate_blocks + blk_byte_off;
    device const uchar* u_blk = up_blocks + blk_byte_off;

    const half g_scale = *reinterpret_cast<device const half*>(g_blk);
    const half u_scale = *reinterpret_cast<device const half*>(u_blk);
    const uchar g_packed = g_blk[2u + l];
    const uchar u_packed = u_blk[2u + l];

    const uint k_idx = block_idx * 32u + l;
    const float g_q_low = float(int(g_packed & 15u) - 8);
    const float g_q_high = float(int(g_packed >> 4u) - 8);
    const float u_q_low = float(int(u_packed & 15u) - 8);
    const float u_q_high = float(int(u_packed >> 4u) - 8);
    const float gs = float(g_scale);
    const float us = float(u_scale);

    const float in0_l = float(in0[k_idx]);
    const float in0_h = float(in0[k_idx + 16u]);
    gate_acc0 += gs * (in0_l * g_q_low + in0_h * g_q_high);
    up_acc0 += us * (in0_l * u_q_low + in0_h * u_q_high);

    const float in1_l = float(in1[k_idx]);
    const float in1_h = float(in1[k_idx + 16u]);
    gate_acc1 += gs * (in1_l * g_q_low + in1_h * g_q_high);
    up_acc1 += us * (in1_l * u_q_low + in1_h * u_q_high);

    const float in2_l = float(in2[k_idx]);
    const float in2_h = float(in2[k_idx + 16u]);
    gate_acc2 += gs * (in2_l * g_q_low + in2_h * g_q_high);
    up_acc2 += us * (in2_l * u_q_low + in2_h * u_q_high);

    const float in3_l = float(in3[k_idx]);
    const float in3_h = float(in3[k_idx + 16u]);
    gate_acc3 += gs * (in3_l * g_q_low + in3_h * g_q_high);
    up_acc3 += us * (in3_l * u_q_low + in3_h * u_q_high);
  }

  if ((blocks & 1u) != 0u) {
    const uint block_idx = blocks - 1u;
    const uint blk_byte_off = block_idx * 18u;
    device const uchar* g_blk = gate_blocks + blk_byte_off;
    device const uchar* u_blk = up_blocks + blk_byte_off;
    const half g_scale = *reinterpret_cast<device const half*>(g_blk);
    const half u_scale = *reinterpret_cast<device const half*>(u_blk);
    const uint8_t g_byte = (lane < 16u) ? g_blk[2u + lane] : g_blk[2u + lane - 16u];
    const uint8_t u_byte = (lane < 16u) ? u_blk[2u + lane] : u_blk[2u + lane - 16u];
    const int g_q = int((lane < 16u) ? (g_byte & 0x0Fu) : (g_byte >> 4u)) - 8;
    const int u_q = int((lane < 16u) ? (u_byte & 0x0Fu) : (u_byte >> 4u)) - 8;
    const float g_dec = float(g_scale) * float(g_q);
    const float u_dec = float(u_scale) * float(u_q);
    const uint k_idx = block_idx * 32u + lane;

    gate_acc0 += float(in0[k_idx]) * g_dec;
    up_acc0 += float(in0[k_idx]) * u_dec;
    gate_acc1 += float(in1[k_idx]) * g_dec;
    up_acc1 += float(in1[k_idx]) * u_dec;
    gate_acc2 += float(in2[k_idx]) * g_dec;
    up_acc2 += float(in2[k_idx]) * u_dec;
    gate_acc3 += float(in3[k_idx]) * g_dec;
    up_acc3 += float(in3[k_idx]) * u_dec;
  }

  gate_acc0 = simd_sum(gate_acc0);
  up_acc0 = simd_sum(up_acc0);
  gate_acc1 = simd_sum(gate_acc1);
  up_acc1 = simd_sum(up_acc1);
  gate_acc2 = simd_sum(gate_acc2);
  up_acc2 = simd_sum(up_acc2);
  gate_acc3 = simd_sum(gate_acc3);
  up_acc3 = simd_sum(up_acc3);

  if (lane == 0u) {
    product[row0 * params.n + col] = llmopt_gated_product(gate_acc0, up_acc0, params.has_bias);
    if (valid1)
      product[row1 * params.n + col] = llmopt_gated_product(gate_acc1, up_acc1, params.has_bias);
    if (valid2)
      product[row2 * params.n + col] = llmopt_gated_product(gate_acc2, up_acc2, params.has_bias);
    if (valid3)
      product[row3 * params.n + col] = llmopt_gated_product(gate_acc3, up_acc3, params.has_bias);
  }
}

kernel void llmopt_q4_0_gated_linear_f16_m16(
    device const half* input [[buffer(0)]],
    device const block_q4_0* gate_weight [[buffer(1)]],
    device const block_q4_0* up_weight [[buffer(2)]],
    device half* product [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint m_tiles = (params.m + 15u) >> 4u;
  const uint total_simd = m_tiles * params.n;
  if (simd_idx >= total_simd) return;
  const uint m_tile = simd_idx / params.n;
  const uint col = simd_idx - m_tile * params.n;
  const uint row_base = m_tile << 4u;

  const uint blocks = params.k >> 5u;
  device const uchar* gate_blocks = reinterpret_cast<device const uchar*>(gate_weight) + col * blocks * 18u;
  device const uchar* up_blocks = reinterpret_cast<device const uchar*>(up_weight) + col * blocks * 18u;

  device const half* in_ptrs[16];
  #pragma unroll
  for (uint r = 0; r < 16; ++r) {
    in_ptrs[r] = (row_base + r < params.m) ? input + (row_base + r) * params.k : input;
  }

  float gate_acc[16] = {0.0f};
  float up_acc[16] = {0.0f};

  const uint half_blocks = blocks >> 1u;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;

  for (uint b = 0u; b < half_blocks; ++b) {
    const uint block_idx = (b << 1u) + b_offset;
    const uint blk_byte_off = block_idx * 18u;
    device const uchar* g_blk = gate_blocks + blk_byte_off;
    device const uchar* u_blk = up_blocks + blk_byte_off;

    const half g_scale = *reinterpret_cast<device const half*>(g_blk);
    const half u_scale = *reinterpret_cast<device const half*>(u_blk);
    const uchar g_packed = g_blk[2u + l];
    const uchar u_packed = u_blk[2u + l];

    const uint k_idx = block_idx * 32u + l;
    const float g_q_low = float(int(g_packed & 15u) - 8);
    const float g_q_high = float(int(g_packed >> 4u) - 8);
    const float u_q_low = float(int(u_packed & 15u) - 8);
    const float u_q_high = float(int(u_packed >> 4u) - 8);
    const float gs = float(g_scale);
    const float us = float(u_scale);
    const float g_low_w = gs * g_q_low;
    const float g_high_w = gs * g_q_high;
    const float u_low_w = us * u_q_low;
    const float u_high_w = us * u_q_high;

    #pragma unroll
    for (uint r = 0; r < 16; ++r) {
      const float in_l = float(in_ptrs[r][k_idx]);
      const float in_h = float(in_ptrs[r][k_idx + 16u]);
      gate_acc[r] += in_l * g_low_w + in_h * g_high_w;
      up_acc[r] += in_l * u_low_w + in_h * u_high_w;
    }
  }

  if ((blocks & 1u) != 0u) {
    const uint block_idx = blocks - 1u;
    const uint blk_byte_off = block_idx * 18u;
    device const uchar* g_blk = gate_blocks + blk_byte_off;
    device const uchar* u_blk = up_blocks + blk_byte_off;
    const half g_scale = *reinterpret_cast<device const half*>(g_blk);
    const half u_scale = *reinterpret_cast<device const half*>(u_blk);
    const uint8_t g_byte = (lane < 16u) ? g_blk[2u + lane] : g_blk[2u + lane - 16u];
    const uint8_t u_byte = (lane < 16u) ? u_blk[2u + lane] : u_blk[2u + lane - 16u];
    const int g_q = int((lane < 16u) ? (g_byte & 0x0Fu) : (g_byte >> 4u)) - 8;
    const int u_q = int((lane < 16u) ? (u_byte & 0x0Fu) : (u_byte >> 4u)) - 8;
    const float g_dec = float(g_scale) * float(g_q);
    const float u_dec = float(u_scale) * float(u_q);
    const uint k_idx = block_idx * 32u + lane;

    #pragma unroll
    for (uint r = 0; r < 16; ++r) {
      const float in_val = float(in_ptrs[r][k_idx]);
      gate_acc[r] += in_val * g_dec;
      up_acc[r] += in_val * u_dec;
    }
  }

  #pragma unroll
  for (uint r = 0; r < 16; ++r) {
    gate_acc[r] = simd_sum(gate_acc[r]);
    up_acc[r] = simd_sum(up_acc[r]);
  }

  if (lane == 0u) {
    #pragma unroll
    for (uint r = 0; r < 16; ++r) {
      if (row_base + r < params.m) {
        product[(row_base + r) * params.n + col] = llmopt_gated_product(gate_acc[r], up_acc[r], params.has_bias);
      }
    }
  }
}

kernel void llmopt_q4_0_gated_linear_f16_m32(
    device const half* input [[buffer(0)]],
    device const block_q4_0* gate_weight [[buffer(1)]],
    device const block_q4_0* up_weight [[buffer(2)]],
    device half* product [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint m_tiles = (params.m + 31u) >> 5u;
  const uint total_simd = m_tiles * params.n;
  if (simd_idx >= total_simd) return;
  const uint m_tile = simd_idx / params.n;
  const uint col = simd_idx - m_tile * params.n;
  const uint row_base = m_tile << 5u;

  const uint blocks = params.k >> 5u;
  device const uchar* gate_blocks = reinterpret_cast<device const uchar*>(gate_weight) + col * blocks * 18u;
  device const uchar* up_blocks = reinterpret_cast<device const uchar*>(up_weight) + col * blocks * 18u;

  device const half* in_ptrs[32];
  #pragma unroll
  for (uint r = 0; r < 32; ++r) {
    in_ptrs[r] = (row_base + r < params.m) ? input + (row_base + r) * params.k : input;
  }

  float gate_acc[32] = {0.0f};
  float up_acc[32] = {0.0f};

  const uint half_blocks = blocks >> 1u;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;

  for (uint b = 0u; b < half_blocks; ++b) {
    const uint block_idx = (b << 1u) + b_offset;
    const uint blk_byte_off = block_idx * 18u;
    device const uchar* g_blk = gate_blocks + blk_byte_off;
    device const uchar* u_blk = up_blocks + blk_byte_off;

    const half g_scale = *reinterpret_cast<device const half*>(g_blk);
    const half u_scale = *reinterpret_cast<device const half*>(u_blk);
    const uchar g_packed = g_blk[2u + l];
    const uchar u_packed = u_blk[2u + l];

    const uint k_idx = block_idx * 32u + l;
    const float g_q_low = float(int(g_packed & 15u) - 8);
    const float g_q_high = float(int(g_packed >> 4u) - 8);
    const float u_q_low = float(int(u_packed & 15u) - 8);
    const float u_q_high = float(int(u_packed >> 4u) - 8);
    const float gs = float(g_scale);
    const float us = float(u_scale);
    const float g_low_w = gs * g_q_low;
    const float g_high_w = gs * g_q_high;
    const float u_low_w = us * u_q_low;
    const float u_high_w = us * u_q_high;

    #pragma unroll
    for (uint r = 0; r < 32; ++r) {
      const float in_l = float(in_ptrs[r][k_idx]);
      const float in_h = float(in_ptrs[r][k_idx + 16u]);
      gate_acc[r] += in_l * g_low_w + in_h * g_high_w;
      up_acc[r] += in_l * u_low_w + in_h * u_high_w;
    }
  }

  if ((blocks & 1u) != 0u) {
    const uint block_idx = blocks - 1u;
    const uint blk_byte_off = block_idx * 18u;
    device const uchar* g_blk = gate_blocks + blk_byte_off;
    device const uchar* u_blk = up_blocks + blk_byte_off;
    const half g_scale = *reinterpret_cast<device const half*>(g_blk);
    const half u_scale = *reinterpret_cast<device const half*>(u_blk);
    const uint8_t g_byte = (lane < 16u) ? g_blk[2u + lane] : g_blk[2u + lane - 16u];
    const uint8_t u_byte = (lane < 16u) ? u_blk[2u + lane] : u_blk[2u + lane - 16u];
    const int g_q = int((lane < 16u) ? (g_byte & 0x0Fu) : (g_byte >> 4u)) - 8;
    const int u_q = int((lane < 16u) ? (u_byte & 0x0Fu) : (u_byte >> 4u)) - 8;
    const float g_dec = float(g_scale) * float(g_q);
    const float u_dec = float(u_scale) * float(u_q);
    const uint k_idx = block_idx * 32u + lane;

    #pragma unroll
    for (uint r = 0; r < 32; ++r) {
      const float in_val = float(in_ptrs[r][k_idx]);
      gate_acc[r] += in_val * g_dec;
      up_acc[r] += in_val * u_dec;
    }
  }

  #pragma unroll
  for (uint r = 0; r < 32; ++r) {
    gate_acc[r] = simd_sum(gate_acc[r]);
    up_acc[r] = simd_sum(up_acc[r]);
  }

  if (lane == 0u) {
    #pragma unroll
    for (uint r = 0; r < 32; ++r) {
      if (row_base + r < params.m) {
        product[(row_base + r) * params.n + col] = llmopt_gated_product(gate_acc[r], up_acc[r], params.has_bias);
      }
    }
  }
}

kernel void llmopt_q4_0_dual_swiglu_f16(
    device const half* normalized [[buffer(0)]],
    device const block_q4_0* gate_weight [[buffer(1)]],
    device const block_q4_0* up_weight [[buffer(2)]],
    device half* product [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5;
  const uint total_elements = params.m * params.n;
  if (simd_idx >= total_elements) return;
  const uint row = simd_idx / params.n;
  const uint col = simd_idx - row * params.n;
  const uint num_blocks = params.k >> 5;
  device const block_q4_0* gate_row = gate_weight + col * num_blocks;
  device const block_q4_0* up_row = up_weight + col * num_blocks;
  device const half* in_ptr = normalized + row * params.k;
  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  for (uint b = 0; b < num_blocks; ++b) {
    const half gd = gate_row[b].d;
    const uint8_t g_byte = (lane < 16u) ? gate_row[b].qs[lane] : gate_row[b].qs[lane - 16u];
    const uint8_t g_nib = (lane < 16u) ? (g_byte & 0x0Fu) : (g_byte >> 4);
    const int gq = int(g_nib) - 8;

    const half ud = up_row[b].d;
    const uint8_t u_byte = (lane < 16u) ? up_row[b].qs[lane] : up_row[b].qs[lane - 16u];
    const uint8_t u_nib = (lane < 16u) ? (u_byte & 0x0Fu) : (u_byte >> 4);
    const int uq = int(u_nib) - 8;

    const half in_val = in_ptr[b * 32 + lane];
    gate_acc += float(in_val) * (float(gd) * float(gq));
    up_acc   += float(in_val) * (float(ud) * float(uq));
  }
  gate_acc = simd_sum(gate_acc);
  up_acc = simd_sum(up_acc);
  if (lane == 0) {
    const float silu_gate = gate_acc / (1.0f + exp(-gate_acc));
    product[row * params.n + col] = half(silu_gate * up_acc);
  }
}

kernel void llmopt_q4_0_down_add_f16(
    device const half* product [[buffer(0)]],
    device const block_q4_0* down_weight [[buffer(1)]],
    device const half* residual [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5;
  const uint total_elements = params.m * params.k;
  if (simd_idx >= total_elements) return;
  const uint row = simd_idx / params.k;
  const uint col = simd_idx - row * params.k;
  const uint num_blocks = params.n >> 5;
  device const block_q4_0* down_row = down_weight + col * num_blocks;
  device const half* in_ptr = product + row * params.n;
  float down_acc = 0.0f;
  for (uint b = 0; b < num_blocks; ++b) {
    const half d = down_row[b].d;
    const uint8_t byte_val = (lane < 16u) ? down_row[b].qs[lane] : down_row[b].qs[lane - 16u];
    const uint8_t nib = (lane < 16u) ? (byte_val & 0x0Fu) : (byte_val >> 4);
    const int q = int(nib) - 8;
    const half in_val = in_ptr[b * 32 + lane];
    down_acc += float(in_val) * (float(d) * float(q));
  }
  down_acc = simd_sum(down_acc);
  if (lane == 0) {
    output[row * params.k + col] = half(down_acc + float(residual[row * params.k + col]));
  }
}
|}
