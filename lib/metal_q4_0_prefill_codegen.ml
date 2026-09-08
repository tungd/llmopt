open Metal_q4_0_epilogue_codegen

let emit_q4_0_linear_f16_m16 ?(name = "llmopt_q4_0_linear_f16_m16")
    ?(epilogue = Epilogue.none) ~scale_offset ~quant_offset ~zero ~block_elements
    ~block_bytes () =
  let flat = Printf.sprintf {|
kernel void llmopt_q4_0_linear_f16_m16(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint m_tiles = (params.m + 15u) >> 4u;
  const uint total_tiles = m_tiles * params.n;
  if (simd_idx >= total_tiles) return;
  const uint m_tile = simd_idx / params.n;
  const uint column = simd_idx - m_tile * params.n;
  const uint row_base = m_tile << 4u;
  const uint blocks = params.k / %du;
  device const uchar* row_blocks = weight + column * blocks * %du;

  float acc[16] = {0.0f};

  const uint half_blocks = blocks >> 1u;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;

  device const half* in_ptrs[16];
  #pragma unroll
  for (uint r = 0; r < 16; ++r) {
    in_ptrs[r] = (row_base + r < params.m) ? input + (row_base + r) * params.k : input;
  }

  for (uint b = 0u; b < half_blocks; ++b) {
    const uint block_idx = (b << 1u) + b_offset;
    device const uchar* blk = row_blocks + block_idx * %du;
    const half scale = *reinterpret_cast<device const half*>(blk + %du);
    const uchar packed = blk[%du + l];
    const float s = float(scale);
    const uint k = block_idx * %du + l;
    const float q_low = float(int(packed & 15u) - %d);
    const float q_high = float(int(packed >> 4u) - %d);
    const float w_low = s * q_low;
    const float w_high = s * q_high;

    #pragma unroll
    for (uint r = 0; r < 16; ++r) {
      acc[r] += float(in_ptrs[r][k]) * w_low + float(in_ptrs[r][k + 16u]) * w_high;
    }
  }

  if ((blocks & 1u) != 0u) {
    const uint block_idx = blocks - 1u;
    device const uchar* blk = row_blocks + block_idx * %du;
    const half scale = *reinterpret_cast<device const half*>(blk + %du);
    const uint8_t byte_val = (lane < 16u) ? blk[%du + lane] : blk[%du + lane - 16u];
    const uint8_t nib = (lane < 16u) ? (byte_val & 0x0Fu) : (byte_val >> 4u);
    const float decoded = float(scale) * float(int(nib) - %d);
    const uint k = block_idx * %du + lane;
    #pragma unroll
    for (uint r = 0; r < 16; ++r) {
      acc[r] += float(in_ptrs[r][k]) * decoded;
    }
  }

  #pragma unroll
  for (uint r = 0; r < 16; ++r) {
    acc[r] = simd_sum(acc[r]);
  }

  if (lane == 0u) {
    const float b_val = (params.has_bias != 0u) ? float(bias[column]) : 0.0f;
    #pragma unroll
    for (uint r = 0; r < 16; ++r) {
      if (row_base + r < params.m) {
        output[(row_base + r) * params.n + column] = half(acc[r] + b_val);
      }
    }
  }
}
|}
    block_elements block_bytes block_bytes scale_offset quant_offset
    block_elements zero zero block_bytes scale_offset quant_offset
    quant_offset zero block_elements
  in
  if Epilogue.equal epilogue Epilogue.none then flat
  else
    match q4_0_m16_digest_of_flat ~name epilogue flat with
    | Ok text -> text
    | Error message ->
        invalid_arg
          (Printf.sprintf "Metal_block_codegen q4_0_linear_f16_m16: %s" message)

let emit_q4_0_linear_f16_m16_add ~scale_offset ~quant_offset ~zero ~block_elements
    ~block_bytes =
  Printf.sprintf {|
kernel void llmopt_q4_0_linear_f16_m16_add(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint m_tiles = (params.m + 15u) >> 4u;
  const uint total_tiles = m_tiles * params.n;
  if (simd_idx >= total_tiles) return;
  const uint m_tile = simd_idx / params.n;
  const uint column = simd_idx - m_tile * params.n;
  const uint row_base = m_tile << 4u;
  const uint blocks = params.k / %du;
  device const uchar* row_blocks = weight + column * blocks * %du;

  float acc[16] = {0.0f};

  const uint half_blocks = blocks >> 1u;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;

  device const half* in_ptrs[16];
  #pragma unroll
  for (uint r = 0; r < 16; ++r) {
    in_ptrs[r] = (row_base + r < params.m) ? input + (row_base + r) * params.k : input;
  }

  for (uint b = 0u; b < half_blocks; ++b) {
    const uint block_idx = (b << 1u) + b_offset;
    device const uchar* blk = row_blocks + block_idx * %du;
    const half scale = *reinterpret_cast<device const half*>(blk + %du);
    const uchar packed = blk[%du + l];
    const float s = float(scale);
    const uint k = block_idx * %du + l;
    const float q_low = float(int(packed & 15u) - %d);
    const float q_high = float(int(packed >> 4u) - %d);
    const float w_low = s * q_low;
    const float w_high = s * q_high;

    #pragma unroll
    for (uint r = 0; r < 16; ++r) {
      acc[r] += float(in_ptrs[r][k]) * w_low + float(in_ptrs[r][k + 16u]) * w_high;
    }
  }

  if ((blocks & 1u) != 0u) {
    const uint block_idx = blocks - 1u;
    device const uchar* blk = row_blocks + block_idx * %du;
    const half scale = *reinterpret_cast<device const half*>(blk + %du);
    const uint8_t byte_val = (lane < 16u) ? blk[%du + lane] : blk[%du + lane - 16u];
    const uint8_t nib = (lane < 16u) ? (byte_val & 0x0Fu) : (byte_val >> 4u);
    const float decoded = float(scale) * float(int(nib) - %d);
    const uint k = block_idx * %du + lane;
    #pragma unroll
    for (uint r = 0; r < 16; ++r) {
      acc[r] += float(in_ptrs[r][k]) * decoded;
    }
  }

  #pragma unroll
  for (uint r = 0; r < 16; ++r) {
    acc[r] = simd_sum(acc[r]);
  }

  if (lane == 0u) {
    #pragma unroll
    for (uint r = 0; r < 16; ++r) {
      if (row_base + r < params.m) {
        const uint out_idx = (row_base + r) * params.n + column;
        output[out_idx] = half(half(acc[r]) + bias[out_idx]);
      }
    }
  }
}
|}
    block_elements block_bytes block_bytes scale_offset quant_offset
    block_elements zero zero block_bytes scale_offset quant_offset
    quant_offset zero block_elements

let emit_q4_0_linear_f16_m32 ?(name = "llmopt_q4_0_linear_f16_m32")
    ?(epilogue = Epilogue.none) ~scale_offset ~quant_offset ~zero ~block_elements
    ~block_bytes () =
  let flat = Printf.sprintf {|
kernel void llmopt_q4_0_linear_f16_m32(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint m_tiles = (params.m + 31u) >> 5u;
  const uint total_tiles = m_tiles * params.n;
  const uint tile_idx = simd_idx >> 1u;
  const uint simd_in_tile = simd_idx & 1u;
  if (tile_idx >= total_tiles) return;
  const uint m_tile = tile_idx / params.n;
  const uint column = tile_idx - m_tile * params.n;
  const uint row_base = m_tile << 5u;
  const uint sub_row_base = row_base + (simd_in_tile << 4u);
  const uint blocks = params.k / %du;
  device const uchar* row_blocks = weight + column * blocks * %du;

  float acc[16] = {0.0f};

  const uint half_blocks = blocks >> 1u;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;

  device const half* in_ptrs[16];
  #pragma unroll
  for (uint r = 0; r < 16; ++r) {
    in_ptrs[r] = (sub_row_base + r < params.m) ? input + (sub_row_base + r) * params.k : input;
  }

  for (uint b = 0u; b < half_blocks; ++b) {
    const uint block_idx = (b << 1u) + b_offset;
    device const uchar* blk = row_blocks + block_idx * %du;
    const half scale = *reinterpret_cast<device const half*>(blk + %du);
    const uchar packed = blk[%du + l];
    const float s = float(scale);
    const uint k = block_idx * %du + l;
    const float q_low = float(int(packed & 15u) - %d);
    const float q_high = float(int(packed >> 4u) - %d);
    const float w_low = s * q_low;
    const float w_high = s * q_high;

    #pragma unroll
    for (uint r = 0; r < 16; ++r) {
      acc[r] += float(in_ptrs[r][k]) * w_low + float(in_ptrs[r][k + 16u]) * w_high;
    }
  }

  if ((blocks & 1u) != 0u) {
    const uint block_idx = blocks - 1u;
    device const uchar* blk = row_blocks + block_idx * %du;
    const half scale = *reinterpret_cast<device const half*>(blk + %du);
    const uint8_t byte_val = (lane < 16u) ? blk[%du + lane] : blk[%du + lane - 16u];
    const uint8_t nib = (lane < 16u) ? (byte_val & 0x0Fu) : (byte_val >> 4u);
    const float decoded = float(scale) * float(int(nib) - %d);
    const uint k = block_idx * %du + lane;
    #pragma unroll
    for (uint r = 0; r < 16; ++r) {
      acc[r] += float(in_ptrs[r][k]) * decoded;
    }
  }

  #pragma unroll
  for (uint r = 0; r < 16; ++r) {
    acc[r] = simd_sum(acc[r]);
  }

  if (lane == 0u) {
    const float b_val = (params.has_bias != 0u) ? float(bias[column]) : 0.0f;
    #pragma unroll
    for (uint r = 0; r < 16; ++r) {
      if (sub_row_base + r < params.m) {
        output[(sub_row_base + r) * params.n + column] = half(acc[r] + b_val);
      }
    }
  }
}
|}
    block_elements block_bytes block_bytes scale_offset quant_offset
    block_elements zero zero block_bytes scale_offset quant_offset
    quant_offset zero block_elements
  in
  if Epilogue.equal epilogue Epilogue.none then flat
  else
    match q4_0_m32_digest_of_flat ~name epilogue flat with
    | Ok text -> text
    | Error message ->
        invalid_arg
          (Printf.sprintf "Metal_block_codegen q4_0_linear_f16_m32: %s" message)

let emit_q4_0_linear_f16_m32_add ~scale_offset ~quant_offset ~zero ~block_elements
    ~block_bytes =
  Printf.sprintf {|
kernel void llmopt_q4_0_linear_f16_m32_add(
    device const half* input [[buffer(0)]],
    device const uchar* weight [[buffer(1)]],
    device const half* bias [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint m_tiles = (params.m + 31u) >> 5u;
  const uint total_tiles = m_tiles * params.n;
  const uint tile_idx = simd_idx >> 1u;
  const uint simd_in_tile = simd_idx & 1u;
  if (tile_idx >= total_tiles) return;
  const uint m_tile = tile_idx / params.n;
  const uint column = tile_idx - m_tile * params.n;
  const uint row_base = m_tile << 5u;
  const uint sub_row_base = row_base + (simd_in_tile << 4u);
  const uint blocks = params.k / %du;
  device const uchar* row_blocks = weight + column * blocks * %du;

  float acc[16] = {0.0f};

  const uint half_blocks = blocks >> 1u;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;

  device const half* in_ptrs[16];
  #pragma unroll
  for (uint r = 0; r < 16; ++r) {
    in_ptrs[r] = (sub_row_base + r < params.m) ? input + (sub_row_base + r) * params.k : input;
  }

  for (uint b = 0u; b < half_blocks; ++b) {
    const uint block_idx = (b << 1u) + b_offset;
    device const uchar* blk = row_blocks + block_idx * %du;
    const half scale = *reinterpret_cast<device const half*>(blk + %du);
    const uchar packed = blk[%du + l];
    const float s = float(scale);
    const uint k = block_idx * %du + l;
    const float q_low = float(int(packed & 15u) - %d);
    const float q_high = float(int(packed >> 4u) - %d);
    const float w_low = s * q_low;
    const float w_high = s * q_high;

    #pragma unroll
    for (uint r = 0; r < 16; ++r) {
      acc[r] += float(in_ptrs[r][k]) * w_low + float(in_ptrs[r][k + 16u]) * w_high;
    }
  }

  if ((blocks & 1u) != 0u) {
    const uint block_idx = blocks - 1u;
    device const uchar* blk = row_blocks + block_idx * %du;
    const half scale = *reinterpret_cast<device const half*>(blk + %du);
    const uint8_t byte_val = (lane < 16u) ? blk[%du + lane] : blk[%du + lane - 16u];
    const uint8_t nib = (lane < 16u) ? (byte_val & 0x0Fu) : (byte_val >> 4u);
    const float decoded = float(scale) * float(int(nib) - %d);
    const uint k = block_idx * %du + lane;
    #pragma unroll
    for (uint r = 0; r < 16; ++r) {
      acc[r] += float(in_ptrs[r][k]) * decoded;
    }
  }

  #pragma unroll
  for (uint r = 0; r < 16; ++r) {
    acc[r] = simd_sum(acc[r]);
  }

  if (lane == 0u) {
    #pragma unroll
    for (uint r = 0; r < 16; ++r) {
      if (sub_row_base + r < params.m) {
        const uint out_idx = (sub_row_base + r) * params.n + column;
        output[out_idx] = half(half(acc[r]) + bias[out_idx]);
      }
    }
  }
}
|}
    block_elements block_bytes block_bytes scale_offset quant_offset
    block_elements zero zero block_bytes scale_offset quant_offset
    quant_offset zero block_elements

(* G4P12-04d-A2: parametric-style _add twins for the generic (arbitrary-row,
   llmopt_q4_0_linear_f16) and paired-row (_m2) tiles.  Metal's legacy
   quant_linear_add_source text-patcher used to produce these two twins from
   the llama-style legacy bodies; they are now emitted here in the same
   parametric style as the m4/m16/m32 zoo twins, keeping the legacy twin
   numerics contract: the additive tensor (the residual, bound to the [bias]
   buffer(2) slot as in the flat kernels) is applied element-wise with the
   double-rounding store half(half(acc) + bias[idx]) -- numerically distinct
   from the single-rounding digest Residual_add tail.  Same ABI as the flat
   kernels: input buffer(0), weight buffer(1), bias buffer(2), output
   buffer(3), params buffer(4); the decode order matches the parametric flat
   bodies already serving plain decode. *)
