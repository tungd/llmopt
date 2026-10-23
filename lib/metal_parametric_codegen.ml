(** Metal parametric shader emitters: w4a16_linear and the
   simdgroup 8x8 matrix gemm, parameterized by epilogue text. *)

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
