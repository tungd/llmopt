(* EXL3 packing and mul1 formula follow ExLlamaV3 cdc785db89e263e65e5849cb53e1ada156411414.
   Upstream MIT notice: test/fixtures/exl3/UPSTREAM_LICENSE. *)
module Schedule = struct
  type t = Vector of int | Matrix of int
  let vector ~rows =
    if List.mem rows [1; 2; 4; 8] then Ok (Vector rows)
    else Error "EXL3 vector rows must be 1, 2, 4, or 8"
  let matrix ~rows =
    if List.mem rows [8; 16; 32] then Ok (Matrix rows)
    else Error "EXL3 matrix rows must be 8, 16, or 32"
  let rows = function Vector r | Matrix r -> r
  let name = function Vector r -> Printf.sprintf "vector_m%d" r
                       | Matrix r -> Printf.sprintf "matrix_m%d_n16_k16" r
  let threads = function Vector _ -> 128 | Matrix r -> (r / 8) * 2 * 32
  let columns = function Vector _ -> 4 | Matrix _ -> 16
  let grid t ~m ~n =
    let columns = columns t in
    ((n + columns - 1) / columns, (m + rows t - 1) / rows t, 1)
end

let decoder = {|#include <metal_stdlib>
using namespace metal;
struct Exl3Params { uint m, n, k, bits, indexed, repeat_input, top_k, reserved; };

inline half exl3_mul1(uint window) {
  uint x = window * 0x83DCD12Du;
  uint sum = (x & 255u) + ((x >> 8) & 255u) + ((x >> 16) & 255u) + (x >> 24);
  // F32 evaluates the dyadic expression exactly; one final F16 rounding
  // reproduces the upstream half FMA, including ties to even.
  return half(fma(float(1024u + sum), float(as_type<half>(ushort(0x1eee))),
                  float(as_type<half>(ushort(0xc931)))));
}
inline half exl3_weight(device const uint* packed, uint k, uint n, uint n_dim, uint bits) {
  uint r = k & 15u, c = n & 15u;
  uint t = (c & 7u) * 4u + ((r & 7u) >> 1u);
  uint j = (r & 1u) | ((r >> 2u) & 2u) | ((c >> 1u) & 4u);
  uint position = t * 8u + j;
  uint ring = 256u * bits;
  uint start = ((position + 1u) * bits + ring - 16u) % ring;
  uint word = start >> 5u, shift = start & 31u;
  uint tile = (k >> 4u) * (n_dim >> 4u) + (n >> 4u);
  device const uint* p = packed + tile * (8u * bits);
  uint a = p[word], b = p[(word + 1u) % (8u * bits)];
  uint window = shift <= 16u ? ((a >> (16u - shift)) & 65535u)
      : (((a << (shift - 16u)) | (b >> (48u - shift))) & 65535u);
  return exl3_mul1(window);
}
kernel void exl3_unpack(device const uint* w [[buffer(0)]],
    device half* out [[buffer(1)]], constant Exl3Params& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < p.k * p.n) out[i] = exl3_weight(w, i / p.n, i % p.n, p.n, p.bits);
}
kernel void exl3_codebook(device half* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  if (i < 65536u) out[i] = exl3_mul1(i);
}
|}

let transforms epilogue =
  let declarations = Epilogue.msl_buffer_decls ~start_index:5 epilogue in
  let expression = Epilogue.emit_msl_writeback ~acc:"value" ~idx:"idx" epilogue in
  Printf.sprintf {|
// One group owns one 128-element transform. No cross-group synchronization.
kernel void exl3_input(device const half* x [[buffer(0)]],
    device const half* suh [[buffer(1)]], device half* out [[buffer(2)]],
    constant Exl3Params& p [[buffer(4)]], device const uint* indices [[buffer(3)]], uint lane [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
  threadgroup float v[128];
  uint row = group.y;
  uint expert = p.indexed ? indices[row] : 0u;
  uint source_row = p.repeat_input ? row / p.top_k : row;
  uint idx = row * p.k + group.x * 128u + lane;
  v[lane] = float(x[source_row * p.k + group.x * 128u + lane]) * float(suh[expert * p.k + group.x * 128u + lane]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint step = 1u; step < 128u; step <<= 1u) {
    float a = v[lane], b = v[lane ^ step];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v[lane] = (lane & step) ? b - a : a + b;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  out[idx] = half(v[lane] * 0.08838834764831845f);
}
kernel void exl3_output(device const float* x [[buffer(0)]],
    device const half* svh [[buffer(1)]], device half* out [[buffer(2)]],
    constant Exl3Params& p [[buffer(4)]], device const uint* indices [[buffer(3)]]%s,
    uint lane [[thread_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
  threadgroup float v[128];
  uint expert = p.indexed ? indices[group.y] : 0u;
  uint idx = group.y * p.n + group.x * 128u + lane;
  v[lane] = x[idx];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint step = 1u; step < 128u; step <<= 1u) {
    float a = v[lane], b = v[lane ^ step];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v[lane] = (lane & step) ? b - a : a + b;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float value = v[lane] * 0.08838834764831845f * float(svh[expert * p.n + group.x * 128u + lane]);
  out[idx] = half(%s);
}
|} declarations expression

let vector ~name rows = Printf.sprintf {|
kernel void %s(device const half* x [[buffer(0)]],
    device const uint* w [[buffer(1)]], device float* out [[buffer(2)]],
    constant Exl3Params& p [[buffer(4)]], device const uint* indices [[buffer(3)]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
  uint col = group.x * 4u + sg, row = group.y * %du;
  uint expert = p.indexed ? indices[row] : 0u;
  device const uint* weights = w + ulong(expert) * p.k * p.n * p.bits / 32u;
  float acc[%d] = {};
  for (uint k = lane; k < p.k; k += 32u) {
    float weight = float(exl3_weight(weights, k, col, p.n, p.bits));
    #pragma unroll
    for (uint r = 0; r < %du; ++r)
      if (row + r < p.m) acc[r] = fma(float(x[(row + r) * p.k + k]), weight, acc[r]);
  }
  #pragma unroll
  for (uint r = 0; r < %du; ++r) {
    float sum = simd_sum(acc[r]);
    if (lane == 0u && row + r < p.m) out[(row + r) * p.n + col] = sum;
  }
}
|} name rows rows rows rows

let matrix ~name rows = Printf.sprintf {|
kernel void %s(device const half* x [[buffer(0)]],
    device const uint* w [[buffer(1)]], device float* out [[buffer(2)]],
    constant Exl3Params& p [[buffer(4)]], device const uint* indices [[buffer(3)]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
  threadgroup half a[%d * 16];
  threadgroup half b[16 * 16];
  threadgroup float c[%d * 16];
  uint row = group.y * %du, col = group.x * 16u;
  simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (uint base = 0; base < p.k; base += 16u) {
    for (uint i = tid; i < %du * 16u; i += %du) {
      uint r = row + i / 16u;
      a[i] = r < p.m ? x[r * p.k + base + i %% 16u] : half(0);
    }
    for (uint i = tid; i < 256u; i += %du)
      b[i] = exl3_weight(w, base + i / 16u, col + i %% 16u, p.n, p.bits);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 16u; kk += 8u) {
      simdgroup_half8x8 av, bv;
      simdgroup_load(av, a + (sg / 2u) * 8u * 16u + kk, 16);
      simdgroup_load(bv, b + kk * 16u + (sg %% 2u) * 8u, 16);
      simdgroup_multiply_accumulate(acc, av, bv, acc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  simdgroup_store(acc, c + (sg / 2u) * 8u * 16u + (sg %% 2u) * 8u, 16);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < %du * 16u; i += %du) {
    uint r = row + i / 16u;
    if (r < p.m) out[r * p.n + col + i %% 16u] = c[i];
  }
}
|} name rows rows rows rows ((rows / 8) * 64) ((rows / 8) * 64) rows ((rows / 8) * 64)

let emit _storage schedule ~epilogue =
  decoder ^ transforms epilogue ^
  (match schedule with Schedule.Vector rows -> vector ~name:"exl3_contract" rows
                     | Schedule.Matrix rows -> matrix ~name:"exl3_contract" rows)

let auxiliary = {|
kernel void exl3_swiglu(device const half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]], device half* out [[buffer(2)]],
    constant uint& count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i < count) { float g = float(gate[i]); out[i] = half(g / (1.0f + exp(-g)) * float(up[i])); }
}
kernel void exl3_moe_reduce(device const half* down [[buffer(0)]],
    device const half* weights [[buffer(1)]], device half* out [[buffer(2)]],
    constant uint4& p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.x * p.y) return;
  uint token = i / p.y, col = i % p.y;
  float sum = 0.0f;
  for (uint j = 0; j < p.z; ++j) sum += float(down[(token * p.z + j) * p.y + col]) * float(weights[token * p.z + j]);
  out[i] = half(sum);
}
|}

let runtime_source =
  decoder ^ transforms Epilogue.none
  ^ vector ~name:"exl3_contract_vector" 1
  ^ vector ~name:"exl3_contract_vector4" 4
  ^ matrix ~name:"exl3_contract_matrix" 16
  ^ auxiliary

let runtime_kernels =
  ["exl3_input", 128; "exl3_output", 128; "exl3_contract_vector", 128;
   "exl3_contract_vector4", 128; "exl3_contract_matrix", 128;
   "exl3_swiglu", 256; "exl3_moe_reduce", 256]
