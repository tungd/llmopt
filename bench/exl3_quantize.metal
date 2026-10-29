// Approximate mul1 encoder: two-step lookahead, two circular warm-up sweeps.
// One SIMD group owns a tile. Runtime decoding comes from the compiler emitter.
inline uint exl3_position(uint r, uint c) {
  return ((c & 7u) * 4u + ((r & 7u) >> 1u)) * 8u
    + ((r & 1u) | ((r >> 2u) & 2u) | ((c >> 1u) & 4u));
}
kernel void exl3_quantize(device const float* input [[buffer(0)]],
    device uint* output [[buffer(1)]], constant uint2& dims [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]], uint tile [[threadgroup_position_in_grid]]) {
  threadgroup float target[256];
  uint kt = tile / (dims.y / 16u), nt = tile % (dims.y / 16u);
  for (uint i = lane; i < 256; i += 32) {
    uint r = i / 16u, c = i % 16u;
    target[exl3_position(r, c)] = input[(kt * 16u + r) * dims.y + nt * 16u + c];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint state = 0, word = 0;
  for (uint sweep = 0; sweep < 3; ++sweep) {
    for (uint i = 0; i < 256; ++i) {
      uint next = ((state << 4u) | lane) & 65535u;
      float delta = float(exl3_mul1(next)) - target[i];
      float cost = INFINITY;
      if (lane < 16) {
        float look = INFINITY;
        for (uint b = 0; b < 16; ++b) {
          float d = float(exl3_mul1(((next << 4u) | b) & 65535u)) - target[(i + 1u) & 255u];
          look = min(look, d * d);
        }
        cost = delta * delta + look;
      }
      float best = simd_min(cost);
      uint choice = simd_min(cost == best ? lane : 32u);
      state = ((state << 4u) | choice) & 65535u;
      word = (word << 4u) | choice;
      if (sweep == 2 && (i & 7u) == 7u && lane == 0)
        output[tile * 32u + i / 8u] = word;
    }
  }
}
