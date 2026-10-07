// Benchmark baseline vs 4b-vectorized Q4_0 gated linear kernel on Apple M4 Pro
import Foundation
import Metal

func require(_ condition: Bool, _ message: String) {
    if !condition { fatalError(message) }
}

guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    fatalError("Metal unavailable")
}

let mslSource = """
#include <metal_stdlib>
using namespace metal;

struct QuantLinearParams {
    uint m;
    uint n;
    uint k;
    uint has_bias;
};

inline float gelu_quick(float x) {
    return x * (1.0f / (1.0f + metal::exp(-1.702f * x)));
}

// -----------------------------------------------------------------------------
// Baseline Gated Linear (from metal_q4_0_decode_codegen.ml)
// -----------------------------------------------------------------------------
kernel void q4_0_gated_linear_baseline_x8(
    device const half* input [[buffer(0)]],
    device const uchar* gate_weight [[buffer(1)]],
    device const uchar* up_weight [[buffer(2)]],
    device half* product [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint col_groups = (params.n + 7u) >> 3u;
  const uint total_elements = params.m * col_groups;
  if (simd_idx >= total_elements) return;

  const uint row = simd_idx / col_groups;
  const uint group_idx = simd_idx - row * col_groups;
  const uint base_col = group_idx << 3u;

  device const half* in_ptr = input + row * params.k;
  const uint blocks = params.k / 32u;
  const uint col_stride = blocks * 18u;
  device const uchar* gate_base = gate_weight;
  device const uchar* up_base = up_weight;

  device const uchar* gate_row0 = gate_base + base_col * col_stride;
  device const uchar* gate_row1 = gate_base + min(base_col + 1u, params.n - 1u) * col_stride;
  device const uchar* gate_row2 = gate_base + min(base_col + 2u, params.n - 1u) * col_stride;
  device const uchar* gate_row3 = gate_base + min(base_col + 3u, params.n - 1u) * col_stride;
  device const uchar* gate_row4 = gate_base + min(base_col + 4u, params.n - 1u) * col_stride;
  device const uchar* gate_row5 = gate_base + min(base_col + 5u, params.n - 1u) * col_stride;
  device const uchar* gate_row6 = gate_base + min(base_col + 6u, params.n - 1u) * col_stride;
  device const uchar* gate_row7 = gate_base + min(base_col + 7u, params.n - 1u) * col_stride;

  device const uchar* up_row0 = up_base + base_col * col_stride;
  device const uchar* up_row1 = up_base + min(base_col + 1u, params.n - 1u) * col_stride;
  device const uchar* up_row2 = up_base + min(base_col + 2u, params.n - 1u) * col_stride;
  device const uchar* up_row3 = up_base + min(base_col + 3u, params.n - 1u) * col_stride;
  device const uchar* up_row4 = up_base + min(base_col + 4u, params.n - 1u) * col_stride;
  device const uchar* up_row5 = up_base + min(base_col + 5u, params.n - 1u) * col_stride;
  device const uchar* up_row6 = up_base + min(base_col + 6u, params.n - 1u) * col_stride;
  device const uchar* up_row7 = up_base + min(base_col + 7u, params.n - 1u) * col_stride;

  float gate_acc0 = 0.0f, gate_acc1 = 0.0f, gate_acc2 = 0.0f, gate_acc3 = 0.0f;
  float gate_acc4 = 0.0f, gate_acc5 = 0.0f, gate_acc6 = 0.0f, gate_acc7 = 0.0f;
  float up_acc0 = 0.0f, up_acc1 = 0.0f, up_acc2 = 0.0f, up_acc3 = 0.0f;
  float up_acc4 = 0.0f, up_acc5 = 0.0f, up_acc6 = 0.0f, up_acc7 = 0.0f;

  const uint half_blocks = blocks >> 1u;
  const uint l = lane & 15u;
  const uint b_offset = lane >> 4u;

  for (uint b = 0u; b < half_blocks; ++b) {
    const uint block_idx = (b << 1u) + b_offset;
    const uint blk_byte_off = block_idx * 18u;

    device const uchar* g_blk0 = gate_row0 + blk_byte_off;
    device const uchar* g_blk1 = gate_row1 + blk_byte_off;
    device const uchar* g_blk2 = gate_row2 + blk_byte_off;
    device const uchar* g_blk3 = gate_row3 + blk_byte_off;
    device const uchar* g_blk4 = gate_row4 + blk_byte_off;
    device const uchar* g_blk5 = gate_row5 + blk_byte_off;
    device const uchar* g_blk6 = gate_row6 + blk_byte_off;
    device const uchar* g_blk7 = gate_row7 + blk_byte_off;

    device const uchar* u_blk0 = up_row0 + blk_byte_off;
    device const uchar* u_blk1 = up_row1 + blk_byte_off;
    device const uchar* u_blk2 = up_row2 + blk_byte_off;
    device const uchar* u_blk3 = up_row3 + blk_byte_off;
    device const uchar* u_blk4 = up_row4 + blk_byte_off;
    device const uchar* u_blk5 = up_row5 + blk_byte_off;
    device const uchar* u_blk6 = up_row6 + blk_byte_off;
    device const uchar* u_blk7 = up_row7 + blk_byte_off;

    const half g_scale0 = *reinterpret_cast<device const half*>(g_blk0);
    const half g_scale1 = *reinterpret_cast<device const half*>(g_blk1);
    const half g_scale2 = *reinterpret_cast<device const half*>(g_blk2);
    const half g_scale3 = *reinterpret_cast<device const half*>(g_blk3);
    const half g_scale4 = *reinterpret_cast<device const half*>(g_blk4);
    const half g_scale5 = *reinterpret_cast<device const half*>(g_blk5);
    const half g_scale6 = *reinterpret_cast<device const half*>(g_blk6);
    const half g_scale7 = *reinterpret_cast<device const half*>(g_blk7);

    const half u_scale0 = *reinterpret_cast<device const half*>(u_blk0);
    const half u_scale1 = *reinterpret_cast<device const half*>(u_blk1);
    const half u_scale2 = *reinterpret_cast<device const half*>(u_blk2);
    const half u_scale3 = *reinterpret_cast<device const half*>(u_blk3);
    const half u_scale4 = *reinterpret_cast<device const half*>(u_blk4);
    const half u_scale5 = *reinterpret_cast<device const half*>(u_blk5);
    const half u_scale6 = *reinterpret_cast<device const half*>(u_blk6);
    const half u_scale7 = *reinterpret_cast<device const half*>(u_blk7);

    const uchar g_packed0 = g_blk0[2u + l];
    const uchar g_packed1 = g_blk1[2u + l];
    const uchar g_packed2 = g_blk2[2u + l];
    const uchar g_packed3 = g_blk3[2u + l];
    const uchar g_packed4 = g_blk4[2u + l];
    const uchar g_packed5 = g_blk5[2u + l];
    const uchar g_packed6 = g_blk6[2u + l];
    const uchar g_packed7 = g_blk7[2u + l];

    const uchar u_packed0 = u_blk0[2u + l];
    const uchar u_packed1 = u_blk1[2u + l];
    const uchar u_packed2 = u_blk2[2u + l];
    const uchar u_packed3 = u_blk3[2u + l];
    const uchar u_packed4 = u_blk4[2u + l];
    const uchar u_packed5 = u_blk5[2u + l];
    const uchar u_packed6 = u_blk6[2u + l];
    const uchar u_packed7 = u_blk7[2u + l];

    const uint k_idx = block_idx * 32u + l;
    const float a_low = float(in_ptr[k_idx]);
    const float a_high = float(in_ptr[k_idx + 16u]);

    #define DEQ(p) float(int(p & 15u) - 8), float(int(p >> 4u) - 8)
    float g_l0, g_h0; g_l0 = float(int(g_packed0 & 15u) - 8); g_h0 = float(int(g_packed0 >> 4u) - 8);
    float g_l1, g_h1; g_l1 = float(int(g_packed1 & 15u) - 8); g_h1 = float(int(g_packed1 >> 4u) - 8);
    float g_l2, g_h2; g_l2 = float(int(g_packed2 & 15u) - 8); g_h2 = float(int(g_packed2 >> 4u) - 8);
    float g_l3, g_h3; g_l3 = float(int(g_packed3 & 15u) - 8); g_h3 = float(int(g_packed3 >> 4u) - 8);
    float g_l4, g_h4; g_l4 = float(int(g_packed4 & 15u) - 8); g_h4 = float(int(g_packed4 >> 4u) - 8);
    float g_l5, g_h5; g_l5 = float(int(g_packed5 & 15u) - 8); g_h5 = float(int(g_packed5 >> 4u) - 8);
    float g_l6, g_h6; g_l6 = float(int(g_packed6 & 15u) - 8); g_h6 = float(int(g_packed6 >> 4u) - 8);
    float g_l7, g_h7; g_l7 = float(int(g_packed7 & 15u) - 8); g_h7 = float(int(g_packed7 >> 4u) - 8);

    float u_l0, u_h0; u_l0 = float(int(u_packed0 & 15u) - 8); u_h0 = float(int(u_packed0 >> 4u) - 8);
    float u_l1, u_h1; u_l1 = float(int(u_packed1 & 15u) - 8); u_h1 = float(int(u_packed1 >> 4u) - 8);
    float u_l2, u_h2; u_l2 = float(int(u_packed2 & 15u) - 8); u_h2 = float(int(u_packed2 >> 4u) - 8);
    float u_l3, u_h3; u_l3 = float(int(u_packed3 & 15u) - 8); u_h3 = float(int(u_packed3 >> 4u) - 8);
    float u_l4, u_h4; u_l4 = float(int(u_packed4 & 15u) - 8); u_h4 = float(int(u_packed4 >> 4u) - 8);
    float u_l5, u_h5; u_l5 = float(int(u_packed5 & 15u) - 8); u_h5 = float(int(u_packed5 >> 4u) - 8);
    float u_l6, u_h6; u_l6 = float(int(u_packed6 & 15u) - 8); u_h6 = float(int(u_packed6 >> 4u) - 8);
    float u_l7, u_h7; u_l7 = float(int(u_packed7 & 15u) - 8); u_h7 = float(int(u_packed7 >> 4u) - 8);

    gate_acc0 += float(g_scale0) * (a_low * g_l0 + a_high * g_h0);
    gate_acc1 += float(g_scale1) * (a_low * g_l1 + a_high * g_h1);
    gate_acc2 += float(g_scale2) * (a_low * g_l2 + a_high * g_h2);
    gate_acc3 += float(g_scale3) * (a_low * g_l3 + a_high * g_h3);
    gate_acc4 += float(g_scale4) * (a_low * g_l4 + a_high * g_h4);
    gate_acc5 += float(g_scale5) * (a_low * g_l5 + a_high * g_h5);
    gate_acc6 += float(g_scale6) * (a_low * g_l6 + a_high * g_h6);
    gate_acc7 += float(g_scale7) * (a_low * g_l7 + a_high * g_h7);

    up_acc0 += float(u_scale0) * (a_low * u_l0 + a_high * u_h0);
    up_acc1 += float(u_scale1) * (a_low * u_l1 + a_high * u_h1);
    up_acc2 += float(u_scale2) * (a_low * u_l2 + a_high * u_h2);
    up_acc3 += float(u_scale3) * (a_low * u_l3 + a_high * u_h3);
    up_acc4 += float(u_scale4) * (a_low * u_l4 + a_high * u_h4);
    up_acc5 += float(u_scale5) * (a_low * u_l5 + a_high * u_h5);
    up_acc6 += float(u_scale6) * (a_low * u_l6 + a_high * u_h6);
    up_acc7 += float(u_scale7) * (a_low * u_l7 + a_high * u_h7);
    #undef DEQ
  }

  gate_acc0 = simd_sum(gate_acc0); gate_acc1 = simd_sum(gate_acc1);
  gate_acc2 = simd_sum(gate_acc2); gate_acc3 = simd_sum(gate_acc3);
  gate_acc4 = simd_sum(gate_acc4); gate_acc5 = simd_sum(gate_acc5);
  gate_acc6 = simd_sum(gate_acc6); gate_acc7 = simd_sum(gate_acc7);

  up_acc0 = simd_sum(up_acc0); up_acc1 = simd_sum(up_acc1);
  up_acc2 = simd_sum(up_acc2); up_acc3 = simd_sum(up_acc3);
  up_acc4 = simd_sum(up_acc4); up_acc5 = simd_sum(up_acc5);
  up_acc6 = simd_sum(up_acc6); up_acc7 = simd_sum(up_acc7);

  if (lane == 0u) {
    device half* out_row = product + row * params.n;
    if (base_col < params.n) out_row[base_col] = half(gate_acc0 * up_acc0);
    if (base_col + 1u < params.n) out_row[base_col + 1u] = half(gate_acc1 * up_acc1);
    if (base_col + 2u < params.n) out_row[base_col + 2u] = half(gate_acc2 * up_acc2);
    if (base_col + 3u < params.n) out_row[base_col + 3u] = half(gate_acc3 * up_acc3);
    if (base_col + 4u < params.n) out_row[base_col + 4u] = half(gate_acc4 * up_acc4);
    if (base_col + 5u < params.n) out_row[base_col + 5u] = half(gate_acc5 * up_acc5);
    if (base_col + 6u < params.n) out_row[base_col + 6u] = half(gate_acc6 * up_acc6);
    if (base_col + 7u < params.n) out_row[base_col + 7u] = half(gate_acc7 * up_acc7);
  }
}

// -----------------------------------------------------------------------------
// Vectorized 4B Gated Linear (4 threads per block, 32 threads do 8 blocks = 256 K)
// -----------------------------------------------------------------------------
kernel void q4_0_gated_linear_vectorized_4b_x8(
    device const half* input [[buffer(0)]],
    device const uchar* gate_weight [[buffer(1)]],
    device const uchar* up_weight [[buffer(2)]],
    device half* product [[buffer(3)]],
    constant QuantLinearParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint simd_idx = gid >> 5u;
  const uint col_groups = (params.n + 7u) >> 3u;
  const uint total_elements = params.m * col_groups;
  if (simd_idx >= total_elements) return;

  const uint row = simd_idx / col_groups;
  const uint group_idx = simd_idx - row * col_groups;
  const uint base_col = group_idx << 3u;

  device const half* in_ptr = input + row * params.k;
  const uint blocks = params.k / 32u;
  const uint col_stride = blocks * 18u;

  device const uchar* g_rows[8];
  device const uchar* u_rows[8];
  for (uint i = 0u; i < 8u; ++i) {
    uint c = min(base_col + i, params.n - 1u);
    g_rows[i] = gate_weight + c * col_stride;
    u_rows[i] = up_weight + c * col_stride;
  }

  float g_acc[8] = {0.0f};
  float u_acc[8] = {0.0f};

  const uint b_sub = lane >> 2u; // 0..7
  const uint t_in_b = lane & 3u; // 0..3 (4 bytes = 8 weights)
  const uint byte_offset = 2u + (t_in_b << 2u);

  const uint cluster_blocks = blocks >> 3u;
  for (uint c = 0u; c < cluster_blocks; ++c) {
    const uint block_idx = (c << 3u) + b_sub;
    const uint blk_byte_off = block_idx * 18u;

    float g_scales[8], u_scales[8];
    ushort g_lo[8], g_hi[8];
    ushort u_lo[8], u_hi[8];

    for (uint i = 0u; i < 8u; ++i) {
      device const uchar* g_blk = g_rows[i] + blk_byte_off;
      device const uchar* u_blk = u_rows[i] + blk_byte_off;
      g_scales[i] = float(*reinterpret_cast<device const half*>(g_blk));
      u_scales[i] = float(*reinterpret_cast<device const half*>(u_blk));
      g_lo[i] = *reinterpret_cast<device const ushort*>(g_blk + byte_offset);
      g_hi[i] = *reinterpret_cast<device const ushort*>(g_blk + byte_offset + 2u);
      u_lo[i] = *reinterpret_cast<device const ushort*>(u_blk + byte_offset);
      u_hi[i] = *reinterpret_cast<device const ushort*>(u_blk + byte_offset + 2u);
    }

    const uint l0 = t_in_b << 2u;
    const uint k_idx = block_idx * 32u + l0;
    const float a0_lo = float(in_ptr[k_idx]);
    const float a0_hi = float(in_ptr[k_idx + 16u]);
    const float a1_lo = float(in_ptr[k_idx + 1u]);
    const float a1_hi = float(in_ptr[k_idx + 17u]);
    const float a2_lo = float(in_ptr[k_idx + 2u]);
    const float a2_hi = float(in_ptr[k_idx + 18u]);
    const float a3_lo = float(in_ptr[k_idx + 3u]);
    const float a3_hi = float(in_ptr[k_idx + 19u]);

    const float a_sum8 = 8.0f * (a0_lo + a0_hi + a1_lo + a1_hi + a2_lo + a2_hi + a3_lo + a3_hi);

    #define DOT_QUAD(plo, phi) \\
        (a0_lo * float(plo & 15u) + \\
         a0_hi * float((plo >> 4u) & 15u) + \\
         a1_lo * float((plo >> 8u) & 15u) + \\
         a1_hi * float((plo >> 12u) & 15u) + \\
         a2_lo * float(phi & 15u) + \\
         a2_hi * float((phi >> 4u) & 15u) + \\
         a3_lo * float((phi >> 8u) & 15u) + \\
         a3_hi * float((phi >> 12u) & 15u) - a_sum8)

    for (uint i = 0u; i < 8u; ++i) {
      g_acc[i] += g_scales[i] * DOT_QUAD(g_lo[i], g_hi[i]);
      u_acc[i] += u_scales[i] * DOT_QUAD(u_lo[i], u_hi[i]);
    }
    #undef DOT_QUAD
  }

  for (uint i = 0u; i < 8u; ++i) {
    g_acc[i] = simd_sum(g_acc[i]);
    u_acc[i] = simd_sum(u_acc[i]);
  }

  if (lane == 0u) {
    device half* out_row = product + row * params.n;
    for (uint i = 0u; i < 8u; ++i) {
      if (base_col + i < params.n) {
        out_row[base_col + i] = half(g_acc[i] * u_acc[i]);
      }
    }
  }
}
""";

let library = try device.makeLibrary(source: mslSource, options: nil)
let kernelNames = ["q4_0_gated_linear_baseline_x8", "q4_0_gated_linear_vectorized_4b_x8"]
let pipelines = try kernelNames.map { name -> MTLComputePipelineState in
    guard let fn = library.makeFunction(name: name) else { fatalError("missing \(name)") }
    return try device.makeComputePipelineState(function: fn)
}

func buffer(_ length: Int) -> MTLBuffer {
    device.makeBuffer(length: length, options: .storageModeShared)!
}
func finish(_ command: MTLCommandBuffer) {
    command.commit()
    command.waitUntilCompleted()
    require(command.status == .completed, "GPU command failed: \\(String(describing: command.error))")
}

let eviction = buffer(256 * 1024 * 1024)
func perturbCache() {
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeBlitCommandEncoder()!
    encoder.fill(buffer: eviction, range: 0..<eviction.length, value: 0xa5)
    encoder.endEncoding()
    finish(command)
}

print("device=\(device.name)")
print("Benchmarking GATED LINEAR Q4_0 kernel on Gemma FFN shape (N=15360, K=3840)")
print("Total weights streamed: 2x 31.6 MB = 63.3 MB per dispatch")
print("-----------------------------------------------------------------------------------------------")

let n = 15360
let k = 3840
let blocks = k / 32
let weightBytes = n * blocks * 18
let gateWeight = buffer(weightBytes)
let upWeight = buffer(weightBytes)
let input = buffer(k * 2)
let output = buffer(n * 2)

let xp = input.contents().bindMemory(to: Float16.self, capacity: k)
let gwp = gateWeight.contents().bindMemory(to: UInt8.self, capacity: weightBytes)
let uwp = upWeight.contents().bindMemory(to: UInt8.self, capacity: weightBytes)

var x = [Int32](repeating: 0, count: k)
var rng: UInt32 = 0x12345678
func random() -> UInt32 {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng
}
for i in 0..<k {
    x[i] = Int32(random() % 17) - 8
    xp[i] = Float16(Float(x[i]) / 16)
}
for b in 0..<(n * blocks) {
    gwp[b * 18] = 0; gwp[b * 18 + 1] = 0x28
    uwp[b * 18] = 0; uwp[b * 18 + 1] = 0x28
    for j in 2..<18 {
        gwp[b * 18 + j] = UInt8(truncatingIfNeeded: random())
        uwp[b * 18 + j] = UInt8(truncatingIfNeeded: random())
    }
}

// Compute reference on CPU
var reference = [UInt16](repeating: 0, count: n)
for col in 0..<n {
    var g_sum: Int32 = 0
    var u_sum: Int32 = 0
    for b in 0..<blocks {
        let offset = (col * blocks + b) * 18 + 2
        for lane in 0..<16 {
            let gp = gwp[offset + lane]
            let up = uwp[offset + lane]
            g_sum += x[b * 32 + lane] * (Int32(gp & 15) - 8)
            g_sum += x[b * 32 + lane + 16] * (Int32(gp >> 4) - 8)
            u_sum += x[b * 32 + lane] * (Int32(up & 15) - 8)
            u_sum += x[b * 32 + lane + 16] * (Int32(up >> 4) - 8)
        }
    }
    let g_val = Float(g_sum) / 512.0
    let u_val = Float(u_sum) / 512.0
    reference[col] = Float16(g_val * u_val).bitPattern
}

var results: [(name: String, median_us: Double, gbps: Double)] = []

for (kIdx, kName) in kernelNames.enumerated() {
    let groups = (n + 8 * 8 - 1) / (8 * 8)
    var samples: [Double] = []
    let op = output.contents().bindMemory(to: UInt16.self, capacity: n)

    for iter in 0..<14 {
        perturbCache()
        for col in 0..<n { op[col] = 0x7e00 }
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipelines[kIdx])
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(gateWeight, offset: 0, index: 1)
        enc.setBuffer(upWeight, offset: 0, index: 2)
        enc.setBuffer(output, offset: 0, index: 3)
        let params: [UInt32] = [1, UInt32(n), UInt32(k), 0]
        params.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 4) }
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        finish(cmd)

        for col in 0..<n {
            require(op[col] == reference[col], "Parity mismatch in \(kName) col \(col): \(op[col]) vs ref \(reference[col])")
        }
        let sec = cmd.gpuEndTime - cmd.gpuStartTime
        if iter >= 3 { samples.append(sec) }
    }
    samples.sort()
    let med = samples[samples.count / 2]
    let totalBytes = weightBytes * 2
    let gbps = Double(totalBytes) / med / 1e9
    results.append((name: kName, median_us: med * 1e6, gbps: gbps))
}

for r in results {
    print(String(format: "  %-36@: %7.2f µs | %6.2f GB/s (vs baseline: %+.1f%%)",
                 r.name as NSString, r.median_us, r.gbps,
                 ((results[0].median_us / r.median_us) - 1.0) * 100))
}
