import Metal
import Foundation

guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { fatalError("Metal unavailable") }
let source = """
#include <metal_stdlib>
using namespace metal;
kernel void stream_read(device const uint4* src [[buffer(0)]],
                        device uint* out [[buffer(1)]],
                        constant uint& vectors [[buffer(2)]],
                        constant uint& stride [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
  uint4 acc = uint4(0);
  for (uint idx = gid; idx < vectors; idx += stride) acc += src[idx];
  out[gid] = acc.x + acc.y + acc.z + acc.w;
}
"""
let library = try device.makeLibrary(source: source, options: nil)
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "stream_read")!)
let bytes = 256 * 1024 * 1024
let src = device.makeBuffer(length: bytes, options: .storageModeShared)!
let values = src.contents().bindMemory(to: UInt32.self, capacity: bytes / 4)
for i in 0..<(bytes / 4) { values[i] = 1 }
let out = device.makeBuffer(length: 4096 * 256 * 4, options: .storageModeShared)!
print("device=\(device.name) bytes=\(bytes) group_threads=256 simd_width=\(pipeline.threadExecutionWidth)")
print("All grids read identical bytes; coalesced uint4 grid-stride loads, checksum required. 2 warmups + 7 samples. GPU timestamps include one dispatch, no upload/readback.")
for groups in [1, 4, 16, 64, 256, 1024, 4096] {
    let threads = groups * 256
    var vectors = UInt32(bytes / 16)
    var stride = UInt32(threads)
    var samples: [Double] = []
    for iteration in 0..<9 {
        let cmd = queue.makeCommandBuffer()!
        let enc = cmd.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(src, offset: 0, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.setBytes(&vectors, length: 4, index: 2)
        enc.setBytes(&stride, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        guard cmd.status == .completed else { fatalError("dispatch failed: \(String(describing: cmd.error))") }
        let result = out.contents().bindMemory(to: UInt32.self, capacity: threads)
        var sum: UInt64 = 0
        for i in 0..<threads { sum += UInt64(result[i]) }
        guard sum == UInt64(bytes / 4) else { fatalError("checksum \(sum)") }
        let seconds = cmd.gpuEndTime - cmd.gpuStartTime
        guard seconds > 0 else { fatalError("missing GPU timestamps") }
        if iteration >= 2 { samples.append(seconds) }
    }
    samples.sort()
    let median = samples[3]
    let gbps = Double(bytes) / median / 1e9
    print(String(format: "groups=%4d simdgroups=%5d median_ms=%8.3f GBps=%7.2f pct_of_273=%5.1f min_ms=%.3f max_ms=%.3f checksum=PASS", groups, groups * 8, median * 1000, gbps, gbps / 273 * 100, samples.first! * 1000, samples.last! * 1000))
}
