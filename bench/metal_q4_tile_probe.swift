// Benchmark existing kernels from a supplied package's kernel.metal; never rewrite MSL.
// swiftc -O bench/metal_q4_tile_probe.swift -o /tmp/llmopt-q4-tile-probe
// /tmp/llmopt-q4-tile-probe /path/to/decode/kernel.metal
import Foundation
import Metal

func require(_ condition: Bool, _ message: String) {
    if !condition { fatalError(message) }
}

require(CommandLine.arguments.count == 2, "usage: metal-q4-tile-probe kernel.metal")
guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    fatalError("Metal unavailable")
}
let source = try String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8)
let library = try device.makeLibrary(source: source, options: nil)
let tiles = [4, 8, 16]
let pipelines = try tiles.map { tile -> MTLComputePipelineState in
    guard let function = library.makeFunction(name: "llmopt_q4_0_linear_f16_x\(tile)") else {
        fatalError("missing x\(tile) kernel")
    }
    let pipeline = try device.makeComputePipelineState(function: function)
    require(pipeline.threadExecutionWidth == 32 && pipeline.maxTotalThreadsPerThreadgroup >= 256,
            "probe requires 32-lane SIMD and 256-thread groups")
    return pipeline
}
func buffer(_ length: Int) -> MTLBuffer {
    guard let result = device.makeBuffer(length: length, options: .storageModeShared) else {
        fatalError("buffer allocation failed: \(length)")
    }
    return result
}
func finish(_ command: MTLCommandBuffer) {
    command.commit()
    command.waitUntilCompleted()
    require(command.status == .completed, "GPU command failed: \(String(describing: command.error))")
}

// A large write before each dispatch reduces repeated-weight cache residency.
// This is a cache perturbation, not proof that every tested load reached DRAM.
let eviction = buffer(256 * 1024 * 1024)
func perturbCache() {
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeBlitCommandEncoder()!
    encoder.fill(buffer: eviction, range: 0..<eviction.length, value: 0xa5)
    encoder.endEncoding()
    finish(command)
}
print("device=\(device.name) source=\(CommandLine.arguments[1])")
print("m=1 threads_per_group=256 warmups=3 samples=11 tile_order=rotating")
print("GPU timestamps: one dispatch, excludes CPU validation and preceding 256MiB blit-fill cache perturbation.")
print("Synthetic Q4_0: dyadic x and scale; full CPU integer-dot oracle; every timed output checked bit-exact.")
print("GBps = packed weight bytes / GPU seconds, NOT measured DRAM traffic. No full-model speedup claim.")

// First shape covers partial x8/x16 column tiles and the x16 reduction remainder.
// Others are the attention-output, FFN gate/up, and FFN-down shapes in exp-0173.
for (n, k) in [(68, 96), (3840, 4096), (15360, 3840), (3840, 15360)] {
    autoreleasepool {
        let blocks = k / 32
        let weightBytes = n * blocks * 18
        let weight = buffer(weightBytes)
        let input = buffer(k * 2)
        let bias = buffer(n * 2)
        let output = buffer(n * 2)
        let xp = input.contents().bindMemory(to: Float16.self, capacity: k)
        let wp = weight.contents().bindMemory(to: UInt8.self, capacity: weightBytes)
        var x = [Int32](repeating: 0, count: k)
        var rng: UInt32 = 0x12345678
        func random() -> UInt32 {
            rng ^= rng << 13
            rng ^= rng >> 17
            rng ^= rng << 5
            return rng
        }
        for i in 0..<k {
            x[i] = Int32(random() % 17) - 8
            xp[i] = Float16(Float(x[i]) / 16)
        }
        for b in 0..<(n * blocks) {
            wp[b * 18] = 0
            wp[b * 18 + 1] = 0x28 // little-endian half scale 1/32
            for j in 2..<18 { wp[b * 18 + j] = UInt8(truncatingIfNeeded: random()) }
        }
        var reference = [UInt16](repeating: 0, count: n)
        for col in 0..<n {
            var sum: Int32 = 0
            for b in 0..<blocks {
                let offset = (col * blocks + b) * 18 + 2
                for lane in 0..<16 {
                    let packed = wp[offset + lane]
                    sum += x[b * 32 + lane] * (Int32(packed & 15) - 8)
                    sum += x[b * 32 + lane + 16] * (Int32(packed >> 4) - 8)
                }
            }
            reference[col] = Float16(Float(sum) / 512).bitPattern
        }
        var samples = [[Double]](repeating: [], count: tiles.count)
        for iteration in 0..<14 {
            for position in tiles.indices {
                let index = (position + iteration) % tiles.count
                let tile = tiles[index]
                let groups = (n + tile * 8 - 1) / (tile * 8)
                perturbCache()
                let op = output.contents().bindMemory(to: UInt16.self, capacity: n)
                for col in 0..<n { op[col] = 0x7e00 } // NaN sentinel catches unwritten outputs
                let command = queue.makeCommandBuffer()!
                let encoder = command.makeComputeCommandEncoder()!
                encoder.setComputePipelineState(pipelines[index])
                for (slot, value) in [input, weight, bias, output].enumerated() {
                    encoder.setBuffer(value, offset: 0, index: slot)
                }
                let params: [UInt32] = [1, UInt32(n), UInt32(k), 0]
                params.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 4) }
                encoder.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                             threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                encoder.endEncoding()
                finish(command)
                for col in 0..<n {
                    require(op[col] == reference[col], "parity failed x\(tile) n=\(n) k=\(k) col=\(col): \(op[col]) != \(reference[col])")
                }
                let seconds = command.gpuEndTime - command.gpuStartTime
                require(seconds > 0, "missing GPU timestamps")
                if iteration >= 3 { samples[index].append(seconds) }
            }
        }
        for index in tiles.indices {
            let sorted = samples[index].sorted()
            let median = sorted[sorted.count / 2]
            let groups = (n + tiles[index] * 8 - 1) / (tiles[index] * 8)
            print(String(format: "n=%5d k=%5d tile=x%2d groups=%3d weight_bytes=%8d median_us=%8.2f min_us=%8.2f max_us=%8.2f GBps=%7.2f parity=PASS(%d/%d)",
                         n, k, tiles[index], groups, weightBytes, median * 1e6,
                         sorted.first! * 1e6, sorted.last! * 1e6, Double(weightBytes) / median / 1e9, n, n))
        }
    }
}
