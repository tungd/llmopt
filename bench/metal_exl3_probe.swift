// Execute OCaml-generated EXL3 stages. The fixture's independent upstream
// oracle supplies all expected values; this runner contains no weight decoder.
import Foundation
import Metal

func require(_ value: Bool, _ message: String) {
    if !value { fputs(message + "\n", stderr); exit(1) }
}
func finish(_ command: MTLCommandBuffer) -> Double {
    command.commit()
    command.waitUntilCompleted()
    require(command.status == .completed, "GPU failed: \(String(describing: command.error))")
    let seconds = command.gpuEndTime - command.gpuStartTime
    require(seconds > 0, "GPU timestamp unavailable")
    return seconds * 1e6
}
func median(_ values: [Double]) -> Double { values.sorted()[values.count / 2] }
require(CommandLine.arguments.count == 3, "usage: metal-exl3-probe probe.json result.json")
let manifestURL = URL(fileURLWithPath: CommandLine.arguments[1])
let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as! [String: Any]
let archive = try Data(contentsOf: URL(fileURLWithPath: manifest["archive"] as! String))
let tensors = manifest["tensors"] as! [String: [String: Int]]
let k = manifest["k"] as! Int, n = manifest["n"] as! Int, maxRows = manifest["rows"] as! Int
let variants = manifest["variants"] as! [[String: Any]]
guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    fatalError("Metal unavailable")
}
func alloc(_ bytes: Int) -> MTLBuffer {
    guard let b = device.makeBuffer(length: bytes, options: .storageModeShared) else { fatalError("buffer allocation failed") }
    return b
}
func payload(_ name: String) -> MTLBuffer {
    let t = tensors[name]!, offset = t["offset"]!, size = t["bytes"]!
    require(offset >= 0 && size > 0 && offset <= archive.count - size, "invalid fixture range")
    let b = alloc(size)
    archive.withUnsafeBytes { raw in b.contents().copyMemory(from: raw.baseAddress!.advanced(by: offset), byteCount: size) }
    return b
}
let packed = payload("weight.trellis"), suh = payload("weight.suh"), svh = payload("weight.svh")
let input = payload("input"), expected = payload("expected"), decoded = payload("decoded"), codebook = payload("codebook")
let transformed = alloc(maxRows * k * 2), product = alloc(maxRows * n * 4), output = alloc(maxRows * n * 2)
let reference = expected.contents().bindMemory(to: Float.self, capacity: maxRows * n)
let op = output.contents().bindMemory(to: Float16.self, capacity: maxRows * n)
var results = [[String: Any]]()
for variant in variants {
    try autoreleasepool {
        let name = variant["name"] as! String
        let sourceURL = manifestURL.deletingLastPathComponent().appendingPathComponent(variant["source"] as! String)
        let source = try String(contentsOf: sourceURL, encoding: .utf8)
        let options = MTLCompileOptions()
        options.fastMathEnabled = false
        let start = ProcessInfo.processInfo.systemUptime
        let library = try device.makeLibrary(source: source, options: options)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { fatalError("missing generated kernel \(name)") }
            return try device.makeComputePipelineState(function: f)
        }
        let pre = try pipeline("exl3_input"), dot = try pipeline("exl3_contract"), post = try pipeline("exl3_output")
        let unpack = try pipeline("exl3_unpack"), cb = try pipeline("exl3_codebook")
        let compileSeconds = ProcessInfo.processInfo.systemUptime - start
        let threads = variant["threads"] as! Int, rows = variant["rows"] as! Int, columns = variant["columns"] as! Int
        require(dot.threadExecutionWidth == 32 && dot.maxTotalThreadsPerThreadgroup >= threads, "unsupported device geometry")
        let decodedGPU = alloc(k * n * 2), codebookGPU = alloc(65536 * 2)
        for (p, buffers, count) in [(unpack, [packed, decodedGPU], k * n), (cb, [codebookGPU], 65536)] {
            let command = queue.makeCommandBuffer()!, e = command.makeComputeCommandEncoder()!
            e.setComputePipelineState(p)
            for (i,b) in buffers.enumerated() { e.setBuffer(b, offset: 0, index: i) }
            e.dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            e.endEncoding()
            _ = finish(command)
        }
        require(memcmp(decodedGPU.contents(), decoded.contents(), k * n * 2) == 0, "packed decode differs from upstream: \(name)")
        require(memcmp(codebookGPU.contents(), codebook.contents(), 65536 * 2) == 0, "mul1 codebook differs from upstream: \(name)")
        for m in Array(Set([1, min(16, maxRows), maxRows])).sorted() {
            let stages: [(MTLComputePipelineState, [MTLBuffer], MTLSize, Int)] = [
                (pre, [input, suh, transformed], MTLSize(width: k / 128, height: m, depth: 1), 128),
                (dot, [transformed, packed, product], MTLSize(width: (n + columns - 1) / columns, height: (m + rows - 1) / rows, depth: 1), threads),
                (post, [product, svh, output], MTLSize(width: n / 128, height: m, depth: 1), 128)
            ]
            func run(_ selected: [Int]) -> Double {
                let command = queue.makeCommandBuffer()!
                for index in selected {
                    let (pipeline, buffers, grid, group) = stages[index]
                    // Separate encoders provide tracked-resource ordering between stages.
                    let e = command.makeComputeCommandEncoder()!
                    e.setComputePipelineState(pipeline)
                    for (i,b) in buffers.enumerated() { e.setBuffer(b, offset: 0, index: i) }
                    var count = UInt32(m)
                    e.setBytes(&count, length: 4, index: 3)
                    e.dispatchThreadgroups(grid, threadsPerThreadgroup: MTLSize(width: group, height: 1, depth: 1))
                    e.endEncoding()
                }
                return finish(command)
            }
            var worst = 0.0, squared = 0.0
            func validate() {
                worst = 0; squared = 0
                for i in 0..<(m * n) {
                    var ref = Double(reference[i])
                    if variant["silu"] as? Bool == true { ref = ref / (1 + exp(-ref)) }
                    let value = Double(op[i]), error = abs(value - ref)
                    require(value.isFinite && error <= 0.001 + 0.002 * abs(ref),
                            "linear parity failed \(name) m=\(m) i=\(i) actual=\(value) expected=\(ref)")
                    worst = max(worst, error); squared += error * error
                }
            }
            for _ in 0..<3 { _ = run([0,1,2]); validate() }
            var total = [Double](), stageTimes = [[Double]](repeating: [], count: 3)
            for _ in 0..<7 {
                for i in 0..<(m * n) { op[i] = .nan }
                total.append(run([0,1,2])); validate()
                for stage in 0..<3 { stageTimes[stage].append(run([stage])) }
                validate()
            }
            let result: [String: Any] = [
                "schedule": name, "m": m, "n": n, "k": k, "bits": manifest["bits"]!,
                "compile_seconds": compileSeconds, "threads": threads,
                "gpu_total_us": total, "gpu_total_median_us": median(total),
                "gpu_input_transform_median_us": median(stageTimes[0]),
                "gpu_contraction_median_us": median(stageTimes[1]),
                "gpu_output_transform_median_us": median(stageTimes[2]),
                "max_abs_error": worst, "rmse": sqrt(squared / Double(m * n)),
                "codebook_exact_values": 65536, "packed_decode_exact_values": k * n,
                "packed_weight_bytes": packed.length, "parity": "passed"
            ]
            results.append(result)
            print("\(name) m=\(m) n=\(n) k=\(k) total_us=\(median(total)) contraction_us=\(median(stageTimes[1])) max_abs=\(worst)")
        }
    }
}
let report: [String: Any] = [
    "device": device.name, "fixture": manifest["archive"]!, "results": results,
    "warmups": 3, "samples": 7, "fast_math": false,
    "timing_scope": "GPU command timestamps; total includes three stages and encoder transitions; excludes compilation, host submission, and validation",
    "workload": "synthetic, repeatedly reused buffers; not a DRAM bandwidth or full-model throughput measurement",
    "parity_tolerance": "abs(error) <= 0.001 + 0.002 * abs(stage-aware float reference); packed decode and all codebook values exact"
]
try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: CommandLine.arguments[2]))
