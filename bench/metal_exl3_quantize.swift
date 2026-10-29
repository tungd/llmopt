// Persistent, bounded converter worker. One request/command buffer at a time.
import Foundation
import Metal
func check(_ b: Bool, _ text: String) { if !b { fputs(text + "\n", stderr); exit(1) } }
check(CommandLine.arguments.count == 2, "usage: metal-exl3-quantize source.metal")
guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { fatalError("Metal unavailable") }
let options = MTLCompileOptions(); options.fastMathEnabled = false
let library = try device.makeLibrary(source: String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8), options: options)
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "exl3_quantize")!)
let unpack = try device.makeComputePipelineState(function: library.makeFunction(name: "exl3_unpack")!)
while let line = readLine() {
  try autoreleasepool {
    let request = try JSONSerialization.jsonObject(with: Data(line.utf8)) as! [String: Any]
    let k = request["k"] as! Int, n = request["n"] as! Int
    check(k > 0 && n > 0 && k % 128 == 0 && n % 128 == 0 && k * n <= 8 * 1024 * 1024, "invalid or unbounded conversion dimensions")
    let data = try Data(contentsOf: URL(fileURLWithPath: request["input"] as! String))
    check(data.count == k * n * 4, "input size mismatch")
    let input = data.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: data.count, options: .storageModeShared)! }
    let output = device.makeBuffer(length: k * n / 2, options: .storageModeShared)!
    let decoded = device.makeBuffer(length: k * n * 2, options: .storageModeShared)!
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: 0, index: 0); encoder.setBuffer(output, offset: 0, index: 1)
    var dims = SIMD2<UInt32>(UInt32(k), UInt32(n)); encoder.setBytes(&dims, length: 8, index: 2)
    encoder.dispatchThreadgroups(MTLSize(width: k * n / 256, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    encoder.endEncoding()
    let e = command.makeComputeCommandEncoder()!; e.setComputePipelineState(unpack)
    e.setBuffer(output, offset: 0, index: 0); e.setBuffer(decoded, offset: 0, index: 1)
    let params: [UInt32] = [1, UInt32(n), UInt32(k), 4, 0, 0, 1, 0]
    params.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: 2) }
    e.dispatchThreads(MTLSize(width: k * n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)); e.endEncoding()
    command.commit(); command.waitUntilCompleted()
    check(command.status == .completed, "GPU conversion failed: \(String(describing: command.error))")
    try Data(bytes: output.contents(), count: output.length).write(to: URL(fileURLWithPath: request["output"] as! String))
    let x = input.contents().bindMemory(to: Float.self, capacity: k * n)
    let y = decoded.contents().bindMemory(to: Float16.self, capacity: k * n)
    var sse = 0.0, energy = 0.0
    for i in 0..<(k * n) { let d = Double(x[i]) - Double(y[i]); sse += d*d; energy += Double(x[i])*Double(x[i]) }
    if let path = request["decoded"] as? String { try Data(bytes: decoded.contents(), count: decoded.length).write(to: URL(fileURLWithPath: path)) }
    let result: [String: Any] = ["status": "ok", "sse": sse, "energy": energy, "gpu_seconds": command.gpuEndTime-command.gpuStartTime]
    print(String(data: try JSONSerialization.data(withJSONObject: result), encoding: .utf8)!); fflush(stdout)
  }
}
