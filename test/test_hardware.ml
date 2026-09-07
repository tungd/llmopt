open Test_support

(* Test_hardware: Target_hardware discovery, bank-conflict and JSON round-trip
   tests, plus the hardware-aware prefill cost model. *)

let run () =

  (* Target_hardware discovery, bank conflicts, and JSON round-trip tests *)
  let target = Target_hardware.discover () in
  expect (String.length target.device_name > 0) "target hardware device name is non-empty";
  expect (target.memory.simd_lanes = 32) "Apple Silicon SIMD lanes is 32";
  expect (target.memory.sram_banks = 32) "Apple Silicon SRAM banks is 32";
  expect (target.memory.sram_bank_width_bytes = 4) "Apple Silicon SRAM bank width is 4 bytes";
  expect (target.memory.sram_capacity_bytes = 32768) "Apple Silicon SRAM capacity is 32 KB";
  expect (target.memory.l1_cache_line_bytes = 128) "Apple Silicon L1 cache line is 128 bytes";

  (* Microarchitectural bank conflict checks *)
  expect
    (Target_hardware.bank_conflict_degree target.memory ~element_bytes:2 ~stride_elements:1 = 2)
    "contiguous FP16 in 4-byte banks has 2-way bank conflict";
  expect
    (Target_hardware.bank_conflict_degree target.memory ~element_bytes:4 ~stride_elements:1 = 1)
    "contiguous FP32/half2 in 4-byte banks is conflict-free (1-way)";
  expect
    (Target_hardware.bank_conflict_degree target.memory ~element_bytes:2 ~stride_elements:2 = 1)
    "stride-2 FP16 in 4-byte banks is conflict-free (1-way)";

  (* Analytical SRAM fusion cost model *)
  expect
    (Target_hardware.should_fuse_sram_reduction target ~elements:1024 ~element_bytes:2 ~threadgroups:1 ~barriers:1)
    "SRAM fusion is profitable for single-threadgroup dispatch";
  expect
    (not (Target_hardware.should_fuse_sram_reduction target ~elements:1024 ~element_bytes:2 ~threadgroups:576 ~barriers:3))
    "SRAM fusion is rejected for 576 threadgroups due to redundant compute and barrier overhead";

  (* Target hardware JSON round-trip *)
  let target_json = Target_hardware.to_json target in
  let target_roundtrip = expect_ok (Target_hardware.of_json target_json) in
  expect (target_roundtrip.device_name = target.device_name) "target JSON round-trip preserves device name";
  expect (target_roundtrip.memory.simd_lanes = 32) "target JSON round-trip preserves simd_lanes";
  expect (target_roundtrip.memory.sram_banks = 32) "target JSON round-trip preserves sram_banks";
  let calibrated =
    Target_hardware.calibrate ~measured_bandwidth_gbps:174.5
      ~measured_dispatch_overhead_seconds:3.6e-6 target
  in
  expect
    (calibrated.execution.memory_bandwidth_gbps = 174.5
    && calibrated.execution.dispatch_overhead_seconds = 3.6e-6)
    "hardware calibration installs measured bandwidth and dispatch cost";
  let profile_json =
    `Assoc
      [
        "schema_version", `Int 1;
        "backend", `String "metal";
        "target", Target_hardware.to_json calibrated;
      ]
  in
  let profiled =
    Target_hardware.of_profile_json profile_json |> expect_ok
  in
  expect
    (profiled.execution.memory_bandwidth_gbps = 174.5
    && profiled.execution.dispatch_overhead_seconds = 3.6e-6)
    "hardware profile envelope round-trips measured device costs";

  (* Prefill batch/chunk size cost model tests *)
  let prefill_cm_350m =
    Target_hardware.Prefill_cost_model.analyze ~target:target.execution
      ~weight_bytes:175_000_000 ~active_params:350_000_000 ()
  in
  expect (prefill_cm_350m.roofline_knee_tokens >= 8 && prefill_cm_350m.roofline_knee_tokens <= 64)
    "350M roofline knee is within expected range [8, 64]";
  expect (prefill_cm_350m.core_saturation_tokens >= 32)
    "core saturation tokens is >= 32";
  expect (prefill_cm_350m.optimal_chunk_size mod 64 = 0)
    "optimal chunk size is 64-token aligned";
  expect (prefill_cm_350m.optimal_chunk_size >= 64 && prefill_cm_350m.optimal_chunk_size <= 512)
    "optimal chunk size is in serving range [64, 512]";
  expect (List.mem 64 prefill_cm_350m.template_buckets)
    "template buckets include 64-token interactive bucket";
  expect (List.mem prefill_cm_350m.optimal_chunk_size prefill_cm_350m.template_buckets)
    "template buckets include optimal chunk size";
  expect (prefill_cm_350m.predicted_chunk_latency_ms > 0.0 && prefill_cm_350m.predicted_chunk_latency_ms < 50.0)
    "predicted chunk latency is within interactive budget (< 50ms)";

  (* Prefill cost model JSON round-trip *)
  let pcm_json = Target_hardware.Prefill_cost_model.to_json prefill_cm_350m in
  let pcm_roundtrip = expect_ok (Target_hardware.Prefill_cost_model.of_json pcm_json) in
  expect (pcm_roundtrip.roofline_knee_tokens = prefill_cm_350m.roofline_knee_tokens)
    "prefill cost model JSON round-trip preserves roofline knee";
  expect (pcm_roundtrip.optimal_chunk_size = prefill_cm_350m.optimal_chunk_size)
    "prefill cost model JSON round-trip preserves optimal chunk size";
  expect (pcm_roundtrip.template_buckets = prefill_cm_350m.template_buckets)
    "prefill cost model JSON round-trip preserves template buckets"
