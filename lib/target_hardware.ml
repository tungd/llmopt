let ( let* ) = Result.bind

module Memory_hierarchy = struct
  type t = {
    simd_lanes : int;
    sram_banks : int;
    sram_bank_width_bytes : int;
    sram_capacity_bytes : int;
    l1_cache_line_bytes : int;
    l1_cache_capacity_bytes : int;
  }

  let create ~simd_lanes ~sram_banks ~sram_bank_width_bytes ~sram_capacity_bytes
      ~l1_cache_line_bytes ~l1_cache_capacity_bytes =
    if simd_lanes <= 0 then Error "simd_lanes must be positive"
    else if sram_banks <= 0 then Error "sram_banks must be positive"
    else if sram_bank_width_bytes <= 0 then Error "sram_bank_width_bytes must be positive"
    else if sram_capacity_bytes <= 0 then Error "sram_capacity_bytes must be positive"
    else if l1_cache_line_bytes <= 0 then Error "l1_cache_line_bytes must be positive"
    else if l1_cache_capacity_bytes <= 0 then Error "l1_cache_capacity_bytes must be positive"
    else
      Ok
        {
          simd_lanes;
          sram_banks;
          sram_bank_width_bytes;
          sram_capacity_bytes;
          l1_cache_line_bytes;
          l1_cache_capacity_bytes;
        }

  let apple_silicon =
    {
      simd_lanes = 32;
      sram_banks = 32;
      sram_bank_width_bytes = 4;
      sram_capacity_bytes = 32 * 1024;
      l1_cache_line_bytes = 128;
      l1_cache_capacity_bytes = 128 * 1024;
    }
end

module Execution_profile = struct
  type t = {
    gpu_cores : int;
    max_threads_per_threadgroup : int;
    memory_bandwidth_gbps : float;
    fp16_tflops : float;
    dispatch_overhead_seconds : float;
  }

  let create ~gpu_cores ~max_threads_per_threadgroup ~memory_bandwidth_gbps
      ~fp16_tflops ~dispatch_overhead_seconds =
    if gpu_cores <= 0 then Error "gpu_cores must be positive"
    else if max_threads_per_threadgroup <= 0 then
      Error "max_threads_per_threadgroup must be positive"
    else if (not (Float.is_finite memory_bandwidth_gbps)) || memory_bandwidth_gbps <= 0.0 then
      Error "memory_bandwidth_gbps must be positive"
    else if (not (Float.is_finite fp16_tflops)) || fp16_tflops <= 0.0 then
      Error "fp16_tflops must be positive"
    else if (not (Float.is_finite dispatch_overhead_seconds)) || dispatch_overhead_seconds <= 0.0 then
      Error "dispatch_overhead_seconds must be positive"
    else
      Ok
        {
          gpu_cores;
          max_threads_per_threadgroup;
          memory_bandwidth_gbps;
          fp16_tflops;
          dispatch_overhead_seconds;
        }

  let m4_pro =
    {
      gpu_cores = 16;
      max_threads_per_threadgroup = 1024;
      memory_bandwidth_gbps = 273.0;
      fp16_tflops = 18.7;
      dispatch_overhead_seconds = 6.5e-6;
    }

  let default_apple =
    {
      gpu_cores = 8;
      max_threads_per_threadgroup = 1024;
      memory_bandwidth_gbps = 100.0;
      fp16_tflops = 10.0;
      dispatch_overhead_seconds = 7.0e-6;
    }
end

module Prefill_cost_model = struct
  type t = {
    roofline_knee_tokens : int;
    core_saturation_tokens : int;
    optimal_chunk_size : int;
    template_buckets : int list;
    predicted_chunk_latency_ms : float;
  }

  let analyze ?(max_latency_ms = 25.0) ~(target : Execution_profile.t)
      ~weight_bytes ~active_params () =
    let safe_params = max 1_000_000 active_params in
    let safe_weights = max 1_000_000 weight_bytes in
    let bw_bytes_per_sec = target.memory_bandwidth_gbps *. 1e9 in
    let peak_flops_per_sec = target.fp16_tflops *. 1e12 in
    let flops_per_token = 2.0 *. float_of_int safe_params in

    (* 1. Roofline arithmetic intensity knee: W/B = (M * F)/P *)
    let t_mem = float_of_int safe_weights /. bw_bytes_per_sec in
    let knee_float = t_mem *. peak_flops_per_sec /. flops_per_token in
    let roofline_knee_tokens = max 8 (int_of_float (Float.ceil knee_float)) in

    (* 2. Core saturation: need enough threadgroups to keep all compute units busy *)
    let core_saturation_tokens = max 32 (target.gpu_cores * 4) in

    (* 3. SLA budget constraint: M_sla = ((max_latency - dispatch) * P) / F *)
    let max_sec = max_latency_ms *. 1e-3 in
    let usable_sec = max 0.001 (max_sec -. target.dispatch_overhead_seconds) in
    let max_sla_tokens =
      int_of_float (usable_sec *. peak_flops_per_sec /. flops_per_token)
    in

    (* 4. Optimal chunk size M*: balance roofline compute intensity and SLA latency *)
    let raw_chunk = max roofline_knee_tokens core_saturation_tokens * 4 in
    let bounded_chunk =
      min (max 64 raw_chunk) (max 64 max_sla_tokens)
    in
    (* Align to multiple of 64 *)
    let optimal_chunk_size =
      let rem = bounded_chunk mod 64 in
      if rem = 0 then bounded_chunk else bounded_chunk + (64 - rem)
    in

    (* 5. Standard template buckets *)
    let candidate_buckets =
      [ 16; 64; 128; optimal_chunk_size;
        min (optimal_chunk_size * 2) 1024;
        min (optimal_chunk_size * 4) 4096 ]
    in
    let template_buckets =
      candidate_buckets
      |> List.sort_uniq Int.compare
      |> List.filter (fun b -> b > 0)
    in

    (* 6. Predicted latency for optimal chunk *)
    let t_compute =
      (float_of_int optimal_chunk_size *. flops_per_token) /. peak_flops_per_sec
    in
    let predicted_chunk_latency_ms =
      (Float.max t_mem t_compute +. target.dispatch_overhead_seconds) *. 1000.0
    in

    {
      roofline_knee_tokens;
      core_saturation_tokens;
      optimal_chunk_size;
      template_buckets;
      predicted_chunk_latency_ms;
    }

  let to_json (m : t) : Yojson.Basic.t =
    `Assoc
      [
        ("roofline_knee_tokens", `Int m.roofline_knee_tokens);
        ("core_saturation_tokens", `Int m.core_saturation_tokens);
        ("optimal_chunk_size", `Int m.optimal_chunk_size);
        ("template_buckets", `List (List.map (fun b -> `Int b) m.template_buckets));
        ("predicted_chunk_latency_ms", `Float m.predicted_chunk_latency_ms);
      ]

  let of_json (json : Yojson.Basic.t) : (t, string) result =
    let open Yojson.Basic.Util in
    try
      let roofline_knee_tokens = json |> member "roofline_knee_tokens" |> to_int in
      let core_saturation_tokens = json |> member "core_saturation_tokens" |> to_int in
      let optimal_chunk_size = json |> member "optimal_chunk_size" |> to_int in
      let template_buckets =
        json |> member "template_buckets" |> to_list |> List.map to_int
      in
      let predicted_chunk_latency_ms =
        json |> member "predicted_chunk_latency_ms" |> to_float
      in
      Ok
        {
          roofline_knee_tokens;
          core_saturation_tokens;
          optimal_chunk_size;
          template_buckets;
          predicted_chunk_latency_ms;
        }
    with Type_error (msg, _) -> Error ("Prefill_cost_model JSON error: " ^ msg)
end

type t = {
  device_name : string;
  memory : Memory_hierarchy.t;
  execution : Execution_profile.t;
}

let create ~device_name ~memory ~execution =
  if String.trim device_name = "" then Error "device_name cannot be empty"
  else Ok { device_name; memory; execution }

let default =
  {
    device_name = "Apple M4 Pro";
    memory = Memory_hierarchy.apple_silicon;
    execution = Execution_profile.m4_pro;
  }

let discover () =
  let device_name =
    try
      let ic = Unix.open_process_in "sysctl -n machdep.cpu.brand_string" in
      let name = input_line ic in
      close_in ic;
      String.trim name
    with _ -> "Apple M4 Pro"
  in
  let execution =
    if String.contains device_name '4' && String.contains device_name 'P' then
      Execution_profile.m4_pro
    else Execution_profile.default_apple
  in
  {
    device_name;
    memory = Memory_hierarchy.apple_silicon;
    execution;
  }

let bank_conflict_degree (mem : Memory_hierarchy.t) ~element_bytes ~stride_elements =
  if stride_elements <= 0 || element_bytes <= 0 then 1
  else
    let counts = Array.make mem.sram_banks 0 in
    for lane = 0 to mem.simd_lanes - 1 do
      let byte_offset = lane * stride_elements * element_bytes in
      let bank_id = (byte_offset / mem.sram_bank_width_bytes) mod mem.sram_banks in
      counts.(bank_id) <- counts.(bank_id) + 1
    done;
    Array.fold_left max 1 counts

let should_fuse_sram_reduction (hw : t) ~elements ~element_bytes ~threadgroups ~barriers =
  let footprint = elements * element_bytes in
  if footprint > hw.memory.sram_capacity_bytes then false
  else
    (* Analytical comparison:
       1. Fusing reduction into all threadgroups repeats footprint read/compute N_tg times.
       2. Keeping separate reduction kernel: 1 TG writes footprint to L1 cache,
          which broadcasts to all consumer threadgroups without redundant work. *)
    let sram_bandwidth_gbps = float_of_int hw.execution.gpu_cores *. 128.0 in
    let l1_bandwidth_gbps = float_of_int hw.execution.gpu_cores *. 64.0 in
    let conflict =
      float_of_int
        (bank_conflict_degree hw.memory ~element_bytes ~stride_elements:1)
    in
    let bytes_sram = float_of_int (elements * element_bytes * threadgroups) in
    let t_sram =
      (bytes_sram /. (sram_bandwidth_gbps *. 1e9) *. conflict)
      +. (float_of_int barriers *. 1.0e-7 *. float_of_int threadgroups)
    in
    let bytes_separate = float_of_int (elements * element_bytes * 2) in
    let t_separate =
      hw.execution.dispatch_overhead_seconds
      +. (bytes_separate /. (l1_bandwidth_gbps *. 1e9))
    in
    t_sram < t_separate

let to_json (hw : t) : Yojson.Basic.t =
  `Assoc
    [
      ("device_name", `String hw.device_name);
      ( "memory",
        `Assoc
          [
            ("simd_lanes", `Int hw.memory.simd_lanes);
            ("sram_banks", `Int hw.memory.sram_banks);
            ("sram_bank_width_bytes", `Int hw.memory.sram_bank_width_bytes);
            ("sram_capacity_bytes", `Int hw.memory.sram_capacity_bytes);
            ("l1_cache_line_bytes", `Int hw.memory.l1_cache_line_bytes);
            ("l1_cache_capacity_bytes", `Int hw.memory.l1_cache_capacity_bytes);
          ] );
      ( "execution",
        `Assoc
          [
            ("gpu_cores", `Int hw.execution.gpu_cores);
            ( "max_threads_per_threadgroup",
              `Int hw.execution.max_threads_per_threadgroup );
            ( "memory_bandwidth_gbps",
              `Float hw.execution.memory_bandwidth_gbps );
            ("fp16_tflops", `Float hw.execution.fp16_tflops);
            ( "dispatch_overhead_seconds",
              `Float hw.execution.dispatch_overhead_seconds );
          ] );
    ]

let of_json (json : Yojson.Basic.t) : (t, string) result =
  let open Yojson.Basic.Util in
  try
    let device_name = json |> member "device_name" |> to_string in
    let mem = json |> member "memory" in
    let simd_lanes = mem |> member "simd_lanes" |> to_int in
    let sram_banks = mem |> member "sram_banks" |> to_int in
    let sram_bank_width_bytes = mem |> member "sram_bank_width_bytes" |> to_int in
    let sram_capacity_bytes = mem |> member "sram_capacity_bytes" |> to_int in
    let l1_cache_line_bytes = mem |> member "l1_cache_line_bytes" |> to_int in
    let l1_cache_capacity_bytes = mem |> member "l1_cache_capacity_bytes" |> to_int in
    let* memory =
      Memory_hierarchy.create ~simd_lanes ~sram_banks ~sram_bank_width_bytes
        ~sram_capacity_bytes ~l1_cache_line_bytes ~l1_cache_capacity_bytes
    in
    let exec = json |> member "execution" in
    let gpu_cores = exec |> member "gpu_cores" |> to_int in
    let max_threads_per_threadgroup =
      exec |> member "max_threads_per_threadgroup" |> to_int
    in
    let memory_bandwidth_gbps =
      exec |> member "memory_bandwidth_gbps" |> to_float
    in
    let fp16_tflops = exec |> member "fp16_tflops" |> to_float in
    let dispatch_overhead_seconds =
      exec |> member "dispatch_overhead_seconds" |> to_float
    in
    let* execution =
      Execution_profile.create ~gpu_cores ~max_threads_per_threadgroup
        ~memory_bandwidth_gbps ~fp16_tflops ~dispatch_overhead_seconds
    in
    create ~device_name ~memory ~execution
  with Type_error (msg, _) -> Error ("JSON decoding error: " ^ msg)

let to_string (hw : t) : string =
  Printf.sprintf
    "Target_hardware(device=%s, cores=%d, bw=%.1f GB/s, simd=%d, banks=%d, sram=%d KB, l1_line=%d B)"
    hw.device_name hw.execution.gpu_cores hw.execution.memory_bandwidth_gbps
    hw.memory.simd_lanes hw.memory.sram_banks
    (hw.memory.sram_capacity_bytes / 1024)
    hw.memory.l1_cache_line_bytes
