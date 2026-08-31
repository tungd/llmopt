module Device = struct
  type t = {
    gpu_core_count : int;
    memory_bandwidth_gbps : float;
    dispatch_overhead_seconds : float;
  }

  let create ?(dispatch_overhead_seconds = 7.0e-6) ~gpu_core_count
      ~memory_bandwidth_gbps () =
    if gpu_core_count <= 0 then Error "GPU core count must be positive"
    else if
      (not (Float.is_finite memory_bandwidth_gbps))
      || memory_bandwidth_gbps <= 0.0
    then Error "memory bandwidth must be finite and positive"
    else if
      (not (Float.is_finite dispatch_overhead_seconds))
      || dispatch_overhead_seconds <= 0.0
    then Error "dispatch overhead must be finite and positive"
    else
      Ok
        { gpu_core_count; memory_bandwidth_gbps; dispatch_overhead_seconds }

  let default =
    {
      gpu_core_count = 16;
      memory_bandwidth_gbps = 273.0;
      dispatch_overhead_seconds = 6.5e-6;
    }
  let gpu_core_count device = device.gpu_core_count
  let memory_bandwidth_gbps device = device.memory_bandwidth_gbps
  let dispatch_overhead_seconds device = device.dispatch_overhead_seconds
end

module Megakernel = struct
  type qkv_rope_tile = {
    threads_per_threadgroup : int;
    simdgroups_per_threadgroup : int;
    pairs_per_threadgroup : int;
    grid_threadgroups : int;
  }

  type lm_head_tile = {
    stage1_threadgroups : int;
    stage1_threads : int;
    columns_per_thread : int;
  }

  let select_qkv_rope_tile ~device ~total_pairs =
    let simdgroups = if Device.gpu_core_count device >= 16 then 4 else 8 in
    let pairs_per_threadgroup = simdgroups in
    {
      threads_per_threadgroup = simdgroups * 32;
      simdgroups_per_threadgroup = simdgroups;
      pairs_per_threadgroup;
      grid_threadgroups =
        (total_pairs + pairs_per_threadgroup - 1) / pairs_per_threadgroup;
    }

  let select_lm_head_tile ~device ~vocab_size =
    let stage1_threads = 256 in
    let stage1_threadgroups =
      Device.gpu_core_count device * 16 |> max 64 |> min 256
    in
    {
      stage1_threadgroups;
      stage1_threads;
      columns_per_thread =
        (vocab_size + (stage1_threadgroups * stage1_threads) - 1)
        / (stage1_threadgroups * stage1_threads);
    }
end
