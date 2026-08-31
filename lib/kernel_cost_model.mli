(** Structural dispatch geometry selected from a measured target profile.

    Latency and bandwidth are measured by [llmopt-hardware-probe] and carried
    by [Target_hardware]. This module deliberately contains no learned model or
    generated predictor. *)

module Device : sig
  type t

  val create :
    ?dispatch_overhead_seconds:float ->
    gpu_core_count:int ->
    memory_bandwidth_gbps:float ->
    unit ->
    (t, string) result

  val default : t
  val gpu_core_count : t -> int
  val memory_bandwidth_gbps : t -> float
  val dispatch_overhead_seconds : t -> float
end

module Megakernel : sig
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

  val select_qkv_rope_tile :
    device:Device.t -> total_pairs:int -> qkv_rope_tile

  val select_lm_head_tile :
    device:Device.t -> vocab_size:int -> lm_head_tile
end
