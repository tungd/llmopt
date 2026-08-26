module Device : sig
  type t

  val create :
    gpu_core_count:int -> memory_bandwidth_gbps:float -> (t, string) result

  val default : t
  val gpu_core_count : t -> int
  val memory_bandwidth_gbps : t -> float
end

val predict_latency :
  float ->
  float ->
  float ->
  float ->
  float ->
  float ->
  float ->
  float ->
  float ->
  float -> float

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
