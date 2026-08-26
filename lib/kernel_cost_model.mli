type mode = Gemm | Gemv_single | Gemv_pair

type selection =
  | Gemm_16x16x64
  | Gemm_32x8x64
  | Gemm_8x32x64
  | Gemm_32x32x64
  | Gemm_16x16x128
  | Gemm_32x8x128
  | Gemm_8x32x128
  | Gemm_32x32x128
  | Gemv_single
  | Gemv_pair

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

val mode : selection -> mode
val tile : selection -> int * int * int
val threadgroup : selection -> int * int * int
val kernel_name : selection -> string

val select_optimal_tile :
  m:int -> n:int -> k:int -> device:Device.t -> (selection, string) result

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
