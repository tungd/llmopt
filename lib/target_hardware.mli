(** Hardware microarchitectural discovery and target profiling.

    Provides exact execution geometry, memory hierarchy parameters, and
    microarchitectural cost formulas (SIMD lanes, SRAM banks, bank conflicts,
    L1 line sizes, dispatch overhead) for hardware-aware AOT compilation. *)

module Memory_hierarchy : sig
  type t = {
    simd_lanes : int;
    sram_banks : int;
    sram_bank_width_bytes : int;
    sram_capacity_bytes : int;
    l1_cache_line_bytes : int;
    l1_cache_capacity_bytes : int;
  }

  val create :
    simd_lanes:int ->
    sram_banks:int ->
    sram_bank_width_bytes:int ->
    sram_capacity_bytes:int ->
    l1_cache_line_bytes:int ->
    l1_cache_capacity_bytes:int ->
    (t, string) result

  val apple_silicon : t
end

module Execution_profile : sig
  type t = {
    gpu_cores : int;
    max_threads_per_threadgroup : int;
    memory_bandwidth_gbps : float;
    fp16_tflops : float;
    dispatch_overhead_seconds : float;
  }

  val create :
    gpu_cores:int ->
    max_threads_per_threadgroup:int ->
    memory_bandwidth_gbps:float ->
    fp16_tflops:float ->
    dispatch_overhead_seconds:float ->
    (t, string) result

  val m4_pro : t
  val default_apple : t
end

type t = {
  device_name : string;
  memory : Memory_hierarchy.t;
  execution : Execution_profile.t;
}

val create :
  device_name:string ->
  memory:Memory_hierarchy.t ->
  execution:Execution_profile.t ->
  (t, string) result

val default : t

val discover : unit -> t

val bank_conflict_degree :
  Memory_hierarchy.t -> element_bytes:int -> stride_elements:int -> int

val should_fuse_sram_reduction :
  t ->
  elements:int ->
  element_bytes:int ->
  threadgroups:int ->
  barriers:int ->
  bool

val to_json : t -> Yojson.Basic.t
val of_json : Yojson.Basic.t -> (t, string) result
val to_string : t -> string
