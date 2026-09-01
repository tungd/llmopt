module Operation : sig
  type t =
    | Matmul
    | Fused_linear
    | Linear
    | Gated_linear
    | W4a16_linear
    | W4a16_qkv_linear
    | W4a16_swiglu_ffn
    | W4a16_lm_head_argmax
    | Rms_norm
    | Rms_rope
    | Rms_rope_qk
    | Rotary_qk
    | Short_conv
    | Short_conv_step
    | Short_conv_prefill
    | Attention
    | Embedding
    | Arange
    | Diff
    | Cumsum
    | Fill
    | Eye
    | Gather2
    | Cast
    | Pointwise
    | Movement
    | Reduction
    | Update_slice
    | Batched_matmul
    | Triangular_recurrence
    | Gated_delta
    | Cache

  val to_string : t -> string
  val of_string : string -> (t, string) result
end

module Entry : sig
  type tile = int * int * int
  type t

  val create :
    name:string ->
    operation:Operation.t ->
    input_dtype:Ir.Dtype.t ->
    output_dtype:Ir.Dtype.t ->
    threadgroup:int * int * int ->
    (t, string) result

  val create_with_tile :
    tile:tile ->
    name:string ->
    operation:Operation.t ->
    input_dtype:Ir.Dtype.t ->
    output_dtype:Ir.Dtype.t ->
    threadgroup:int * int * int ->
    (t, string) result

  val name : t -> string
  val operation : t -> Operation.t
  val input_dtype : t -> Ir.Dtype.t
  val output_dtype : t -> Ir.Dtype.t
  val threadgroup : t -> int * int * int
  val tile : t -> tile option
end

module Linear_tactic : sig
  type t
  type linear_tactic = t

  module Candidate : sig
    module Resource : sig
      type t

      val threadgroup_threads : t -> int
      val simdgroups_per_threadgroup : t -> int
      val output_values_per_threadgroup : t -> int
      val decoded_weight_reuse : t -> int
      val activation_reuse : t -> int
      val split_k_partitions : t -> int
      val threadgroup_sram_bytes : t -> int
      val dispatches : t -> int
    end

    module Legality : sig
      type t

      val resource : t -> Resource.t
      val simd_width : t -> int
      val target_simd_width : t -> int
      val target_max_threads_per_threadgroup : t -> int
      val target_sram_capacity_bytes : t -> int
    end

    module Cost : sig
      type t

      val flops : t -> float
      val physical_storage_bytes : t -> float
      val input_bytes : t -> float
      val output_bytes : t -> float
      val weight_traffic_bytes : t -> float
      val activation_traffic_bytes : t -> float
      val memory_bytes : t -> float
      val workgroups : t -> int
      val core_utilization : t -> float
      val dispatches : t -> int
      val compute_seconds : t -> float
      val memory_seconds : t -> float
      val roofline_seconds : t -> float
      val dispatch_seconds : t -> float
      val total_seconds : t -> float
    end

    type t

    val tactic : t -> linear_tactic
    val schema : t -> Block_schema.t
    val target_tactic : t -> Block_schema.Target_tactic.t
    val legality : t -> Legality.t
    val cost : t -> Cost.t
  end

  module Selection : sig
    type t

    val chosen : t -> Candidate.t
    (** The chosen candidate is the first element of this cost/name-ordered
        list; the list includes every legal alternative. *)
    val alternatives : t -> Candidate.t list
  end

  val select :
    supports_simd:(threads:int -> bool) ->
    m:int ->
    n:int ->
    k:int ->
    input_dtype:Ir.Dtype.t ->
    storage:Ir.Linear_storage.layout ->
    output_dtype:Ir.Dtype.t ->
    t option

  val select_all :
    supports_simd:(threads:int -> bool) ->
    m:int ->
    n:int ->
    k:int ->
    input_dtype:Ir.Dtype.t ->
    storage:Ir.Linear_storage.layout ->
    output_dtype:Ir.Dtype.t ->
    t list

  val name : t -> string
  val threadgroup : t -> int * int * int

  val select_costed :
    target:Target_hardware.t ->
    m:int ->
    n:int ->
    k:int ->
    input_dtype:Ir.Dtype.t ->
    storage:Ir.Linear_storage.layout ->
    output_dtype:Ir.Dtype.t ->
    Selection.t option
end
