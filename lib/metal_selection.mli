(** Metal kernel-tactic selection interface (internal superset; the Metal facade narrows it). *)
module Tactic :
  sig
    type t = { name : string; threadgroup : int * int * int; }
    type attention_problem = {
      target : Target_hardware.t;
      head_dimension : int;
      input_dtype : Ir.Dtype.t;
      output_dtype : Ir.Dtype.t;
    }
    type attention_registration = {
      build : attention_problem -> t;
      accepts : attention_problem -> bool;
    }
    val name : t -> string
    val threadgroup : t -> int * int * int
    val supports_simd : Target_hardware.t -> threads:int -> bool
    val f16_io : Ir.Dtype.t -> Ir.Dtype.t -> bool
    val select_linear :
      target:Target_hardware.t ->
      m:int ->
      n:int ->
      k:int ->
      input_dtype:Ir.Dtype.t ->
      storage:Ir.Linear_storage.layout -> output_dtype:Ir.Dtype.t -> t option
    val attention_registry : attention_registration list
    val select_attention :
      target:Target_hardware.t ->
      head_dimension:int ->
      input_dtype:Ir.Dtype.t -> output_dtype:Ir.Dtype.t -> t option
  end
val attention_simd_kernel_name : int -> string
type gated_delta_tactic =
  Metal_recurrent_source.gated_delta_tactic =
    Gated_delta_head_major_f32 of int
  | Gated_delta_token_major_f16 of int
val gated_delta_tactics : Ir.Graph.t -> gated_delta_tactic list
val has_rms_norm : Ir.Graph.t -> bool
val has_l2_norm : Ir.Graph.t -> bool
val has_l2_norm_slice : Ir.Graph.t -> bool
val has_rms_rope : Ir.Graph.t -> bool
val has_rms_rope_qk : Ir.Graph.t -> bool
val has_rotary_qk : Ir.Graph.t -> bool
val has_short_conv : Ir.Graph.t -> bool
val has_short_conv_silu : Ir.Graph.t -> bool
val has_short_conv_step : Ir.Graph.t -> bool
val has_short_conv_step_fused : Ir.Graph.t -> bool
val has_short_conv_prefill : Ir.Graph.t -> bool
val has_attention : Ir.Graph.t -> bool
val attention_tactics :
  target:Target_hardware.t -> Ir.Graph.t -> (int * Tactic.t) list
val attention_simd_head_dimensions : (int * Tactic.t) list -> int list
val has_embedding : Ir.Graph.t -> bool
val has_primitive : Ir.Graph.t -> (Ir.Primitive.t -> bool) -> bool
val has_materialized_movement : Ir.Graph.t -> bool
val has_tensor_primitive : Ir.Graph.t -> bool
val has_w4a16_linear : Ir.Graph.t -> bool
val has_w4a16_qkv_linear : Ir.Graph.t -> bool
val has_quant_qkv_linear : Ir.Graph.t -> bool
val has_w4a16_swiglu_ffn : Ir.Graph.t -> bool
val has_w4a16_lm_head_argmax : Ir.Graph.t -> bool
val has_f16_linear : Ir.Graph.t -> bool
val has_f16_bf16_linear : Ir.Graph.t -> bool
val has_f16_f32_linear : Ir.Graph.t -> bool
val has_quant_linear : Ir.Graph.t -> bool
val quant_linear_tactics :
  target:Target_hardware.t ->
  Ir.Graph.t -> (Ir.Dtype.quant_type * Tactic.t) list
val graph_quant_types : Ir.Graph.t -> Ir.Dtype.quant_type list
val quant_linear_add_tactics :
  target:Target_hardware.t ->
  Ir.Graph.t -> (Ir.Dtype.quant_type * Tactic.t) list
module Region_node_set :
  sig
    type elt = Int.t
    type t = Set.Make(Int).t
    val empty : t
    val add : elt -> t -> t
    val singleton : elt -> t
    val remove : elt -> t -> t
    val union : t -> t -> t
    val inter : t -> t -> t
    val disjoint : t -> t -> bool
    val diff : t -> t -> t
    val cardinal : t -> int
    val elements : t -> elt list
    val min_elt : t -> elt
    val min_elt_opt : t -> elt option
    val max_elt : t -> elt
    val max_elt_opt : t -> elt option
    val choose : t -> elt
    val choose_opt : t -> elt option
    val find : elt -> t -> elt
    val find_opt : elt -> t -> elt option
    val find_first : (elt -> bool) -> t -> elt
    val find_first_opt : (elt -> bool) -> t -> elt option
    val find_last : (elt -> bool) -> t -> elt
    val find_last_opt : (elt -> bool) -> t -> elt option
    val iter : (elt -> unit) -> t -> unit
    val fold : (elt -> 'acc -> 'acc) -> t -> 'acc -> 'acc
    val map : (elt -> elt) -> t -> t
    val filter : (elt -> bool) -> t -> t
    val filter_map : (elt -> elt option) -> t -> t
    val partition : (elt -> bool) -> t -> t * t
    val split : elt -> t -> t * bool * t
    val is_empty : t -> bool
    val is_singleton : t -> bool
    val mem : elt -> t -> bool
    val equal : t -> t -> bool
    val compare : t -> t -> int
    val subset : t -> t -> bool
    val for_all : (elt -> bool) -> t -> bool
    val exists : (elt -> bool) -> t -> bool
    val to_list : t -> elt list
    val of_list : elt list -> t
    val to_seq_from : elt -> t -> elt Seq.t
    val to_seq : t -> elt Seq.t
    val to_rev_seq : t -> elt Seq.t
    val add_seq : elt Seq.t -> t -> t
    val of_seq : elt Seq.t -> t
  end
val region_node_ids : Kernel_region.t list -> Region_node_set.t
val without_region_nodes : Ir.Graph.t -> Kernel_region.t list -> Ir.Graph.t
val quant_add_twin_servable : Ir.Dtype.quant_type -> Tactic.t -> bool
val is_residual_add_epilogue : Epilogue.t -> bool
val region_tactics :
  target:Target_hardware.t ->
  Kernel_region.region list ->
  (Ir.Dtype.quant_type * Tactic.t) list *
  (Ir.Dtype.quant_type * Tactic.t) list *
  (Ir.Dtype.quant_type * Tactic.t * Epilogue.t) list *
  (Ir.Dtype.quant_type * Tactic.t * int * int Applicative_expr.t) list *
  bool * bool * bool * bool * gated_delta_tactic list
