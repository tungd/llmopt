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
