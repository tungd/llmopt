module Program : sig
  type t

  val source : t -> string
  val kernels : t -> Kernel_abi.Entry.t list
end

module Tactic : sig
  type t

  val select_linear :
    target:Target_hardware.t ->
    m:int ->
    n:int ->
    k:int ->
    input_dtype:Ir.Dtype.t ->
    storage:Ir.Linear_storage.layout ->
    output_dtype:Ir.Dtype.t ->
    t option

  val select_attention :
    target:Target_hardware.t ->
    head_dimension:int ->
    input_dtype:Ir.Dtype.t ->
    output_dtype:Ir.Dtype.t ->
    t option

  val name : t -> string
  val threadgroup : t -> int * int * int
end

val lower :
  ?target:Target_hardware.t ->
  ?fusion_regions:Kernel_region.t list ->
  Ir.Graph.t ->
  (Program.t, string) result
val add_cache_kernels : Program.t -> Program.t
val add_block32_kernels : Program.t -> Program.t
val emit : ?target:Target_hardware.t -> Ir.Graph.t -> (string, string) result
val emit_dequant_q4_0 : unit -> string
val emit_tile32_affine4 : ?target:Target_hardware.t -> unit -> string
val emit_tile32_affine8 : ?target:Target_hardware.t -> unit -> string
val q4_0_source : string
val block32_entries : Kernel_abi.Entry.t list
val attention_entries : Kernel_abi.Entry.t list
val attention_source : string
val emit_parametric_w4a16_linear : name:string -> epilogue:Epilogue.t -> string
val emit_simdgroup_matrix_gemm : name:string -> epilogue:Epilogue.t -> string
