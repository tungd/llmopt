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

val lower : ?target:Target_hardware.t -> Ir.Graph.t -> (Program.t, string) result
val add_cache_kernels : Program.t -> Program.t
val add_block32_kernels : Program.t -> Program.t
val add_kquant_kernels : Program.t -> Program.t
val emit : ?target:Target_hardware.t -> Ir.Graph.t -> (string, string) result
val emit_dequant_q8_0 : unit -> string
val emit_dequant_q5_0 : unit -> string
val emit_dequant_q4_0 : unit -> string
val emit_dequant_q4_k : unit -> string
val emit_dequant_q5_k : unit -> string
val emit_dequant_q6_k : unit -> string
val q8_0_source : string
val q5_0_source : string
val q4_0_source : string
val q4_k_source : string
val q5_k_source : string
val q6_k_source : string
val block32_entries : Kernel_abi.Entry.t list
val kquant_entries : Kernel_abi.Entry.t list
val attention_entries : Kernel_abi.Entry.t list
val attention_source : string
val emit_parametric_w4a16_linear : name:string -> epilogue:Epilogue.t -> string
val emit_parametric_q4_k_linear : name:string -> epilogue:Epilogue.t -> string
val emit_simdgroup_matrix_gemm : name:string -> epilogue:Epilogue.t -> string


