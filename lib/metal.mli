module Program : sig
  type t

  val source : t -> string
  val kernels : t -> Kernel_abi.Entry.t list
end

val lower : Ir.Graph.t -> (Program.t, string) result
val add_cache_kernels : Program.t -> Program.t
val add_block32_kernels : Program.t -> Program.t
val add_kquant_kernels : Program.t -> Program.t
val emit : Ir.Graph.t -> (string, string) result
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
