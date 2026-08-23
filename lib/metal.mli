module Program : sig
  type t

  val source : t -> string
  val kernels : t -> Kernel_abi.Entry.t list
end

val lower : Ir.Graph.t -> (Program.t, string) result
val add_cache_kernels : formats:Kv_cache.Format.t list -> Program.t -> Program.t
val emit : Ir.Graph.t -> (string, string) result
