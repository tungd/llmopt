module Program : sig
  type t

  val source : t -> string
  val kernels : t -> Kernel_abi.Entry.t list
end

val lower : Ir.Graph.t -> (Program.t, string) result
val emit : Ir.Graph.t -> (string, string) result
