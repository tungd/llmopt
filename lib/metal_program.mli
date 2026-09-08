(** Minimal assembled Metal program interface. *)
module Program :
  sig
    type t = { source : string; kernels : Kernel_abi.Entry.t list; }
    val make : source:string -> kernels:Kernel_abi.Entry.t list -> t
    val source : t -> string
    val kernels : t -> Kernel_abi.Entry.t list
  end
