(** Metal cache shader catalog interface. *)
val cache_source : string
val cache_entry : string -> Ir.Dtype.t -> Ir.Dtype.t -> Kernel_abi.Entry.t
val cache_entries : Kernel_abi.Entry.t list
