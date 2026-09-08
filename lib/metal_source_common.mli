(** Shared Metal kernel-entry constructors and source preamble. *)
val kernel_entry_with_threadgroup :
  threadgroup:int * int * int ->
  name:string ->
  operation:Kernel_abi.Operation.t ->
  input_dtype:Ir.Dtype.t -> output_dtype:Ir.Dtype.t -> Kernel_abi.Entry.t
val kernel_entry :
  name:string ->
  operation:Kernel_abi.Operation.t ->
  input_dtype:Ir.Dtype.t -> output_dtype:Ir.Dtype.t -> Kernel_abi.Entry.t
val kernel_entry_with_threadgroup_and_tile :
  tile:Kernel_abi.Entry.tile ->
  threadgroup:int * int * int ->
  name:string ->
  operation:Kernel_abi.Operation.t ->
  input_dtype:Ir.Dtype.t -> output_dtype:Ir.Dtype.t -> Kernel_abi.Entry.t
val kernel_entry_with_tile :
  tile:Kernel_abi.Entry.tile ->
  name:string ->
  operation:Kernel_abi.Operation.t ->
  input_dtype:Ir.Dtype.t -> output_dtype:Ir.Dtype.t -> Kernel_abi.Entry.t
val metal_header : string
