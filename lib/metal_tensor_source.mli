(** Metal dense Linear shader catalog interface. *)
val linear_f16_source : string
val linear_f16_entries : Kernel_abi.Entry.t list
val linear_f16_bf16_source : string
val linear_f16_bf16_entries : Kernel_abi.Entry.t list
val linear_f16_f32_source : string
val linear_f16_f32_entries : Kernel_abi.Entry.t list
val replace_first : needle:string -> replacement:string -> string -> string
val replace_all : needle:string -> replacement:string -> String.t -> String.t
