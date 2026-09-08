(** Metal attention shader catalog interface. *)
val replace_all : needle:string -> replacement:string -> String.t -> String.t
val attention_source : string
val attention_simd_template : string
val attention_tree_speculative_source : string
val attention_simd_kernel_name : int -> string
val attention_simd_source : int -> String.t
val attention_simd_entry : int -> Kernel_abi.Entry.t
val attention_entries : Kernel_abi.Entry.t list
