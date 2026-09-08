(** Metal W4A16 shader catalog interface. *)
val w4a16_source : string
val w4a16_entries : Kernel_abi.Entry.t list
val w4a16_qkv_linear_source : string
val w4a16_qkv_linear_entries : Kernel_abi.Entry.t list
val w4a16_swiglu_rms_kernel : name:string -> input_type:string -> string
val w4a16_swiglu_parallel_source : string
val w4a16_swiglu_ffn_source : string
val w4a16_swiglu_ffn_entries : Kernel_abi.Entry.t list
val w4a16_lm_head_argmax_kernel :
  name:string -> value_type:string -> extra_output:bool -> string
val w4a16_lm_head_argmax_stage1_kernel :
  name:string -> value_type:string -> extra_output:bool -> string
val w4a16_lm_head_reduce_kernel : string
val w4a16_lm_head_argmax_source : string
val w4a16_lm_head_argmax_entries : Kernel_abi.Entry.t list
