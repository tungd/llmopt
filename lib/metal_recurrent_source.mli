(** Metal recurrent shader catalog interface. *)
val replace_first : needle:string -> replacement:string -> string -> string
val short_conv_source : string
val short_conv_f32_weight_source : string
val short_conv_silu_source : string
val short_conv_entries : Kernel_abi.Entry.t list
val short_conv_silu_entries : Kernel_abi.Entry.t list
val short_conv_step_source : string
val short_conv_step_entries : Kernel_abi.Entry.t list
val short_conv_step_fused_source : string
val short_conv_step_fused_entries : Kernel_abi.Entry.t list
val short_conv_prefill_source : string
val short_conv_prefill_entries : Kernel_abi.Entry.t list
val short_conv_prefill_init_source : string
val short_conv_prefill_init_entries : Kernel_abi.Entry.t list
