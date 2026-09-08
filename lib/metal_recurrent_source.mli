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
val triangular_recurrence_source : string
val triangular_recurrence_entries : Kernel_abi.Entry.t list
type gated_delta_tactic =
    Gated_delta_head_major_f32 of int
  | Gated_delta_token_major_f16 of int
val gated_delta_width : gated_delta_tactic -> int
val gated_delta_kernel_name : gated_delta_tactic -> string
val gated_delta_source : gated_delta_tactic -> string
val gated_delta_entry : gated_delta_tactic -> Kernel_abi.Entry.t
