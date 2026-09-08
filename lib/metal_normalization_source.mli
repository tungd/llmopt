(** Metal normalization shader catalog interface. *)
val replace_first : needle:string -> replacement:string -> string -> string
val replace_all : needle:string -> replacement:string -> String.t -> String.t
val rms_norm_source : string
val rms_norm_entries : Kernel_abi.Entry.t list
val l2_norm_source : string
val l2_norm_entries : Kernel_abi.Entry.t list
val l2_norm_slice_source : string
val l2_norm_slice_entries : Kernel_abi.Entry.t list
val rms_rope_f16_weight_source : string
val rms_rope_f32_weight_source : string
val rms_rope_source : string
val rms_rope_entries : Kernel_abi.Entry.t list
val rms_rope_qk_f16_weight_source : string
val rms_rope_qk_f32_weight_source : string
val rms_rope_qk_source : string
val rms_rope_qk_entries : Kernel_abi.Entry.t list
val rotary_qk_source : string
val rotary_qk_entries : Kernel_abi.Entry.t list
