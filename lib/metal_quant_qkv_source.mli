(** Metal quant QKV shader catalog interface. *)
val quant_qkv_params_source : string
val q4_0_qkv_base_source : string
val q4_0_qkv_source : string
val q5_0_qkv_source : string
val q4_0_qkv_entries : Kernel_abi.Entry.t list
val q5_0_qkv_entries : Kernel_abi.Entry.t list
val quant_qkv_entries : Kernel_abi.Entry.t list
val quant_qkv_source : string

(** Separate sources so homogeneous Q5 packages do not require Q8 declarations. *)
val q5_0_qkv_homogeneous_source : string
val q5_0_qkv_mixed_source : string
val q5_0_qkv_homogeneous_entries : Kernel_abi.Entry.t list
val q5_0_qkv_mixed_entries : Kernel_abi.Entry.t list
