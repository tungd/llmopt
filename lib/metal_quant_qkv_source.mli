(** Metal quant QKV shader catalog interface. *)
val quant_qkv_params_source : string
val q4_0_qkv_base_source : string
val q4_0_qkv_source : string
val q5_0_qkv_source : string
val q4_0_qkv_entries : Kernel_abi.Entry.t list
val q5_0_qkv_entries : Kernel_abi.Entry.t list
val quant_qkv_entries : Kernel_abi.Entry.t list
val quant_qkv_source : string
