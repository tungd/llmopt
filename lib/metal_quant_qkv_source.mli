(** Metal quant QKV shader catalog interface. *)
val quant_qkv_params_source : string
val q4_0_qkv_base_source : string
val q4_0_qkv_source : string
val q4_0_qkv_entries : Kernel_abi.Entry.t list
val tile32_affine4_qkv_source : string
val tile32_affine4_qkv_entries : Kernel_abi.Entry.t list
val tile32_affine8_qkv_source : string
val tile32_affine8_qkv_entries : Kernel_abi.Entry.t list
val quant_qkv_entries : Kernel_abi.Entry.t list
val quant_qkv_source : string
