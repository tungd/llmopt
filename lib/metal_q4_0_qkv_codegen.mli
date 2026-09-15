(** Metal metal_q4_0_qkv_codegen interface. *)
val emit_q4_0_qkv_linear_f16_x8 :
  scale_offset:int ->
  quant_offset:int ->
  zero:int -> block_elements:int -> block_bytes:int -> string
val emit_q4_0_qkv_linear_f16_x16 :
  scale_offset:int ->
  quant_offset:int ->
  zero:int -> block_elements:int -> block_bytes:int -> string
val emit_q4_0_qkv_linear_f16_m32 :
  scale_offset:int ->
  quant_offset:int ->
  zero:int -> block_elements:int -> block_bytes:int -> string
