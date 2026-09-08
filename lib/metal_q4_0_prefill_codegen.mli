(** Metal metal_q4_0_prefill_codegen interface. *)
val emit_q4_0_linear_f16_m16 :
  ?name:string ->
  ?epilogue:Epilogue.t ->
  scale_offset:int ->
  quant_offset:int ->
  zero:int -> block_elements:int -> block_bytes:int -> unit -> string
val emit_q4_0_linear_f16_m16_add :
  scale_offset:int ->
  quant_offset:int ->
  zero:int -> block_elements:int -> block_bytes:int -> string
val emit_q4_0_linear_f16_m32 :
  ?name:string ->
  ?epilogue:Epilogue.t ->
  scale_offset:int ->
  quant_offset:int ->
  zero:int -> block_elements:int -> block_bytes:int -> unit -> string
val emit_q4_0_linear_f16_m32_add :
  scale_offset:int ->
  quant_offset:int ->
  zero:int -> block_elements:int -> block_bytes:int -> string
