(** Metal metal_q4_0_decode_codegen interface. *)
val emit_q4_0_linear_f16_x4 :
  ?name:string ->
  ?epilogue:Epilogue.t ->
  scale_offset:int ->
  quant_offset:int ->
  zero:int -> block_elements:int -> block_bytes:int -> unit -> string
val emit_q4_0_linear_f16_x8 :
  ?split_k:int ->
  ?name:string ->
  ?epilogue:Epilogue.t ->
  scale_offset:int ->
  quant_offset:int ->
  zero:int -> block_elements:int -> block_bytes:int -> unit -> string
val emit_q4_0_gated_linear_f16_x8 :
  scale_offset:int ->
  quant_offset:int ->
  zero:int -> block_elements:int -> block_bytes:int -> string
