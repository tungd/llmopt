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
val emit_q4_0_linear_f16_x16 :
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
val emit_q4_0_multi_linear_f16_x8 :
  name:string ->
  epilogue:int Applicative_expr.t ->
  ?num_slots:int ->
  scale_offset:int ->
  quant_offset:int ->
  zero:int -> block_elements:int -> block_bytes:int -> unit -> string
val emit_q4_0_multi_linear_f16_x16 :
  name:string ->
  epilogue:int Applicative_expr.t ->
  ?num_slots:int ->
  scale_offset:int ->
  quant_offset:int ->
  zero:int -> block_elements:int -> block_bytes:int -> unit -> string
val emit_q4_0_multi_export_linear_f16_x8 :
  name:string ->
  epilogues:int Applicative_expr.t list ->
  ?num_slots:int ->
  scale_offset:int ->
  quant_offset:int ->
  zero:int -> block_elements:int -> block_bytes:int -> unit -> string
val emit_q4_0_multi_export_linear_f16_x16 :
  name:string ->
  epilogues:int Applicative_expr.t list ->
  ?num_slots:int ->
  scale_offset:int ->
  quant_offset:int ->
  zero:int -> block_elements:int -> block_bytes:int -> unit -> string
