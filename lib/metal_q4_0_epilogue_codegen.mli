(** Metal metal_q4_0_epilogue_codegen interface. *)
val ( let* ) : ('a, 'b) result -> ('a -> ('c, 'b) result) -> ('c, 'b) result
val find_substring_from : string -> string -> int -> int option
val q4_0_single_row_epilogue_store : columns:int -> Epilogue.t -> string
val q4_0_single_row_digest_of_flat :
  columns:int ->
  label:string ->
  name:string -> Epilogue.t -> string -> (string, string) result
val q4_0_x8_digest_of_flat :
  name:string -> Epilogue.t -> string -> (string, string) result
val q4_0_x4_digest_of_flat :
  name:string -> Epilogue.t -> string -> (string, string) result
val q4_0_row_block_epilogue_stores :
  rows:int -> row_guard:string -> Epilogue.t -> string
val q4_0_row_block_digest_of_flat :
  rows:int ->
  row_guard:string ->
  label:string ->
  name:string -> Epilogue.t -> string -> (string, string) result
val q4_0_m16_digest_of_flat :
  name:string -> Epilogue.t -> string -> (string, string) result
val q4_0_m32_digest_of_flat :
  name:string -> Epilogue.t -> string -> (string, string) result
val q4_0_decode_epilogue_kernel :
  ?rows:int ->
  ?row_guard:string ->
  columns:int ->
  label:string ->
  emit_flat:(scale_offset:int ->
             quant_offset:int ->
             zero:int -> block_elements:int -> block_bytes:int -> string) ->
  name:string ->
  epilogue:Epilogue.t ->
  target:Target_hardware.t -> Block_schema.t -> (string, string) result
