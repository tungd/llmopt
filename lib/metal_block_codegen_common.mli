(** Metal metal_block_codegen_common interface. *)
val field_offset :
  Block_schema.Physical_layout.t ->
  Block_schema.Physical_layout.Field.id -> (int, string) result
val offset_zero :
  Block_schema.Physical_layout.Signedness.t -> (int, string) result
val require_block32 :
  target:Target_hardware.t ->
  Block_schema.Physical_layout.t -> string -> (unit, string) result
val emit_linear_pair :
  name:string ->
  value_function:string -> block_elements:int -> block_bytes:int -> string
val emit_block32_linear_add :
  name:string ->
  value_function:string -> block_elements:int -> block_bytes:int -> string
val emit_block32_m2_add :
  name:string ->
  value_function:string -> block_elements:int -> block_bytes:int -> string
