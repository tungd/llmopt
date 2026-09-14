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
val emit_block32_multi_linear :
  name:string ->
  epilogue:int Applicative_expr.t ->
  ?num_slots:int ->
  value_function:string ->
  block_elements:int ->
  block_bytes:int ->
  unit ->
  string
val emit_block32_linear_add :
  name:string ->
  value_function:string -> block_elements:int -> block_bytes:int -> string
val emit_block32_m2_add :
  name:string ->
  value_function:string -> block_elements:int -> block_bytes:int -> string
val emit_superblock_linear_add :
  name:string ->
  value_function:string ->
  block_elements:int -> block_bytes:int -> subblock_elements:int -> string
val emit_superblock_m2_add :
  name:string ->
  value_function:string ->
  block_elements:int -> block_bytes:int -> subblock_elements:int -> string
val emit_superblock_multi_linear :
  name:string ->
  epilogue:int Applicative_expr.t ->
  ?num_slots:int ->
  value_function:string ->
  block_elements:int ->
  block_bytes:int ->
  subblock_elements:int ->
  unit ->
  string
val emit_superblock_linear_pair :
  name:string ->
  value_function:string ->
  block_elements:int -> block_bytes:int -> subblock_elements:int -> string
val require_superblock :
  target:Target_hardware.t ->
  Block_schema.Physical_layout.t ->
  Block_schema.Logical_decode.t -> string -> (unit, string) result
val emit_superblock_m2_x4_flat :
  name:string ->
  value_function:string ->
  block_elements:int -> block_bytes:int -> subblock_elements:int -> string
val emit_superblock_m2_x4_add :
  name:string ->
  value_function:string ->
  block_elements:int -> block_bytes:int -> subblock_elements:int -> string
val emit_superblock :
  m2_x4:bool ->
  Block_schema.t -> name:string -> value_function:string -> string -> string
