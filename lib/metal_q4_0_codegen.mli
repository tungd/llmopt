(** Q4_0 constructive Metal emission, specialized epilogue/multi-linear wrappers,
    and add twins. *)

val q4_0_linear :
  target:Target_hardware.t -> Block_schema.t -> (string, string) result

val q4_0_qkv :
  target:Target_hardware.t -> Block_schema.t -> (string, string) result

val q4_0_x8_epilogue_kernel :
  name:string ->
  epilogue:Epilogue.t ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q4_0_x4_epilogue_kernel :
  name:string ->
  epilogue:Epilogue.t ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q4_0_m16_epilogue_kernel :
  name:string ->
  epilogue:Epilogue.t ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q4_0_m32_epilogue_kernel :
  name:string ->
  epilogue:Epilogue.t ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q4_0_multi_linear_f16_x8_kernel :
  name:string ->
  epilogue:int Applicative_expr.t ->
  ?num_slots:int ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q4_0_multi_export_linear_f16_x8_kernel :
  name:string ->
  epilogues:int Applicative_expr.t list ->
  ?num_slots:int ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q4_0_linear_add_kernel :
  name:string ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q4_0_m2_add_kernel :
  name:string ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result
