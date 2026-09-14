(** Constructive Metal emission for ordinary and paired-row block-quantized
    Linear assignments. Physical decode details come from [Block_schema] and
    lane geometry comes from the target profile. *)

val linear_kernels :
  target:Target_hardware.t -> Block_schema.t -> (string, string) result

val qkv_kernels :
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

val q8_0_linear_add_kernel :
  name:string ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q8_0_m2_add_kernel :
  name:string ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q5_0_linear_add_kernel :
  name:string ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q5_0_m2_add_kernel :
  name:string ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q4_k_linear_add_kernel :
  name:string ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q5_k_linear_add_kernel :
  name:string ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q6_k_linear_add_kernel :
  name:string ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val iq4_xs_linear_add_kernel :
  name:string ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val iq4_xs_m2_add_kernel :
  name:string ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q4_k_m2_x4_add_kernel :
  name:string ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q5_k_m2_x4_add_kernel :
  name:string ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q6_k_m2_x4_add_kernel :
  name:string ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q8_0_multi_linear_kernel :
  name:string ->
  epilogue:int Applicative_expr.t ->
  ?num_slots:int ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q5_0_multi_linear_kernel :
  name:string ->
  epilogue:int Applicative_expr.t ->
  ?num_slots:int ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q4_k_multi_linear_kernel :
  name:string ->
  epilogue:int Applicative_expr.t ->
  ?num_slots:int ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q5_k_multi_linear_kernel :
  name:string ->
  epilogue:int Applicative_expr.t ->
  ?num_slots:int ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q6_k_multi_linear_kernel :
  name:string ->
  epilogue:int Applicative_expr.t ->
  ?num_slots:int ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val iq4_xs_multi_linear_kernel :
  name:string ->
  epilogue:int Applicative_expr.t ->
  ?num_slots:int ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q8_0_multi_export_linear_kernel :
  name:string ->
  epilogues:int Applicative_expr.t list ->
  ?num_slots:int ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q5_0_multi_export_linear_kernel :
  name:string ->
  epilogues:int Applicative_expr.t list ->
  ?num_slots:int ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q4_k_multi_export_linear_kernel :
  name:string ->
  epilogues:int Applicative_expr.t list ->
  ?num_slots:int ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q5_k_multi_export_linear_kernel :
  name:string ->
  epilogues:int Applicative_expr.t list ->
  ?num_slots:int ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val q6_k_multi_export_linear_kernel :
  name:string ->
  epilogues:int Applicative_expr.t list ->
  ?num_slots:int ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

val iq4_xs_multi_export_linear_kernel :
  name:string ->
  epilogues:int Applicative_expr.t list ->
  ?num_slots:int ->
  target:Target_hardware.t ->
  Block_schema.t ->
  (string, string) result

