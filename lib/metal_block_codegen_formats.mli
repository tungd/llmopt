(** Constructive Metal emission for non-Q4_0 block-quantized formats
    (Q8_0, Q5_0, Q4_K, Q5_K, Q6_K, IQ4_XS) and their residual/add twins. *)

val q8_0_linear :
  target:Target_hardware.t -> Block_schema.t -> (string, string) result

val q5_0_linear :
  target:Target_hardware.t -> Block_schema.t -> (string, string) result

val q4_k_linear :
  target:Target_hardware.t -> Block_schema.t -> (string, string) result

val q5_k_linear :
  target:Target_hardware.t -> Block_schema.t -> (string, string) result

val q6_k_linear :
  target:Target_hardware.t -> Block_schema.t -> (string, string) result

val iq4_xs_linear :
  target:Target_hardware.t -> Block_schema.t -> (string, string) result

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
