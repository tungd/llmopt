val dispatch_rms_norm :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  epsilon:float ->
  input:Ir.Value.t ->
  weight:Ir.Value.t option ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_rms_norm_add :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  epsilon:float ->
  input:Ir.Value.t ->
  weight:Ir.Value.t ->
  residual:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_l2_norm :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  epsilon:float ->
  input:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_l2_norm_slice :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  config:Ir.L2_norm_slice.t ->
  input:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_rms_rope :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  config:Ir.Rms_rope.t ->
  input:Ir.Value.t ->
  weight:Ir.Value.t ->
  cosine:Ir.Value.t ->
  sine:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_rms_rope_qk :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  q_heads:int ->
  k_heads:int ->
  width:int ->
  half_dimension:int ->
  epsilon:float ->
  q_input:Ir.Value.t ->
  q_weight:Ir.Value.t ->
  k_input:Ir.Value.t ->
  k_weight:Ir.Value.t ->
  cosine:Ir.Value.t ->
  sine:Ir.Value.t ->
  q_output:Ir.Value.t ->
  k_output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_rotary_qk :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  q_heads:int ->
  k_heads:int ->
  width:int ->
  half_dimension:int ->
  q_input:Ir.Value.t ->
  k_input:Ir.Value.t ->
  cosine:Ir.Value.t ->
  sine:Ir.Value.t ->
  q_output:Ir.Value.t ->
  k_output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result
