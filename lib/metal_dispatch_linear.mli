val dispatch_w4a16_swiglu_ffn_command :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  m:int ->
  n:int ->
  k:int ->
  epsilon:float ->
  Ir.Value.t list ->
  Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_w4a16_qkv_linear_command :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  m:int ->
  k:int ->
  n_q:int ->
  n_k:int ->
  n_v:int ->
  extra_outputs:Ir.Value.t list ->
  Ir.Value.t list ->
  Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_quant_qkv_linear_command :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  m:int ->
  k:int ->
  n_q:int ->
  n_k:int ->
  n_v:int ->
  extra_outputs:Ir.Value.t list ->
  Ir.Value.t list ->
  Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_w4a16_lm_head_argmax_command :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  m:int ->
  n:int ->
  k:int ->
  epsilon:float ->
  extra_outputs:Ir.Value.t list ->
  Ir.Value.t list ->
  Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_linear :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  m:int ->
  n:int ->
  k:int ->
  bias:bool ->
  Ir.Value.t list ->
  Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_linear_add :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  m:int ->
  n:int ->
  k:int ->
  input:Ir.Value.t ->
  weight:Ir.Value.t ->
  residual:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_gated_linear :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  m:int ->
  n:int ->
  k:int ->
  activation:Ir.Gated_activation.t ->
  input:Ir.Value.t ->
  gate_weight:Ir.Value.t ->
  up_weight:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_w4a16_linear :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  m:int ->
  n:int ->
  k:int ->
  has_bias:bool ->
  Ir.Value.t list ->
  Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result
