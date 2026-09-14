val dispatch_triangular_recurrence :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  config:Ir.Triangular_recurrence.t ->
  input:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_gated_delta :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  query:Ir.Value.t ->
  key:Ir.Value.t ->
  value:Ir.Value.t ->
  gate:Ir.Value.t ->
  beta:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_short_conv_step :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  fused:bool ->
  config:Ir.Short_conv_step.t ->
  in_proj:Ir.Value.t ->
  conv_state:Ir.Value.t ->
  conv_weight:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_short_conv_prefill :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  config:Ir.Short_conv_prefill.t ->
  in_proj:Ir.Value.t ->
  conv_weight:Ir.Value.t ->
  conv_state_out:Ir.Value.t ->
  rest:Ir.Value.t list ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_short_conv :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  config:Ir.Short_conv.t ->
  input:Ir.Value.t ->
  weight:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_short_conv_silu :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  config:Ir.Short_conv_silu.t ->
  input:Ir.Value.t ->
  weight:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result
