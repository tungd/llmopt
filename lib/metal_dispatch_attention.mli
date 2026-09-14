val dispatch_attention :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  config:Ir.Attention.t ->
  query:Ir.Value.t ->
  key:Ir.Value.t ->
  value:Ir.Value.t ->
  mask:Ir.Value.t option ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_paged_attention_q8 :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  config:Ir.Paged_attention_q8.t ->
  query:Ir.Value.t ->
  current_key:Ir.Value.t ->
  current_value:Ir.Value.t ->
  pool:Ir.Value.t ->
  slots:Ir.Value.t ->
  mask_values:Ir.Value.t list ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_speculative_attention :
  ?batch:Metal_execution_state.Execution_batch.t ->
  Metal_native.runtime ->
  query:Metal_native.Buffer.t ->
  key:Metal_native.Buffer.t ->
  value:Metal_native.Buffer.t ->
  output:Metal_native.Buffer.t ->
  batches:int ->
  heads:int ->
  query_length:int ->
  key_length:int ->
  head_dimension:int ->
  past_length:int ->
  scale:float ->
  ?kv_heads:int ->
  ?tree_masks:int array ->
  unit ->
  (string, string) result
