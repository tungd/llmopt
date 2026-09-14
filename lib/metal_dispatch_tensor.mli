val pointwise_kernel :
  Ir.Pointwise.t -> Ir.Value.t -> (string * Ir.Dtype.t, string) result

val pointwise_operand_shape :
  Ir.Pointwise.operand -> int list option

val pointwise_operand_scalar :
  Ir.Pointwise.operand -> Ir.Scalar.t option

val movement_kernel :
  Ir.Movement.t ->
  Ir.Value.t ->
  Ir.Value.t ->
  (string * Ir.Dtype.t, string) result

val reduction_kernel :
  Ir.Reduction.t ->
  Ir.Value.t ->
  Ir.Value.t ->
  (string * int, string) result

val update_slice_kernel :
  Ir.Value.t ->
  Ir.Value.t ->
  Ir.Value.t ->
  (string, string) result

val dispatch_alias_movement :
  Metal_execution_state.t ->
  input:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_transpose :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  movement:Ir.Movement.t ->
  axis0:int ->
  axis1:int ->
  input:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_index :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  movement:Ir.Movement.t ->
  index:Tensor_shape.Index.t ->
  input:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_expand :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  movement:Ir.Movement.t ->
  input:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_concat :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  movement:Ir.Movement.t ->
  axis:int ->
  left:Ir.Value.t ->
  right:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_roll :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  movement:Ir.Movement.t ->
  axis:int ->
  shift:int ->
  input:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_pad_right_zero :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  axis:int ->
  input:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_reduce :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  reduction:Ir.Reduction.t ->
  input:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_update_slice :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  index:Tensor_shape.Index.t ->
  destination:Ir.Value.t ->
  source:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_cast :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  dtype:Ir.Dtype.t ->
  input:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_gelu :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  input:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_pointwise :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  operation:Ir.Pointwise.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_matmul :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  m:int ->
  n:int ->
  k:int ->
  lhs:Ir.Value.t ->
  rhs:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_fused_matmul_bias :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  m:int ->
  n:int ->
  k:int ->
  lhs:Ir.Value.t ->
  rhs:Ir.Value.t ->
  bias:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_embedding :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  indices:Ir.Value.t ->
  weight:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_arange :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  config:Ir.Arange.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_fill :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  scalar:Ir.Scalar.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_triangular :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  upper:bool ->
  diagonal:int ->
  input:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_masked_fill :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  scalar:Ir.Scalar.t ->
  input:Ir.Value.t ->
  mask:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_eye :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_batched_matmul :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  lhs:Ir.Value.t ->
  rhs:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_diff :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  config:Ir.Diff.t ->
  source:Ir.Value.t ->
  prepend:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_cumsum :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  config:Ir.Cumsum.t ->
  input:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result

val dispatch_gather2 :
  Metal_execution_state.context ->
  Metal_execution_state.t ->
  source:Ir.Value.t ->
  first_index:Ir.Value.t ->
  second_index:Ir.Value.t ->
  output:Ir.Value.t ->
  (Metal_execution_state.t * string list, string) result
