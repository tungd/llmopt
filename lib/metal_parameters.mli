val max_rank : int

val set_u32 : bytes -> int -> int -> (unit, string) result
val u32s : int list -> (bytes, string) result

val rms_norm : rows:int -> width:int -> epsilon:float -> (bytes, string) result

val l2_norm_slice :
  input_rows:int ->
  groups:int ->
  width:int ->
  input_width:int ->
  offset:int ->
  epsilon:float ->
  (bytes, string) result

val w4a16_lm_head_argmax :
  m:int -> n:int -> k:int -> epsilon:float -> (bytes, string) result

val w4a16_swiglu_ffn :
  m:int -> n:int -> k:int -> epsilon:float -> (bytes, string) result

val rms_rope :
  batches:int ->
  tokens:int ->
  heads:int ->
  width:int ->
  half_dimension:int ->
  trig_batches:int ->
  epsilon:float ->
  (bytes, string) result

val rms_rope_qk :
  batches:int ->
  tokens:int ->
  q_heads:int ->
  k_heads:int ->
  width:int ->
  half_dimension:int ->
  trig_batches:int ->
  epsilon:float ->
  (bytes, string) result

val rotary_qk :
  batches:int ->
  tokens:int ->
  q_heads:int ->
  k_heads:int ->
  width:int ->
  half_dimension:int ->
  trig_batches:int ->
  (bytes, string) result

val arange : count:int -> start:int -> step:int -> (bytes, string) result

val fill : count:int -> Ir.Scalar.t -> (bytes, string) result

val attention :
  batches:int ->
  heads:int ->
  query_length:int ->
  key_length:int ->
  head_dimension:int ->
  mask_batches:int ->
  mask_heads:int ->
  causal:bool ->
  scale:float ->
  ?kv_heads:int ->
  ?token_first_output:int ->
  ?past_length:int ->
  ?token_first_value:int ->
  ?mask_dtype:int ->
  unit ->
  (bytes, string) result

val paged_attention_q8 :
  batches:int ->
  query_heads:int ->
  kv_heads:int ->
  past_length:int ->
  head_dimension:int ->
  mask_batches:int ->
  mask_heads:int ->
  token_elements:int ->
  key_offset:int ->
  value_offset:int ->
  group_size:int ->
  token_stride:int ->
  scale:float ->
  mask_dtype:int ->
  ?query_length:int ->
  ?window:int ->
  unit ->
  (bytes, string) result

val speculative_attention :
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
  (bytes, string) result

val movement :
  input_dimensions:int list ->
  output_dimensions:int list ->
  axis0:int ->
  axis1:int ->
  (bytes, string) result

val transpose :
  input_shape:Tensor_shape.t ->
  output_shape:Tensor_shape.t ->
  axis0:int ->
  axis1:int ->
  (bytes, string) result

val expand :
  input_shape:Tensor_shape.t ->
  output_shape:Tensor_shape.t ->
  (bytes, string) result

val index_parameters :
  count:int ->
  input_shape:Tensor_shape.t ->
  output_shape:Tensor_shape.t ->
  Tensor_shape.Index.t ->
  (bytes, string) result

val index :
  input_shape:Tensor_shape.t ->
  output_shape:Tensor_shape.t ->
  Tensor_shape.Index.t ->
  (bytes, string) result

val update_slice :
  destination_shape:Tensor_shape.t ->
  source_shape:Tensor_shape.t ->
  Tensor_shape.Index.t ->
  (bytes, string) result

val concat :
  left_shape:Tensor_shape.t ->
  right_shape:Tensor_shape.t ->
  output_shape:Tensor_shape.t ->
  axis:int ->
  (bytes, string) result

val roll :
  shape:Tensor_shape.t ->
  axis:int ->
  shift:int ->
  (bytes, string) result

val align_broadcast_shape :
  output_dimensions:int list ->
  int list option ->
  (int list, string) result

val pointwise :
  output_shape:Tensor_shape.t ->
  left_shape:int list option ->
  right_shape:int list option ->
  left_scalar:Ir.Scalar.t option ->
  right_scalar:Ir.Scalar.t option ->
  (bytes, string) result

val masked_fill :
  output_shape:Tensor_shape.t ->
  input_shape:Tensor_shape.t ->
  mask_shape:Tensor_shape.t ->
  Ir.Scalar.t ->
  (bytes, string) result

val batched_matmul :
  lhs_shape:Tensor_shape.t ->
  rhs_shape:Tensor_shape.t ->
  output_shape:Tensor_shape.t ->
  (bytes, string) result
