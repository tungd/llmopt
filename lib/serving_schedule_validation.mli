(* validation interface -- inferred, LOOP item 4 *)
val ( let* ) : ('a, 'b) result -> ('a -> ('c, 'b) result) -> ('c, 'b) result
val scalar_matches_dtype : Ir.Scalar.t -> Ir.Dtype.t -> bool
val q8_matrix_matches :
  rows:int -> columns:int -> dtype:Ir.Dtype.t -> Ir.Value.t -> bool
val q8_vector_matches : length:int -> dtype:Ir.Dtype.t -> Ir.Value.t -> bool
val q8_output_matches :
  rows:int -> columns:int -> dtype:Ir.Dtype.t -> Ir.Value.t -> bool
val attention_value_shape :
  key:Tensor_shape.t -> Tensor_shape.t -> (Tensor_shape.t, string) result
val q8_activation_dtype : Ir.Dtype.t -> bool
val w4a16_linear_metadata_matches :
  m:int ->
  n:int -> k:int -> bias:bool -> Ir.Value.t list -> Ir.Value.t -> bool
val w4a16_swiglu_metadata_matches :
  m:int ->
  n:int -> k:int -> epsilon:float -> Ir.Value.t list -> Ir.Value.t -> bool
val w4a16_qkv_linear_metadata_matches :
  m:int ->
  k:int ->
  n_q:int ->
  n_k:int ->
  n_v:int -> Ir.Value.t list -> Ir.Value.t -> Ir.Value.t list -> bool
val quant_qkv_linear_metadata_matches :
  m:int ->
  k:int ->
  n_q:int ->
  n_k:int ->
  n_v:int -> Ir.Value.t list -> Ir.Value.t -> Ir.Value.t list -> bool
val same_value_metadata : Ir.Value.t -> Ir.Value.t -> bool
val rotary_trig_shape_matches :
  batches:int -> tokens:int -> width:int -> Ir.Value.t -> bool
val q8_residual_metadata_matches : Ir.Value.t list -> Ir.Value.t -> bool
val fresh_outputs :
  Serving_schedule_types.Int_set.t -> Ir.Value.t list -> bool
val validate_fusion_command :
  Serving_schedule_types.Command.t ->
  Kernel_region.region -> (unit, string) result
val validate_command :
  Serving_schedule_types.Int_set.t ->
  Serving_schedule_types.Command.t -> (unit, string) result
