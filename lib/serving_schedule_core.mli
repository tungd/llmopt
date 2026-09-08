(* core interface -- inferred, LOOP item 4 *)
module Command = Serving_schedule_types.Command
module Stage = Serving_schedule_types.Stage
module Tensor_input = Serving_schedule_types.Tensor_input
module Int_set = Serving_schedule_types.Int_set
val value_id : Ir.Value.t -> int
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
type t = { commands : Command.t list; }
val commands : t -> Command.t list
val stages : t -> Stage.t list
val create : Command.t list -> (t, string) result
val attach_fusion_region :
  Command.t list -> Kernel_region.t -> (Command.t list, string) result
val is_gpu_command_in_schedule : Command.t -> bool
val prune_redundant_barriers : Command.t list -> Command.t list
val of_graph :
  ?fusion_regions:Kernel_region.t list -> Ir.Graph.t -> (t, string) result
val tensor_inputs : t -> Tensor_input.t list
val runtime_inputs : t -> (string * Ir.Value.t) list
val opaque_count : t -> int
