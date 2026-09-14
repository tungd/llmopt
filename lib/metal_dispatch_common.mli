module Quant_qkv_tactic : sig
  val kernel_name :
    q_dtype:Ir.Dtype.t ->
    experimental_llama_mv:bool ->
    experimental_mixed_llama_mv:bool ->
    m:int ->
    v_dtype:Ir.Dtype.t ->
    (string, string) result
end

module Paged_attention_tactic : sig
  val kernel_name :
    experimental_gqa3_split8:bool ->
    split8:bool ->
    query_heads:int ->
    kv_heads:int ->
    head_dimension:int ->
    string

  val groups_by_kv_head : string -> bool
end

val round_up : int -> int -> (int, string) result
val linear_f16_grid : int -> (int, string) result
val short_row_quant_grid : int -> (int, string) result
val gated_delta_grid : int -> (int, string) result
val simd_rows_grid : int -> (int, string) result
val wide_rows_grid : int -> (int, string) result
val use_wide_rows : rows:int -> width:int -> bool

val rotary_trig_dimensions : Ir.Value.t -> (int * int * int) option

val rotary_trig_batches :
  batches:int ->
  tokens:int ->
  width:int ->
  Ir.Value.t ->
  Ir.Value.t ->
  (int, string) result

val split_axis : Tensor_shape.t -> int -> (int * int * int, string) result

val rms_norm_kernel_name :
  wide:bool -> Ir.Dtype.t -> Ir.Dtype.t -> (string, string) result

val legacy_rms_norm_kernel_name : Ir.Dtype.t -> Ir.Dtype.t -> string option

val select_rms_norm_kernel :
  Metal_native.runtime ->
  wide:bool ->
  Ir.Dtype.t ->
  Ir.Dtype.t ->
  Ir.Dtype.t ->
  (Kernel_abi.Entry.t * bool, string) result

val select_attention_kernel :
  Metal_native.runtime ->
  Ir.Dtype.t ->
  Ir.Dtype.t ->
  int ->
  (Kernel_abi.Entry.t * bool, string) result

val validate_linear_shapes :
  m:int ->
  n:int ->
  k:int ->
  Ir.Value.t ->
  Ir.Value.t ->
  Ir.Value.t ->
  (unit, string) result

val select_quant_linear_kernel :
  ?suffix:string ->
  Metal_native.runtime ->
  m:int ->
  n:int ->
  k:int ->
  output_dtype:Ir.Dtype.t ->
  Ir.Dtype.quant_type ->
  (Kernel_abi.Entry.t, string) result

val same_value_metadata : Ir.Value.t -> Ir.Value.t -> bool

val macro_kernel_name :
  Ir.Dtype.t -> base:string -> has_bias:bool -> (string, string) result

val dispatch_output :
  ?name:string ->
  ?batch:Metal_native.Batch.t ->
  Metal_native.runtime ->
  Metal_execution_state.t ->
  Ir.Value.t ->
  operation:Kernel_abi.Operation.t ->
  input_dtype:Ir.Dtype.t ->
  buffers:Metal_native.Buffer.t list ->
  parameters:bytes ->
  grid:int * int * int ->
  (Metal_execution_state.t * string list, string) result

val dispatch_quant_linear :
  ?batch:Metal_native.Batch.t ->
  Metal_native.runtime ->
  Metal_execution_state.t ->
  Ir.Value.t ->
  m:int ->
  n:int ->
  k:int ->
  quant:Ir.Dtype.quant_type ->
  input_buffer:Metal_native.Buffer.t ->
  weight_buffer:Metal_native.Buffer.t ->
  additive_buffer:Metal_native.Buffer.t ->
  epilogue_mode:int ->
  (Metal_execution_state.t * string list, string) result
