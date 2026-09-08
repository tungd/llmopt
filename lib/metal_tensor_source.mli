(** Metal tensor-primitive / dense Linear shader catalog interface. *)
val linear_f16_source : string
val linear_f16_entries : Kernel_abi.Entry.t list
val linear_f16_bf16_source : string
val linear_f16_bf16_entries : Kernel_abi.Entry.t list
val linear_f16_f32_source : string
val linear_f16_f32_entries : Kernel_abi.Entry.t list
val replace_first : needle:string -> replacement:string -> string -> string
val replace_all : needle:string -> replacement:string -> String.t -> String.t
val embedding_source : string
val embedding_entries : Kernel_abi.Entry.t list
val arange_source : string
val arange_entries : Kernel_abi.Entry.t list
val diff_source : string
val diff_entries : Kernel_abi.Entry.t list
val cumsum_source : string
val cumsum_entries : Kernel_abi.Entry.t list
val tensor_primitive_source : string
val tensor_primitive_entries : Kernel_abi.Entry.t list
val fill_source : string
val fill_entries : Kernel_abi.Entry.t list
val gather2_source : string
val gather2_entries : Kernel_abi.Entry.t list
val cast_source : string
val cast_entries : Kernel_abi.Entry.t list
val pointwise_binary_kernel :
  name:string ->
  input_type:string ->
  output_type:string ->
  scalar_field:string -> scalar_cast:string -> expression:string -> string
val pointwise_unary_kernel :
  name:string ->
  input_type:string -> output_type:string -> expression:string -> string
val pointwise_mixed_mul_f16_f32 : string
val pointwise_mixed_mul_f16_f32_f16 : string
val pointwise_source : string
val pointwise_entries : Kernel_abi.Entry.t list
val movement_unary_kernel :
  name:string ->
  value_type:string -> parameters:string -> body:string -> string
val movement_transpose_kernel : name:string -> value_type:string -> string
val movement_expand_kernel : name:string -> value_type:string -> string
val movement_index_kernel : name:string -> value_type:string -> string
val movement_roll_kernel : name:string -> value_type:string -> string
val movement_concat_kernel : name:string -> value_type:string -> string
val movement_parameters_source : string
val movement_source : string
val movement_entries : Kernel_abi.Entry.t list
val reduction_source : string
val reduction_entries : Kernel_abi.Entry.t list
val update_slice_kernel_source : name:string -> value_type:string -> string
val update_slice_source : string
val update_slice_entries : Kernel_abi.Entry.t list
