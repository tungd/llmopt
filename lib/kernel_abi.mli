module Operation : sig
  type t =
    | Matmul
    | Fused_linear
    | Linear
    | W4a16_linear
    | W4a16_swiglu_ffn
    | W4a16_lm_head_argmax
    | Rms_norm
    | Rms_rope
    | Short_conv
    | Short_conv_step
    | Short_conv_prefill
    | Attention
    | Embedding
    | Arange
    | Diff
    | Cumsum
    | Fill
    | Gather2
    | Cast
    | Pointwise
    | Movement
    | Reduction
    | Update_slice
    | Cache

  val to_string : t -> string
  val of_string : string -> (t, string) result
end

module Entry : sig
  type tile = int * int * int
  type t

  val create :
    name:string ->
    operation:Operation.t ->
    input_dtype:Ir.Dtype.t ->
    output_dtype:Ir.Dtype.t ->
    threadgroup:int * int * int ->
    (t, string) result

  val create_with_tile :
    tile:tile ->
    name:string ->
    operation:Operation.t ->
    input_dtype:Ir.Dtype.t ->
    output_dtype:Ir.Dtype.t ->
    threadgroup:int * int * int ->
    (t, string) result

  val name : t -> string
  val operation : t -> Operation.t
  val input_dtype : t -> Ir.Dtype.t
  val output_dtype : t -> Ir.Dtype.t
  val threadgroup : t -> int * int * int
  val tile : t -> tile option
end
