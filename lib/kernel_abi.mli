module Operation : sig
  type t =
    | Matmul
    | Fused_linear
    | Linear
    | Q8_linear
    | Q8_linear_silu
    | Q8_linear_add
    | Q8_dequantize
    | Rms_norm
    | Short_conv
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
  type t

  val create :
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
end
