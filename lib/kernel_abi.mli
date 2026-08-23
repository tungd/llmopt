module Operation : sig
  type t = Matmul | Fused_linear | Linear | Q8_linear | Q8_dequantize

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
  val to_yojson : t -> Yojson.Basic.t
  val of_yojson : Yojson.Basic.t -> (t, string) result
end
