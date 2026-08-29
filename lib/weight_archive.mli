module Dtype : sig
  type quant_type = Ir.Dtype.quant_type =
    | Q8_0
    | Q4_K
    | Q5_K
    | Q6_K
    | Q5_0
    | Q4_0
    | IQ4_XS

  type t =
    | F32
    | F16
    | BF16
    | I64
    | I32
    | I8
    | Bool
    | U8
    | Quant of quant_type

  val to_string : t -> string
  val byte_width : t -> int
  val block_size : quant_type -> int
  val bytes_per_block : quant_type -> int
  val quant_to_string : quant_type -> string
  val quant_of_string : string -> quant_type option
end

module Tensor : sig
  type t

  val create :
    name:string ->
    dtype:Dtype.t ->
    shape:int list ->
    offset:int ->
    byte_length:int ->
    t

  val name : t -> string
  val dtype : t -> Dtype.t
  val shape : t -> int list
  val offset : t -> int
  val byte_length : t -> int
end

type t

val create :
  path:string ->
  file_size:int ->
  index_bytes:int ->
  tensors:Tensor.t list ->
  t

val of_file : string -> (t, string) result
val path : t -> string
val file_size : t -> int
val index_bytes : t -> int
val tensors : t -> Tensor.t list
val find : t -> string -> Tensor.t option
