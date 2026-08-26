module Dtype : sig
  type t = F32 | F16 | BF16 | I64 | I32 | I8 | Bool | U8

  val to_string : t -> string
  val byte_width : t -> int
end

module Tensor : sig
  type t

  val name : t -> string
  val dtype : t -> Dtype.t
  val shape : t -> int list
  val offset : t -> int
  val byte_length : t -> int
end

type t

val of_file : string -> (t, string) result
val path : t -> string
val file_size : t -> int
val index_bytes : t -> int
val tensors : t -> Tensor.t list
val find : t -> string -> Tensor.t option
