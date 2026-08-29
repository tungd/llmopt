module Value : sig
  type t =
    | Uint8 of int
    | Int8 of int
    | Uint16 of int
    | Int16 of int
    | Uint32 of int32
    | Int32 of int32
    | Float32 of float
    | Bool of bool
    | String of string
    | Array of t list
    | Uint64 of int64
    | Int64 of int64
    | Float64 of float

  val to_string : t -> string
  val to_int_opt : t -> int option
  val to_string_opt : t -> string option
  val to_float_opt : t -> float option
end

module Tensor_info : sig
  type t = {
    name : string;
    shape : int list;
    ggml_type : int;
    dtype : Weight_archive.Dtype.t;
    offset : int64;
    byte_length : int;
  }

  val name : t -> string
  val shape : t -> int list
  val ggml_type : t -> int
  val dtype : t -> Weight_archive.Dtype.t
  val offset : t -> int64
  val byte_length : t -> int
end

type t = {
  version : int;
  alignment : int;
  metadata : (string * Value.t) list;
  tensors : Tensor_info.t list;
  data_offset : int64;
  file_size : int64;
}

val of_file : string -> (t, string) result
val of_bytes : bytes -> (t, string) result
val find_metadata : t -> string -> Value.t option
val find_string : t -> string -> string option
val find_int : t -> string -> int option
val find_float : t -> string -> float option
val find_tensor : t -> string -> Tensor_info.t option

val architecture : t -> string option
val context_length : t -> int option
val embedding_length : t -> int option
val block_count : t -> int option
val feed_forward_length : t -> int option
val head_count : t -> int option
val head_count_kv : t -> int option
val chat_template : t -> string option

module Transcode : sig
  val iq4_xs_to_q5_k : bytes -> (bytes, string) result
end
