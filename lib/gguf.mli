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
  val tile32_affine4_block_bytes : int
  val tile32_affine8_block_bytes : int

  val iq4_xs_to_q5_k : bytes -> (bytes, string) result

  val q4_0_to_tile32_affine4 : bytes -> (bytes, string) result
  val q4_k_to_tile32_affine4 : bytes -> (bytes, string) result
  val q8_0_to_tile32_affine8 : bytes -> (bytes, string) result
  val q5_0_to_tile32_affine8 : bytes -> (bytes, string) result
  val q5_k_to_tile32_affine8 : bytes -> (bytes, string) result
  val q6_k_to_tile32_affine8 : bytes -> (bytes, string) result
  val iq4_xs_to_tile32_affine8 : bytes -> (bytes, string) result

  val to_tile32_affine :
    Weight_archive.Dtype.quant_type ->
    bytes ->
    ([ `Affine4 of bytes | `Affine8 of bytes ], string) result

  val dequantize_tile32_affine4 : bytes -> (float array, string) result
  val dequantize_tile32_affine8 : bytes -> (float array, string) result
  val dequantize_gguf :
    Weight_archive.Dtype.quant_type -> bytes -> (float array, string) result

  val fwht32 : float array -> (float array, string) result
  val quantize_tile32_affine8 : float array -> (bytes, string) result
  val quantize_tile32_affine4 : float array -> (bytes, string) result

  val bake_tensor_to_tile32_affine :
    ?rotate_fwht32:bool ->
    ?prefer_4bit:bool ->
    Weight_archive.Dtype.quant_type ->
    bytes ->
    (Weight_archive.Dtype.quant_type * bytes, string) result
end

val to_weight_archive : t -> path:string -> (Weight_archive.t, string) result
val of_file_as_archive : string -> (Weight_archive.t, string) result
val write_tile32_affine_archive :
  ?rotate_fwht32:bool ->
  ?prefer_4bit:bool ->
  source_path:string ->
  destination_path:string ->
  unit ->
  (Weight_archive.t, string) result
