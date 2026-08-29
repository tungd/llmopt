module Binding : sig
  type t = Computed | Runtime | Tensor_store of { key : string }

  val input_source : t -> Ir.Input_source.t option
end

module Argument : sig
  type t =
    | Node of string
    | Null
    | Ellipsis
    | Bool of bool
    | Int of int
    | Float of float
    | String of string
    | Symbol of string
    | List of t list
    | Tuple of t list
    | Mapping of (string * t) list
    | Slice of { start : t; stop : t; step : t }

  val node_references : t -> string list
end

module Node : sig
  type t
  val name : t -> string
  val op : t -> string
  val target : t -> string
  val inputs : t -> string list
  val shape : t -> int list option
  val dtype : t -> Ir.Dtype.t
  val binding : t -> Binding.t
  val arguments : t -> Argument.t list
  val keyword_arguments : t -> (string * Argument.t) list
end

type t
val binary_magic : string
val nodes : t -> Node.t list
val outputs : t -> string list
val of_binary : bytes -> (t, string) result
val of_bytes : bytes -> (t, string) result
val of_file : string -> (t, string) result
