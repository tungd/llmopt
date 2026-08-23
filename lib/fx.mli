module Binding : sig
  type t = Computed | Runtime | Tensor_store of { key : string }

  val input_source : t -> Ir.Input_source.t option
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
end

type t
val nodes : t -> Node.t list
val outputs : t -> string list
val of_json : Yojson.Basic.t -> (t, string) result
val of_file : string -> (t, string) result
