module Node : sig
  type t
  val name : t -> string
  val op : t -> string
  val target : t -> string
  val inputs : t -> string list
  val shape : t -> int list option
  val dtype : t -> Ir.Dtype.t
end

type t
val nodes : t -> Node.t list
val outputs : t -> string list
val of_json : Yojson.Basic.t -> (t, string) result
val of_file : string -> (t, string) result
