module Tensor : sig
  type t
  val create : Shape.t -> t
  val of_rows : float array array -> t
  val get : t -> int -> int -> float
  val set : t -> int -> int -> float -> unit
  val fill : t -> float -> unit
  val shape : t -> Shape.t
  val to_rows : t -> float array array
end

type execution
val output : execution -> string -> Tensor.t option
val outputs : execution -> (string * Tensor.t) list
val run : inputs:(string * Tensor.t) list -> (unit -> 'a) -> (('a * execution), exn) result
