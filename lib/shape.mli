type t

type error =
  | Non_positive_dimension of string * int
  | Not_matrix_multipliable of t * t
  | Not_broadcastable of t * t

type broadcast = Same | Row

val create : rows:int -> cols:int -> (t, error) result
val of_ints_exn : rows:int -> cols:int -> t
val rows : t -> int
val cols : t -> int
val equal : t -> t -> bool
val matmul : t -> t -> (t, error) result
val add : t -> t -> ((t * broadcast), error) result
val error_to_string : error -> string
val to_string : t -> string
