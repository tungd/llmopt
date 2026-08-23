type t

type error =
  | Negative_dimension of { axis : int; value : int }
  | Element_count_overflow of int list
  | Matrix_projection_has_zero_elements of t
  | Matrix_projection_overflow of t
  | Invalid_axis of { rank : int; axis : int; allow_end : bool }
  | Duplicate_axis of int
  | Incompatible_broadcast of { left : t; right : t }
  | Element_count_mismatch of { source : t; target : t }
  | Invalid_expansion of { source : t; target : t }

val create : int list -> (t, error) result
val of_ints_exn : int list -> t
val scalar : t
val of_matrix : Shape.t -> t
val dimensions : t -> int list
val rank : t -> int
val numel : t -> int
val equal : t -> t -> bool
val normalize_axis : ?allow_end:bool -> t -> int -> (int, error) result
val normalize_axes : t -> int list -> (int list, error) result
val broadcast : t -> t -> (t, error) result
val reshape : t -> t -> (t, error) result
val transpose : t -> axis0:int -> axis1:int -> (t, error) result
val unsqueeze : t -> axis:int -> (t, error) result
val expand : t -> target:t -> (t, error) result
val reduce : t -> axes:int list -> keepdim:bool -> (t, error) result
val matrix : t -> (Shape.t, error) result
val matrix_exn : t -> Shape.t
val to_string : t -> string
val error_to_string : error -> string
