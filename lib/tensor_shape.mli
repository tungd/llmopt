type t

module Index : sig
  module Spec : sig
    type t =
      | At of int
      | Slice of {
          start : int option;
          stop : int option;
          step : int option;
        }
      | New_axis
      | Ellipsis
  end

  type selector =
    | At of int
    | Slice of {
        start : int;
        step : int;
        length : int;
      }
    | New_axis

  type t

  val of_selectors : selector list -> (t, string) result
  val selectors : t -> selector list
  val to_string : t -> string
end

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
  | Multiple_ellipsis
  | Too_many_indices of { rank : int; consumed : int }
  | Index_out_of_bounds of { axis : int; dimension : int; index : int }
  | Zero_slice_step
  | Empty_concat
  | Concat_rank_mismatch of { expected : int; actual : int }
  | Concat_dimension_mismatch of {
      axis : int;
      expected : int;
      actual : int;
    }
  | Concat_dimension_overflow of int
  | Invalid_chunk_count of int
  | Invalid_depthwise_conv1d of string
  | Convolution_dimension_overflow
  | Malformed_index of string

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
val depthwise_conv1d :
  t ->
  t ->
  stride:int ->
  padding:int ->
  dilation:int ->
  groups:int ->
  (t, error) result
val reduce : t -> axes:int list -> keepdim:bool -> (t, error) result
val index : t -> Index.Spec.t list -> (Index.t * t, error) result
val apply_index : t -> Index.t -> (t, error) result
val concat : t list -> axis:int -> (t, error) result
val chunk : t -> chunks:int -> axis:int -> ((Index.t * t) list, error) result
val matrix : t -> (Shape.t, error) result
val matrix_exn : t -> Shape.t
val to_string : t -> string
val error_to_string : error -> string
