(** Affine Coordinate Category (The Index Monoid).

    Represents coordinate morphisms between index spaces:
      T : Z^m -> Z^n
    allowing view, reshape, transpose, slice, and broadcast operations to be
    composed categorically (T2 ∘ T1) with ZERO memory allocations, ZERO
    kernel dispatches, and ZERO synchronization barriers. *)

type t =
  | Identity
  | Permute of int list
  | Slice of { offsets : int list; strides : int list }
  | Reshape of { in_shape : int list; out_shape : int list }
  | Broadcast of { in_shape : int list; out_shape : int list }
  | Compose of t * t

val identity : t
(** The identity coordinate morphism (id_n : Z^n -> Z^n). *)

val compose : t -> t -> t
(** Morphism composition: (compose t1 t2)(j) = t2(t1(j)). *)

val simplify : t -> t
(** Simplifies composition chains by applying category laws (id ∘ T = T, T ∘ id = T). *)

val is_identity : t -> bool
(** Returns true if the morphism simplifies to Identity. *)

val equal : t -> t -> bool
(** Structural and algebraic equality between coordinate morphisms. *)

val to_string : t -> string
(** Human-readable representation of the coordinate morphism. *)

val emit_msl_index : t -> index_var:string -> shape:int list -> string
(** Emits the C++/MSL expression computing the linear source index from [index_var]. *)
