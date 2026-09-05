(** Affine Coordinate Category (The Index Monoid).

    Represents coordinate morphisms between index spaces:
      T : Z^m -> Z^n
    allowing view, reshape, transpose, slice, and broadcast operations to be
    composed categorically (T2 ∘ T1) with ZERO memory allocations, ZERO
    kernel dispatches, and ZERO synchronization barriers.

    [type t] is abstract: every morphism is validated at construction by the
    smart constructors, so invalid states (non-permutations, rank or element
    count mismatches, negative offsets, non-positive strides) are
    unrepresentable. *)

type t

val identity : t
(** The identity coordinate morphism (id_n : Z^n -> Z^n). *)

val permute : int list -> t
(** [permute p] is the rank-preserving morphism reading source axis [i] from
    output axis [p.(i)]; e.g. [permute [1; 0]] is the 2-D transpose.  [p] must
    be a permutation of [0; length p - 1] (raises [Invalid_argument]
    otherwise).  An identity permutation normalizes to [identity]. *)

val slice : offsets:int list -> strides:int list -> t
(** [slice ~offsets ~strides] keeps every source element at
    [offsets.(i) + k * strides.(i)] along axis [i].  [offsets] and [strides]
    must have equal, non-zero length with non-negative offsets and positive
    strides (raises [Invalid_argument] otherwise).  A single-axis zero
    offset / unit stride slice normalizes to [identity]. *)

val reshape : in_shape:int list -> out_shape:int list -> t
(** [reshape ~in_shape ~out_shape] is the element-count-preserving reindexing
    from [in_shape] to [out_shape].  Shapes must be non-empty with positive
    dimensions and equal products (raises [Invalid_argument] otherwise).  An
    identical reshape normalizes to [identity]. *)

val broadcast : in_shape:int list -> out_shape:int list -> t
(** [broadcast ~in_shape ~out_shape] widens an input shape to an output shape
    of equal rank where every axis is either unchanged or expanded from size
    1.  Raises [Invalid_argument] for rank mismatches or non-expansive axes.  A
    no-op broadcast normalizes to [identity]. *)

val compose : t -> t -> t
(** Morphism composition: (compose t1 t2)(j) = t2(t1(j)). *)

val simplify : t -> t
(** Simplifies composition chains by applying category laws (id ∘ T = T, T ∘ id = T)
    and normalizing composition of adjacent permutations. *)

val is_identity : t -> bool
(** Returns true if the morphism simplifies to Identity. *)

val equal : t -> t -> bool
(** Algebraic equality between coordinate morphisms (compares simplified
    forms). *)

val to_string : t -> string
(** Human-readable representation of the coordinate morphism. *)

val emit_msl_index : t -> index_var:string -> shape:int list -> string
(** Emits the C++/MSL expression computing the linear source index from
    [index_var].  Only morphisms resolvable to a single closed-form linear
    index expression are supported (identity, the 2-D transpose permutation
    [permute [1; 0]], and single-axis slices).  Any other morphism raises
    [Invalid_argument] so an unresolved morphism fails loudly at codegen time
    instead of silently emitting an identity index. *)
