(** Free Pointwise Applicative Functor.

    Represents un-evaluated scalar expression trees over coordinate spaces.
    Pointwise operations form a Reader Applicative:
      F(A) = Coord -> Scalar
    obeying Functor and Applicative laws:
      fmap id = id (cancelling identity casts)
      fmap (f . g) = fmap f . fmap g
      lift2 (op) expr1 expr2
    allowing arbitrary pointwise DAGs to be folded into register evaluation
    with zero intermediate DRAM memory allocations. *)

type unop =
  | Neg
  | Abs
  | Silu
  | Gelu
  | Exp
  | Log
  | Rsqrt
  | Cast of Ir.Dtype.t

type binop =
  | Add
  | Sub
  | Mul
  | Div
  | Min
  | Max

type triop =
  | Clamp
  | Where

type 'var t =
  | Slot of 'var
  | Const of float
  | Read of { value : Ir.Value.t; coord : Coordinate_map.t }
  | Unop of unop * 'var t
  | Binop of binop * 'var t * 'var t
  | Triop of triop * 'var t * 'var t * 'var t

val pure : float -> 'var t
(** Embeds a constant scalar into the applicative functor (pure x = λi. x). *)

val fmap : unop -> 'var t -> 'var t
(** Applies a unary operator, eliding identity casts ([fmap id = id]) at
    construction.  All other constructors ([lift2], [lift3], [fmap]) build raw
    trees; run [simplify] explicitly when normalized terms are required. *)

val lift2 : binop -> 'var t -> 'var t -> 'var t
(** Combines two applicative expressions via a binary operator. *)

val lift3 : triop -> 'var t -> 'var t -> 'var t -> 'var t
(** Combines three applicative expressions via a ternary operator. *)

val subst : ('a -> 'b t) -> 'a t -> 'b t
(** Capture-avoiding monadic bind / variable substitution. *)

val simplify : 'var t -> 'var t
(** Applies algebraic reduction rules (identity casts, zero add, unit mul,
    constant folding).  [simplify] is idempotent.  Soundness contract: rules
    such as [x * 0.0 -> 0.0] and constant [log]/[rsqrt] folding interpret the
    expression over finite reals, so callers must feed values that are
    IEEE-sound under those rules (as the property tests do); no NaN/inf
    semantics are intended. *)

val equal : ('a -> 'a -> bool) -> 'a t -> 'a t -> bool
(** Algebraic equality between expression trees: both sides are [simplify]ed
    before structural comparison, so equal terms with different raw trees
    (e.g. [Add x (Const 0.0)] vs [x]) compare equal. *)

val required_values : 'var t -> Ir.Value.t list
(** Returns all unique tensor inputs required to evaluate this expression
    tree, in stable first-syntactic-occurrence (left-to-right pre-order)
    order, deduplicated by [Value_id].  This order is what device-buffer
    binding derives its slot numbering from. *)

val to_string : ('a -> string) -> 'a t -> string
(** Human-readable representation of the expression tree. *)

val map_reads : rename:(Ir.Value.t -> Ir.Value.t) -> 'var t -> 'var t
(** Transforms all [Read] tensor values in the expression using [rename]. *)

val canonicalize : 'var t -> 'var t
(** Canonicalizes tensor read references in the expression to 0-based ordinals
    according to their first syntactic occurrence ([required_values]). *)

val emit_msl_expr :
  ?value_var:(Ir.Value.t -> string) ->
  slot_emitter:('a -> string) ->
  index_expr:string ->
  'a t ->
  string
(** Emits the C++/MSL arithmetic expression string. *)
