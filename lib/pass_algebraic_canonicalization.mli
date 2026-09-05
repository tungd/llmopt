(** Algebraic IR canonicalization pass based on the Affine Coordinate Category
    and Free Pointwise Applicative Functor.

    Eliminates identity casts, chained precision roundtrips, identity views,
    and duplicate movement expressions via algebraic simplification. *)

val name : string
val description : string
val pass : Pass.t
val run : Ir.Graph.t -> Ir.Graph.t
