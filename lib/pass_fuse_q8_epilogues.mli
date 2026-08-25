include Pass.PASS

val run_silu : Ir.Graph.t -> Ir.Graph.t
val run_add : Ir.Graph.t -> Ir.Graph.t
val run_mul_add : Ir.Graph.t -> Ir.Graph.t

val pass_silu : Pass.t
val pass_add : Pass.t
val pass_mul_add : Pass.t
val pass : Pass.t
