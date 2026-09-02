include Pass.PASS

val pass : Pass.t

val fuse_w4a16_linear_add : Ir.Graph.t -> Ir.Graph.t
val fuse_quant_linear_add : Ir.Graph.t -> Ir.Graph.t
val fuse_quant_gated_linear : Ir.Graph.t -> Ir.Graph.t

val fuse_w4a16_qkv : Ir.Graph.t -> Ir.Graph.t
val fuse_quant_qkv : Ir.Graph.t -> Ir.Graph.t

(** Canonicalize attention operands and results by composing typed logical-axis
    provenance across movement primitives. *)
val canonicalize_layouts : Ir.Graph.t -> Ir.Graph.t
