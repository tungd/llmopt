val fuse_linear_bias : Ir.Graph.t -> Ir.Graph.t
val fuse_rms_norm : Ir.Graph.t -> Ir.Graph.t
val fuse_rms_rope : Ir.Graph.t -> Ir.Graph.t
val fuse_q8_silu : Ir.Graph.t -> Ir.Graph.t
val fuse_q8_add : Ir.Graph.t -> Ir.Graph.t
val fuse_q8_mul_add : Ir.Graph.t -> Ir.Graph.t
val optimize : Ir.Graph.t -> Ir.Graph.t
