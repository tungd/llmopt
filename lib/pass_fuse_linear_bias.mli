include Pass.PASS

val pass : Pass.t

val fuse_w4a16_linear_add : Ir.Graph.t -> Ir.Graph.t

val fuse_w4a16_qkv : Ir.Graph.t -> Ir.Graph.t

val eliminate_attention_transpose : Ir.Graph.t -> Ir.Graph.t

val eliminate_gqa_expansion : Ir.Graph.t -> Ir.Graph.t
