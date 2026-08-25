module Linear_bias = Pass_fuse_linear_bias
module Rms_norm = Pass_fuse_rms_norm
module Rms_rope = Pass_fuse_rms_rope
module Short_conv = Pass_fuse_short_conv
module Q8_epilogues = Pass_fuse_q8_epilogues
module Dual_linear_swiglu = Pass_fuse_dual_linear_swiglu
module Qkv_linear = Pass_fuse_qkv_linear
module Short_conv_step_fused = Pass_fuse_short_conv_step
module Linear_residual_norm = Pass_fuse_linear_residual_norm
module Co_schedule = Pass_co_schedule

val fuse_linear_bias : Ir.Graph.t -> Ir.Graph.t
val fuse_rms_norm : Ir.Graph.t -> Ir.Graph.t
val fuse_rms_rope : Ir.Graph.t -> Ir.Graph.t
val fuse_short_conv : Ir.Graph.t -> Ir.Graph.t
val fuse_q8_silu : Ir.Graph.t -> Ir.Graph.t
val fuse_q8_add : Ir.Graph.t -> Ir.Graph.t
val fuse_q8_mul_add : Ir.Graph.t -> Ir.Graph.t
val fuse_dual_linear_swiglu : Ir.Graph.t -> Ir.Graph.t
val fuse_qkv_linear : Ir.Graph.t -> Ir.Graph.t
val fuse_short_conv_step : Ir.Graph.t -> Ir.Graph.t
val fuse_linear_residual_norm : Ir.Graph.t -> Ir.Graph.t
val co_schedule : Ir.Graph.t -> Ir.Graph.t

val all_passes : Pass.t list
val default_pipeline : Pass.Pipeline.t

val optimize : Ir.Graph.t -> Ir.Graph.t
