module Linear_bias = Pass_fuse_linear_bias
module Rms_norm = Pass_fuse_rms_norm
module Rms_rope = Pass_fuse_rms_rope
module Short_conv = Pass_fuse_short_conv
module Short_conv_step_fused = Pass_fuse_short_conv_step
module Lm_head_argmax = Pass_fuse_lm_head_argmax
module Co_schedule = Pass_co_schedule

val fuse_linear_bias : Ir.Graph.t -> Ir.Graph.t
val fuse_rms_norm : Ir.Graph.t -> Ir.Graph.t
val fuse_rms_rope : Ir.Graph.t -> Ir.Graph.t
val fuse_short_conv : Ir.Graph.t -> Ir.Graph.t
val discover_swiglu_ffn : Ir.Graph.t -> (Kernel_ir.t list, string) result
val fuse_short_conv_step : Ir.Graph.t -> Ir.Graph.t
val fuse_lm_head_argmax : Ir.Graph.t -> Ir.Graph.t
val co_schedule : Ir.Graph.t -> Ir.Graph.t
val co_schedule_plan : Compute_plan.t -> Ir.Graph.t

(** Graph rewrites whose operations all have an executable backend lowering. *)
val semantic_passes : Pass.t list
val default_pipeline : Pass.Pipeline.t

module Optimization : sig
  type t

  (** Canonical optimized graph, before execution barriers are inserted. *)
  val semantic_graph : t -> Ir.Graph.t

  (** Backend-neutral dependency plan for [semantic_graph]. *)
  val plan : t -> Compute_plan.t

  (** Structured fusion candidates; these do not mutate [semantic_graph]. *)
  val fusion_regions : t -> Kernel_ir.t list

  (** Executable rewrite suffix followed by plan-based co-scheduling. *)
  val execution_graph : t -> Ir.Graph.t
end

val optimize : Ir.Graph.t -> (Optimization.t, string) result
