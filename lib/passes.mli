module Linear_bias = Pass_fuse_linear_bias
module Rms_norm = Pass_fuse_rms_norm
module Rms_rope = Pass_fuse_rms_rope
module Short_conv = Pass_fuse_short_conv
module Short_conv_step_fused = Pass_fuse_short_conv_step
module Gated_delta = Pass_fuse_gated_delta
module Lm_head_argmax = Pass_fuse_lm_head_argmax
module Co_schedule = Pass_co_schedule
module Epilogues = Pass_fuse_epilogues

val fuse_linear_bias : Ir.Graph.t -> Ir.Graph.t
val fuse_rms_norm : Ir.Graph.t -> Ir.Graph.t
val fuse_rms_rope : Ir.Graph.t -> Ir.Graph.t
val fuse_short_conv : Ir.Graph.t -> Ir.Graph.t
val discover_swiglu_ffn : Ir.Graph.t -> (Kernel_ir.t list, string) result
val recover_scans : Ir.Graph.t -> Kernel_ir.Scan.t list
val fuse_swiglu_ffn : Ir.Graph.t -> (Ir.Graph.t, string) result
val fuse_short_conv_step : Ir.Graph.t -> Ir.Graph.t
val fuse_gated_delta : Ir.Graph.t -> Ir.Graph.t
val fuse_lm_head_argmax : Ir.Graph.t -> Ir.Graph.t
val parameterize_token_id_output : Ir.Graph.t -> (Ir.Graph.t, string) result
val fuse_epilogues : Ir.Graph.t -> Ir.Graph.t
val co_schedule : Ir.Graph.t -> Ir.Graph.t
val co_schedule_plan : Compute_plan.t -> Ir.Graph.t

(** Graph rewrites whose operations all have an executable backend lowering. *)
val semantic_passes : Pass.t list
val default_pipeline : Pass.Pipeline.t

module Queue_stage : sig
  type kind = Sequential | Concurrent
  type t

  (** Zero-based stage index in the AOT command queue. *)
  val index : t -> int

  (** A stage is concurrent when its independent requests share one antichain. *)
  val kind : t -> kind

  (** The immutable compute requests represented by this AOT stage. *)
  val requests : t -> Compute_plan.Request.t list
end

module Receipt : sig
  module Legacy_matcher : sig
    type t = Quant_linear_add | W4a16_linear_add | Quant_gated_linear
  end

  module Dependency : sig
    type relation = Compute_plan.Edge.relation
    type t

    val source : t -> Compute_plan.Node_id.t
    val destination : t -> Compute_plan.Node_id.t
    val relation : t -> relation
  end

  module Compiler_barrier : sig
    type kind = Create | Arrive | Wait
    type t

    val node_id : t -> int
    val id : t -> int
    val kind : t -> kind
  end

  module Linear_selection : sig
    type t

    val node_id : t -> Compute_plan.Node_id.t
    val operation : t -> Ir.Op.t
    val chosen : t -> Kernel_abi.Linear_tactic.Candidate.t
    val alternatives : t -> Kernel_abi.Linear_tactic.Candidate.t list
  end

  type t

  val target : t -> Target_hardware.t
  val decisions : t -> Constructive_lowering.Decision.t list
  val events : t -> Constructive_lowering.Lowering.Event.t list
  val uncovered : t -> Constructive_lowering.Lowering.Uncovered.t list
  val unified_count : t -> int
  val accepted_count : t -> int
  val cut_count : t -> int
  val legacy_count : t -> int
  val legacy_counts : t -> (Legacy_matcher.t * int) list
  val linear_selections : t -> Linear_selection.t list
  val kernels : t -> Kernel_abi.Entry.t list
  val queue_stages : t -> Queue_stage.t list
  val dependencies : t -> Dependency.t list
  val barriers : t -> Compiler_barrier.t list

  (** Stable compact JSON; object fields and arrays are emitted in a fixed order. *)
  val to_yojson : t -> Yojson.Basic.t
  val to_json_string : t -> string

  (** Write one deterministic JSON document with exactly one trailing newline. *)
  val write_file : string -> t -> (unit, string) result
end

module Optimization : sig
  module Legacy_fallback : sig
    type matcher = Quant_linear_add | W4a16_linear_add | Quant_gated_linear
    type t

    val matcher : t -> matcher
    val invocations : t -> int
    val name : matcher -> string
  end

  type t

  (** Hardware profile retained by this optimization snapshot. *)
  val target : t -> Target_hardware.t

  (** Canonical optimized graph, before execution barriers are inserted. *)
  val semantic_graph : t -> Ir.Graph.t

  (** Backend-neutral dependency plan for [semantic_graph]. *)
  val plan : t -> Compute_plan.t

  (** Structured fusion candidates; these do not mutate [semantic_graph]. *)
  val fusion_regions : t -> Kernel_ir.t list

  (** Immutable dependency facts used to drive covered-region lowering. *)
  val region_analysis : t -> Constructive_lowering.Analysis.t

  (** Constructive rewrites and typed uncovered candidates. *)
  val lowering : t -> Constructive_lowering.Lowering.t

  (** Named legacy matchers invoked for explicitly uncovered covered families. *)
  val legacy_fallbacks : t -> Legacy_fallback.t list

  (** Typed carried-state regions recovered from statically unrolled updates. *)
  val scan_regions : t -> Kernel_ir.Scan.t list

  (** Executable rewrite suffix followed by plan-based co-scheduling. *)
  val execution_graph : t -> Ir.Graph.t

  (** The pre-barrier plan used to construct the AOT antichain queue. *)
  val execution_plan : t -> Compute_plan.t

  (** Build a receipt from the already-emitted kernel ABI entries.  The
      pipeline supplies its cache-augmented entries; this keeps receipt
      assembly pure and does not make optimization depend on backend lowering. *)
  val receipt :
    kernels:Kernel_abi.Entry.t list -> t -> Receipt.t
end

val optimize : ?target:Target_hardware.t -> Ir.Graph.t -> (Optimization.t, string) result
