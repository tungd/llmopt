(** Pure fusion-region analysis over the shared [Compute_plan] authority.

    This module does not define a second assignment language.  An assignment
    is a provenance-bearing view of an existing [Kernel_ir.Primitive.t], and a
    region is only the immutable component of assignments connected by legal
    dependency edges. *)

module Assignment : sig
  type t

  val node_id : t -> Compute_plan.Node_id.t
  val source_node : t -> Ir.node
  val primitive : t -> Kernel_ir.Primitive.t
  val inputs : t -> Ir.Value.t list
  val outputs : t -> Ir.Value.t list
end

module Region : sig
  type t

  (** Assignments are ordered by the plan's deterministic topological order. *)
  val assignments : t -> Assignment.t list
  val member_node_ids : t -> Compute_plan.Node_id.t list
  val inputs : t -> Ir.Value.t list
  val outputs : t -> Ir.Value.t list
end

module Boundary : sig
  module Numerical : sig
    type t =
      | Shape_mismatch of Compute_plan.Node_id.t
      | Dtype_mismatch of Compute_plan.Node_id.t
      | Dimension_mismatch of Compute_plan.Node_id.t
  end

  module Resource : sig
    type t =
      | Threadgroup_size of { required : int; available : int }
      | Sram_capacity of { required : int; available : int }
  end

  type t =
    | Graph_output of Ir.Value.t
    | External_use of {
        value : Ir.Value.t;
        consumers : Compute_plan.Node_id.t list;
      }
    | Effect_state of {
        relation : Compute_plan.Edge.relation;
        source_effect : Compute_plan.Effect.t;
        destination_effect : Compute_plan.Effect.t;
      }
    | Unsupported_operation of {
        node : Compute_plan.Node_id.t;
        operation : Ir.Op.t;
      }
    | Unsupported_index_map of Compute_plan.Node_id.t
    | Missing_normalization of {
        source : Compute_plan.Node_id.t;
        destination : Compute_plan.Node_id.t;
      }
    | Numerical_incompatibility of Numerical.t
    | Target_resource of Resource.t
end

module Legality : sig
  (** [Illegal (first, rest)] is structurally nonempty. *)
  type t = Legal | Illegal of Boundary.t * Boundary.t list

  val is_legal : t -> bool
  val boundaries : t -> Boundary.t list
end

module Decision : sig
  type candidate

  type t =
    | Accepted of candidate
    | Cut of candidate * Legality.t

  val source : candidate -> Compute_plan.Node_id.t
  val destination : candidate -> Compute_plan.Node_id.t
  val relation : candidate -> Compute_plan.Edge.relation
  val candidate : t -> candidate
  val legality : t -> Legality.t
end

module Analysis : sig
  type t

  val target : t -> Target_hardware.t
  val graph : t -> Ir.Graph.t
  val assignments : t -> Assignment.t list
  val regions : t -> Region.t list
  val decisions : t -> Decision.t list
  (** The original dependency antichains remain a separate product. *)
  val stages : t -> Compute_plan.Stage.t list
end

val analyze : target:Target_hardware.t -> Compute_plan.t -> Analysis.t

(** Deterministic rewrites derived from an [Analysis.t].  The lowering pass
    never rediscovers graph adjacency: every event is backed by assignments and
    accepted dependency decisions retained by the analysis. *)
module Lowering : sig
  module Kind : sig
    type t = Linear_add | W4a16_linear_add | Gated_linear
  end

  module Event : sig
    type t

    val kind : t -> Kind.t
    val source_member_node_ids : t -> Compute_plan.Node_id.t list
    val source_assignments : t -> Assignment.t list
    val inputs : t -> Ir.Value.t list
    val outputs : t -> Ir.Value.t list
    val replacement_node_id : t -> Compute_plan.Node_id.t
    val replacement_node : t -> Ir.node
  end

  module Uncovered : sig
    type reason =
      | Cut of Legality.t
      | Unsupported_pattern
      | Overlap of Compute_plan.Node_id.t

    type t

    val kind : t -> Kind.t
    val source : t -> Compute_plan.Node_id.t
    val destination : t -> Compute_plan.Node_id.t
    val source_member_node_ids : t -> Compute_plan.Node_id.t list
    val reason : t -> reason
  end

  type t

  val graph : t -> Ir.Graph.t
  val events : t -> Event.t list
  val uncovered : t -> Uncovered.t list
end

val lower : Analysis.t -> Lowering.t
