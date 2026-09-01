(** A backend-neutral, immutable view of an [Ir.Graph].

    The plan keeps graph requests separate from execution concerns.  Its
    dependency edges describe values and mutable state; stages are antichains
    of requests that can be considered applicatively (the conservative
    policy puts every non-pure request in a singleton stage).
*)

module Node_id : sig
  type t = int

  val compare : t -> t -> int
  val to_int : t -> int
end

module Node_set : Set.S with type elt = Node_id.t

module Effect : sig
  type t = Pure | Stateful | Synchronizing | Opaque

  val to_string : t -> string
end

module Dependency : sig
  type t = Data of Node_id.t | State of Node_id.t

  val node : t -> Node_id.t
  val is_data : t -> bool
  val is_state : t -> bool
end

module Use : sig
  type t

  (** One distinct consumer relationship for a value in this plan.  A value
      mentioned more than once by one request contributes one [Use.t]. *)
  val value : t -> Ir.Value.t
  val producer : t -> Node_id.t
  val consumer : t -> Node_id.t
end

module Edge : sig
  type relation = Data | State | Data_and_state
  type t

  (** A relation is [Data_and_state] when one source and destination share
      both dependency facts; this is not represented as two competing edges. *)
  val source : t -> Node_id.t
  val destination : t -> Node_id.t
  val relation : t -> relation
end

module Request : sig
  type t

  val id : t -> Node_id.t
  val node : t -> Ir.node
  val op : t -> Ir.Op.t
  val inputs : t -> Ir.Value.t list
  (** Ordered primary and secondary semantic results. *)
  val outputs : t -> Ir.Value.t list
  val effect_kind : t -> Effect.t
  val dependencies : t -> Dependency.t list
  val data_dependencies : t -> Node_id.t list
  val state_dependencies : t -> Node_id.t list
end

module Stage : sig
  type t

  (** Zero-based topological stage number. *)
  val index : t -> int

  (** Requests in this stage have no dependency edge between them. *)
  val requests : t -> Request.t list
end

type t

(** Snapshot and analyze a graph.  The returned plan does not share the
    mutable node/output lists of the supplied graph. *)
val of_graph : Ir.Graph.t -> (t, string) result

val graph : t -> Ir.Graph.t
val request : t -> Node_id.t -> Request.t option
val requests : t -> Request.t list
val topological_order : t -> Request.t list
val stages : t -> Stage.t list
val predecessors : t -> Node_id.t -> Node_set.t
val successors : t -> Node_id.t -> Node_set.t
val critical_path_length : t -> int

(** The unique request that produces [value], or [None] when [value] is not
    part of this plan. *)
val producer : t -> Ir.Value.t -> Node_id.t option

(** Distinct consumers of [value], ordered by this plan's deterministic
    topological order.  Duplicate references to [value] in one request do not
    inflate this list or [use_count]. *)
val consumers : t -> Ir.Value.t -> Use.t list

val use_count : t -> Ir.Value.t -> int

(** The aggregate relation from [source] to [destination], if any. *)
val edge : t -> source:Node_id.t -> destination:Node_id.t -> Edge.t option

(** Whether a strict directed path exists.  Unknown nodes and a node's
    zero-length path to itself return [false]. *)
val has_path : t -> source:Node_id.t -> destination:Node_id.t -> bool
