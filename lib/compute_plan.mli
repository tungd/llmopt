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
