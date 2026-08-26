(** Small tree-shaped queries over the semantic graph.

    Queries follow producer edges, but the graph itself remains a DAG.  A
    repeated capture therefore means exact value identity, not a copied tree
    node. *)

module Capture : sig
  type t

  val of_string : string -> (t, string) result
  val of_string_exn : string -> t
  val to_string : t -> string
  val compare : t -> t -> int
end

module Shape : sig
  type dim = Any | Exact of int | Var of string
  type t = Any_shape | Rank of int | Dims of dim list
end

type operation =
  | Any_operation
  | W4a16_linear
  | Rms_norm
  | Silu
  | Mul
  | Add

type effect_kind = Pure | Stateful | Synchronizing | Opaque | Any_effect

type use_count = Exactly of int | At_most of int | At_least of int

type predicate =
  | Op of operation
  | Shape of Capture.t * Shape.t
  | Dtype of Capture.t * Ir.Dtype.t
  | Effect of effect_kind
  | Uses of Capture.t * use_count

type pattern =
  | Node of node_pattern
  | Or of pattern list
  | Optional of pattern

and node_pattern = {
  op : operation;
  inputs : input_pattern list;
  outputs : output_pattern list;
  predicates : predicate list;
  capture : Capture.t option;
}

and input_pattern =
  | Any_input
  | Capture_input of Capture.t
  | Produced_by of pattern

and output_pattern = Ignore_output | Capture_output of Capture.t

module Sexp : sig
  val parse : string -> (pattern, string) result
end

module Match : sig
  type value = Node of Ir.node | Tensor of Ir.Value.t
  type t

  val root : t -> Ir.node
  val capture : Capture.t -> t -> value option
  val member_node_ids : t -> int list
end

val match_graph : pattern -> Ir.Graph.t -> Match.t list

val construct_region :
  name:string ->
  match_:Match.t ->
  inputs:Kernel_ir.input list ->
  bindings:Kernel_ir.binding list ->
  results:Kernel_ir.result list ->
  effects:Kernel_ir.Effect.t ->
  resource:Kernel_ir.Resource.t ->
  (Kernel_ir.t, string) result

module Rule : sig
  type t

  val create :
    pattern:pattern ->
    result_captures:Capture.t list ->
    emit:(Match.t -> (Kernel_ir.t, string) result) ->
    t

  val apply : t -> Ir.Graph.t -> (Kernel_ir.t list, string) result

  (** Replace each non-overlapping match at its root node.  The callback maps
      the validated match and emitted region to the executable operation and
      its external inputs.  Rewriting fails if an intermediate result escapes
      the matched region. *)
  val rewrite :
    t ->
    lower:(Match.t -> Kernel_ir.t -> (Ir.Op.t * Ir.Value.t list, string) result) ->
    Ir.Graph.t ->
    (Ir.Graph.t, string) result
end
