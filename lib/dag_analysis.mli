module Node_set : Set.S with type elt = int
module Node_map : Map.S with type key = int

type t = {
  graph : Ir.Graph.t;
  nodes : Ir.node list;
  node_map : Ir.node Node_map.t;
  predecessors : Node_set.t Node_map.t;
  successors : Node_set.t Node_map.t;
  topological_order : int list;
  topological_levels : int list list;
  heights : int Node_map.t;
  depths : int Node_map.t;
  critical_path_length : int;
}

type critical_path = {
  length : int;
  nodes : Ir.node list;
}

type antichain = {
  level : int;
  nodes : Ir.node list;
}

val analyze : Ir.Graph.t -> (t, string) result

val node_count : t -> int
val topological_order : t -> Ir.node list
val topological_levels : t -> antichain list
val predecessors : t -> int -> Node_set.t
val successors : t -> int -> Node_set.t
val critical_path : t -> critical_path
val critical_path_slack : t -> int -> int
val is_antichain : t -> int list -> bool
val extract_antichains : t -> antichain list
