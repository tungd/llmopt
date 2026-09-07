(** Kernel scan recovery and fusion over [Ir.Graph]: carried-update
  chain discovery and structurally matched triangular recurrence
  fusion.  Split out of Kernel_ir; see LOOP. *)

type iteration = {
  index : int;
  member_node_ids : int list;
  state_input : Ir.Value.t;
  state_output : Ir.Value.t;
  body_inputs : Ir.Value.t list;
}

type t

val create :
  name:string ->
  axis:int ->
  iterations:iteration list ->
  sequence_inputs:Ir.Value.t list ->
  stacked_outputs:Ir.Value.t list ->
  (t, string) result

val name : t -> string
val axis : t -> int
val trip_count : t -> int
val iterations : t -> iteration list
val sequence_inputs : t -> Ir.Value.t list
val stacked_outputs : t -> Ir.Value.t list
val initial_state : t -> Ir.Value.t
val final_state : t -> Ir.Value.t
val member_node_ids : t -> int list
val to_string : t -> string

(** Recover maximal, consecutively indexed carried-update chains. *)
val recover : Ir.Graph.t -> t list

(** Fuse structurally matched FP32 triangular recurrences whose square state
    fits the selected kernel family's width. *)
val fuse_triangular_recurrences :
  max_width:int -> Ir.Graph.t -> Ir.Graph.t
