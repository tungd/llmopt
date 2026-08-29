(** Target-independent structured tensor SSA for model paths.

    Linear is semantic here: packed-weight layout, scale representation,
    storage spaces, launch geometry, SIMD instructions, and generated source
    belong to later lowering interfaces. *)

module Slot : sig
  type t

  val of_int : int -> (t, string) result
  val of_int_exn : int -> t
  val to_int : t -> int
  val compare : t -> t -> int
  val equal : t -> t -> bool
end

module Primitive : sig
  type unary = Silu
  type binary = Mul | Add

  (** [m] rows of activation, [n] output columns, [k] input columns. *)
  type linear = {
    m : int;
    n : int;
    k : int;
    bias : bool;
    storage : Ir.Linear_storage.layout;
  }

  type t =
    | Rms_norm of { epsilon : float }
    | Linear of linear
    | Unary of unary
    | Binary of binary

  val rms_norm : epsilon:float -> t
  val linear :
    m:int ->
    n:int ->
    k:int ->
    bias:bool ->
    storage:Ir.Linear_storage.layout ->
    t
  val unary : unary -> t
  val binary : binary -> t
  val to_string : t -> string
end

module Effect : sig
  type mode = Read | Write | Read_write
  type alias = Distinct | May_alias | Alias_of of Slot.t

  type access = {
    slot : Slot.t;
    mode : mode;
    alias : alias;
  }

  type t

  val empty : t
  val create :
    accesses:access list ->
    ?state_inputs:Slot.t list ->
    ?state_outputs:Slot.t list ->
    ?barriers:int list ->
    unit ->
    (t, string) result
  val accesses : t -> access list
  val state_inputs : t -> Slot.t list
  val state_outputs : t -> Slot.t list
  val barriers : t -> int list
end

module Scan : sig
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
end

module Resource : sig
  type t

  val zero : t
  val create :
    scalar_ops:int64 ->
    bytes_read:int64 ->
    bytes_written:int64 ->
    temporary_bytes:int64 ->
    synchronization_points:int ->
    (t, string) result
  val scalar_ops : t -> int64
  val bytes_read : t -> int64
  val bytes_written : t -> int64
  val temporary_bytes : t -> int64
  val synchronization_points : t -> int
end

type input = {
  slot : Slot.t;
  value : Ir.Value.t;
}

type output_spec = {
  slot : Slot.t;
  shape : Tensor_shape.t;
  dtype : Ir.Dtype.t;
}

type binding

val make_binding :
  outputs:output_spec list ->
  primitive:Primitive.t ->
  inputs:Slot.t list ->
  (binding, string) result

val binding_outputs : binding -> output_spec list
val binding_inputs : binding -> Slot.t list
val binding_primitive : binding -> Primitive.t
val output_slot : output_spec -> Slot.t
val output_shape : output_spec -> Tensor_shape.t
val output_dtype : output_spec -> Ir.Dtype.t

type storage = Fresh | Alias_input of Slot.t | In_place_input of Slot.t

type result = {
  slot : Slot.t;
  value : Ir.Value.t;
  storage : storage;
}

type t

type error =
  | Empty_name
  | Invalid_member_node_id of int
  | Duplicate_member_node_id of int
  | Duplicate_slot of Slot.t
  | Unknown_slot of Slot.t
  | Forward_reference of Slot.t
  | Invalid_binding of string
  | Invalid_result of string
  | Invalid_effect of string

val create :
  name:string ->
  member_node_ids:int list ->
  inputs:input list ->
  bindings:binding list ->
  results:result list ->
  effects:Effect.t ->
  resource:Resource.t ->
  (t, error) Stdlib.result

val name : t -> string
val member_node_ids : t -> int list
val inputs : t -> input list
val bindings : t -> binding list
val results : t -> result list
val effects : t -> Effect.t
val resource : t -> Resource.t

val result_slot : result -> Slot.t
val result_value : result -> Ir.Value.t
val result_storage : result -> storage

val error_to_string : error -> string
val pp : Format.formatter -> t -> unit
val to_string : t -> string
