(** Target-independent structured tensor SSA for quantized model paths.

    The quantized linear primitive is deliberately concrete: it means
    symmetric packed W4 weights with FP16 activations and FP16 scales in
    groups of 64.  Storage spaces, launch geometry, SIMD instructions, and
    generated source belong to a later lowering. *)

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

  (** [m] rows of activation, [n] output columns, [k] input columns.
      Weight input order is activation, packed W4 weight, FP16 group scales.
      The group size is fixed at 64 by this closed primitive. *)
  type w4a16_linear = {
    m : int;
    n : int;
    k : int;
    bias : bool;
  }

  type t =
    | Rms_norm of { epsilon : float }
    | W4a16_linear of w4a16_linear
    | Unary of unary
    | Binary of binary

  val rms_norm : epsilon:float -> t
  val w4a16_linear : m:int -> n:int -> k:int -> bias:bool -> t
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
