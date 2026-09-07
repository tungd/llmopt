(** Kernel region representation and validation: slots, primitives,
    effects, resources, and the validated region construction and
    specialization shared by the Kernel_ir facade and by tactic
    selection.  Split out of Kernel_ir; see LOOP. *)

module Slot : sig
  (** Slots are validated non-negative integers; manifest so sibling kernel
      units can share [Slot_map] indexing and lower layers carry slot ids. *)
  type t = int

  val of_int : int -> (t, string) result
  val of_int_exn : int -> t
  val to_int : t -> int
  val compare : t -> t -> int
  val equal : t -> t -> bool
end

module Primitive : sig
  type unary =
    | Neg
    | Rsqrt
    | Silu
    | Gelu
    | Cos
    | Sin
    | Tanh
    | Exp
    | Sigmoid
    | Softplus
    | Pow of Ir.Scalar.t

  type binary =
    | Mul
    | Add
    | Sub
    | Div
    | Logical_and
    | Equal
    | Not_equal
    | Less_equal
    | Greater

  type scalar_binary = {
    operator : binary;
    scalar : Ir.Scalar.t;
    scalar_on_left : bool;
  }

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
    | Scalar_binary of scalar_binary
    | Reduce of Ir.Reduction.t
    | Cast of Ir.Dtype.t
    | Movement of Ir.Movement.t
    | Short_conv of Ir.Short_conv.t
    | Batched_matmul
    | Pad_right_zero of { axis : int }
    | Update_slice of Tensor_shape.Index.t
    | Triangular_recurrence of Ir.Triangular_recurrence.t

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
  val scalar_binary : operator:binary -> scalar:Ir.Scalar.t -> scalar_on_left:bool -> t
  val reduce : Ir.Reduction.t -> t
  val cast : Ir.Dtype.t -> t
  val movement : Ir.Movement.t -> t
  val short_conv : Ir.Short_conv.t -> t
  val batched_matmul : t
  val pad_right_zero : axis:int -> t
  val update_slice : Tensor_shape.Index.t -> t
  val triangular_recurrence : Ir.Triangular_recurrence.t -> t
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

(** Bindings and inputs are indexed by slot id during validation and in
    tactic selection. *)
module Slot_map : Map.S with type key = int

type input = {
  slot : Slot.t;
  value : Ir.Value.t;
}

type output_spec = {
  slot : Slot.t;
  shape : Tensor_shape.t;
  dtype : Ir.Dtype.t;
  (** Source-value provenance is present while compiling constructive
      assignments and omitted after package deserialization. *)
  value : Ir.Value.t option;
}

type binding = {
  outputs : output_spec list;
  primitive : Primitive.t;
  inputs : Slot.t list;
}

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
val output_value : output_spec -> Ir.Value.t option

type storage = Fresh | Alias_input of Slot.t | In_place_input of Slot.t

type result = {
  slot : Slot.t;
  value : Ir.Value.t;
  storage : storage;
}

type t = {
  name : string;
  member_node_ids : int list;
  inputs : input list;
  bindings : binding list;
  results : result list;
  effects : Effect.t;
  resource : Resource.t;
}

type region = t

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

(** Specialize region dimensions. [value] maps schedule-visible region inputs
    and results; assignment-local output provenance is rebuilt from its
    specialized binding metadata. *)
val specialize :
  dimension:(int -> int) ->
  value:(Ir.Value.t -> (Ir.Value.t, string) Stdlib.result) ->
  t ->
  (t, string) Stdlib.result

(** Backend tactic selection (fused linear epilogues, gated products, RMS
    normalization/RoPE, short-conv-SILU, gated delta) lives in
    [Kernel_fused_tactic]. *)
val result_slot : result -> Slot.t
val result_value : result -> Ir.Value.t
val result_storage : result -> storage

val error_to_string : error -> string
val pp : Format.formatter -> t -> unit
val to_string : t -> string
