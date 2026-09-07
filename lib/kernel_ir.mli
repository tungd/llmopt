(** Target-independent structured tensor SSA for model paths.

    Facade over [Kernel_region] (representation + validation), [Kernel_scan]
    (carry-chain recovery/fusion), and [Kernel_fused_tactic] (backend tactic
    selection).  Linear is semantic here: packed-weight layout, scale
    representation, storage spaces, launch geometry, SIMD instructions, and
    generated source belong to later lowering interfaces. *)

module Slot = Kernel_region.Slot
module Primitive = Kernel_region.Primitive
module Effect = Kernel_region.Effect
module Resource = Kernel_region.Resource

(** Bindings and inputs are indexed by slot id during validation and in
    tactic selection. *)
module Slot_map = Kernel_region.Slot_map

type input = Kernel_region.input
type output_spec = Kernel_region.output_spec
type binding = Kernel_region.binding
type storage = Kernel_region.storage
type result = Kernel_region.result
type t = Kernel_region.t
type region = Kernel_region.region
type error = Kernel_region.error

val make_binding :
  outputs:output_spec list ->
  primitive:Primitive.t ->
  inputs:Slot.t list ->
  (binding, string) Stdlib.result

val binding_outputs : binding -> output_spec list
val binding_inputs : binding -> Slot.t list
val binding_primitive : binding -> Primitive.t
val output_slot : output_spec -> Slot.t
val output_shape : output_spec -> Tensor_shape.t
val output_dtype : output_spec -> Ir.Dtype.t
val output_value : output_spec -> Ir.Value.t option

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

module Scan = Kernel_scan
module Fused_tactic = Kernel_fused_tactic

val result_slot : result -> Slot.t
val result_value : result -> Ir.Value.t
val result_storage : result -> storage

val error_to_string : error -> string
val pp : Format.formatter -> t -> unit
val to_string : t -> string
