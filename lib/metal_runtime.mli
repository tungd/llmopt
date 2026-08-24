type t
type runtime = t

val load_package :
  root:string -> Serving_package.t -> (t, string) result

val load_packages :
  (string * Serving_package.t) list -> (t list, string) result

val device_name : t -> string
val package : t -> Serving_package.t

module Buffer : sig
  type t

  val of_bytes : runtime:runtime -> bytes -> (t, string) result
  val create : runtime:runtime -> bytes:int -> (t, string) result
  val view : parent:t -> offset:int -> bytes:int -> (t, string) result
  val contents : t -> (bytes, string) result
  val byte_length : t -> int
  val copy : source:t -> destination:t -> (unit, string) result
end

module Cache : sig
  module Attention : sig
    type t = Key | Value
  end

  type t

  val create : runtime:runtime -> config:Kv_cache.Config.t -> (t, string) result
  val format : t -> Kv_cache.Format.t
  val token_pool_bytes : t -> int
  val checkpoint_pool_bytes : t -> int

  val pack_attention :
    t ->
    layer:int ->
    kind:Attention.t ->
    slots:Kv_cache.Slot.t array ->
    source:Buffer.t ->
    (string, string) result

  val pack_attention_slice :
    t ->
    layer:int ->
    kind:Attention.t ->
    slots:Kv_cache.Slot.t array ->
    source_items:int ->
    source_offset:int ->
    source:Buffer.t ->
    (string, string) result

  val unpack_attention :
    t ->
    layer:int ->
    kind:Attention.t ->
    slots:Kv_cache.Slot.t array ->
    destination:Buffer.t ->
    (string, string) result

  val pack_checkpoint :
    t ->
    layer:int ->
    checkpoint:Kv_cache.Checkpoint.t ->
    source:Buffer.t ->
    (string, string) result

  val unpack_checkpoint :
    t ->
    layer:int ->
    checkpoint:Kv_cache.Checkpoint.t ->
    destination:Buffer.t ->
    (string, string) result
end

module Execution : sig
  type t

  val output : t -> name:string -> Buffer.t option
  val outputs : t -> (string * Buffer.t) list
  val kernels : t -> string list
  val workspace_bytes : t -> int
end

val execute :
  t -> inputs:(string * Buffer.t) list -> (Execution.t, string) result

val execute_schedule :
  t ->
  schedule:Serving_schedule.t ->
  inputs:(string * Buffer.t) list ->
  (Execution.t, string) result

val tensor :
  t -> name:string -> (Buffer.t * Weight_archive.Tensor.t, string) result

val dispatch_q8_linear :
  t ->
  dtype:Ir.Dtype.t ->
  input:Buffer.t ->
  weight:Buffer.t ->
  scale:Buffer.t ->
  bias:Buffer.t option ->
  output:Buffer.t ->
  m:int ->
  n:int ->
  k:int ->
  (string, string) result
