module Buffer = Metal_native.Buffer
module Batch = Metal_native.Batch

module Value_map : Map.S with type key = Ir.Value_id.t
module String_map : Map.S with type key = String.t

module Execution : sig
  type t = {
    outputs : (string * Buffer.t) list;
    kernels : string list;
    workspace_bytes : int;
  }

  val output : t -> name:string -> Buffer.t option
  val outputs : t -> (string * Buffer.t) list
  val kernels : t -> string list
  val workspace_bytes : t -> int
end

module Execution_batch : sig
  type t = {
    runtime : Metal_native.runtime;
    commands : Batch.t;
    mutable resources : Buffer.t list;
    mutable schedules : int;
    mutable swiglu_normalized : Buffer.t option;
    mutable swiglu_product : Buffer.t option;
    mutable candidates_buffer : Buffer.t option;
  }
end

type t = {
  values : Buffer.t Value_map.t;
  outputs_rev : (string * Buffer.t) list;
  kernels_rev : string list;
  memory_plan : Serving_memory_plan.t;
  workspace : Buffer.t option;
  resources : Buffer.t list;
}

type execution_state = t

type context = {
  runtime : Metal_native.runtime;
  batch : Batch.t;
  execution_batch : Execution_batch.t;
  decode_uniforms : Metal_native.Decode_uniforms.t option;
  cache : Metal_cache.t option;
}

val with_execution_batch :
  Metal_native.runtime ->
  (Execution_batch.t -> ('a, string) result) ->
  ('a, string) result

val encode_cache :
  Execution_batch.t ->
  Metal_cache.t ->
  (Metal_cache.batch -> ('a, string) result) ->
  ('a, string) result

val encode_cache_pack_attention_slice :
  Execution_batch.t ->
  cache:Metal_cache.t ->
  layer:int ->
  kind:Metal_cache.Attention.t ->
  slots:Kv_cache.Slot.t array ->
  source_items:int ->
  source_offset:int ->
  source:Buffer.t ->
  (string, string) result

val encode_cache_pack_attention_pair_slice :
  Execution_batch.t ->
  cache:Metal_cache.t ->
  layer:int ->
  slots:Kv_cache.Slot.t array ->
  source_items:int ->
  source_offset:int ->
  key_source:Buffer.t ->
  value_source:Buffer.t ->
  (string, string) result

val encode_cache_pack_checkpoint :
  Execution_batch.t ->
  cache:Metal_cache.t ->
  layer:int ->
  checkpoint:Kv_cache.Checkpoint.t ->
  source:Buffer.t ->
  (string, string) result

val value_byte_length : Ir.Value.t -> (int, string) result
val validate_buffer : Ir.Value.t -> Buffer.t -> (unit, string) result
val runtime_input_map : (string * Buffer.t) list -> (Buffer.t String_map.t, string) result
val find_value : t -> Ir.Value.t -> (Buffer.t, string) result
val find_values : t -> Ir.Value.t list -> (Buffer.t list, string) result
val bind_value : t -> Ir.Value.t -> Buffer.t -> t
val workspace_buffer : t -> Ir.Value.t -> (Buffer.t, string) result
