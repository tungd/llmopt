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
  val set_int64 : t -> offset:int -> int64 -> (unit, string) result
  val set_u32_array : t -> offset:int -> int array -> (unit, string) result
end

module Cache : sig
  module Attention : sig
    type t = Key | Value
  end

  type t
  type batch

  val create : runtime:runtime -> config:Kv_cache.Config.t -> (t, string) result
  val format : t -> Kv_cache.Format.t
  val token_pool_bytes : t -> int
  val checkpoint_pool_bytes : t -> int

  val with_batch :
    t -> (batch -> ('a, string) result) -> ('a, string) result

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

  val batch_pack_attention :
    batch ->
    layer:int ->
    kind:Attention.t ->
    slots:Kv_cache.Slot.t array ->
    source:Buffer.t ->
    (string, string) result

  val batch_pack_attention_slice :
    batch ->
    layer:int ->
    kind:Attention.t ->
    slots:Kv_cache.Slot.t array ->
    source_items:int ->
    source_offset:int ->
    source:Buffer.t ->
    (string, string) result

  val batch_unpack_attention :
    batch ->
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

  val batch_pack_checkpoint :
    batch ->
    layer:int ->
    checkpoint:Kv_cache.Checkpoint.t ->
    source:Buffer.t ->
    (string, string) result

  val batch_unpack_checkpoint :
    batch ->
    layer:int ->
    checkpoint:Kv_cache.Checkpoint.t ->
    destination:Buffer.t ->
    (string, string) result

  val q8_attention_inputs :
    t ->
    slots:Kv_cache.Slot.t array ->
    ((string * Buffer.t) list, string) result
end

module Execution : sig
  type t

  val output : t -> name:string -> Buffer.t option
  val outputs : t -> (string * Buffer.t) list
  val kernels : t -> string list
  val workspace_bytes : t -> int
end

module Execution_batch : sig
  type t
end

val with_execution_batch :
  runtime ->
  (Execution_batch.t -> ('a, string) result) ->
  ('a, string) result

val encode_schedule :
  ?workspace:Buffer.t ->
  ?memory_plan:Serving_memory_plan.t ->
  Execution_batch.t ->
  schedule:Serving_schedule.t ->
  inputs:(string * Buffer.t) list ->
  (Execution.t, string) result

val encode_cache_pack_attention_slice :
  Execution_batch.t ->
  cache:Cache.t ->
  layer:int ->
  kind:Cache.Attention.t ->
  slots:Kv_cache.Slot.t array ->
  source_items:int ->
  source_offset:int ->
  source:Buffer.t ->
  (string, string) result

val encode_cache_pack_checkpoint :
  Execution_batch.t ->
  cache:Cache.t ->
  layer:int ->
  checkpoint:Kv_cache.Checkpoint.t ->
  source:Buffer.t ->
  (string, string) result

val execute :
  t -> inputs:(string * Buffer.t) list -> (Execution.t, string) result

val execute_schedule :
  ?workspace:Buffer.t ->
  ?memory_plan:Serving_memory_plan.t ->
  t ->
  schedule:Serving_schedule.t ->
  inputs:(string * Buffer.t) list ->
  (Execution.t, string) result

val execute_decode_step :
  ?workspace:Buffer.t ->
  ?memory_plan:Serving_memory_plan.t ->
  t ->
  cache:Cache.t ->
  schedule:Serving_schedule.t ->
  inputs:(string * Buffer.t) list ->
  cache_pack:(Execution.t -> Cache.batch -> (unit, string) result) ->
  (Execution.t, string) result

val precompile_decode_batch :
  ?workspace:Buffer.t ->
  ?memory_plan:Serving_memory_plan.t ->
  t ->
  schedule:Serving_schedule.t ->
  inputs:(string * Buffer.t) list ->
  ((string * Buffer.t list * bytes * int * int * int * int * int * int) array * Execution.t, string) result

val tensor :
  t -> name:string -> (Buffer.t * Weight_archive.Tensor.t, string) result

module Ring_queue : sig
  type t

  val create : unit -> (t, string) result

  val submit :
    t ->
    request_id:int ->
    token:int ->
    past_tokens:int ->
    flags:int ->
    (bool, string) result

  val wait_completion : t -> (int * int * int, string) result
  val poll_completion : t -> ((int * int * int) option, string) result

  val start_worker :
    t ->
    runtime:runtime ->
    dispatches:(string * Buffer.t list * bytes * int * int * int * int * int * int)
               array ->
    token_buffer:Buffer.t ->
    output_buffer:Buffer.t ->
    (unit, string) result

  val destroy : t -> (unit, string) result
end

module Prebaked : sig
  type t

  val create :
    runtime:runtime ->
    dispatches:(string * Buffer.t list * bytes * int * int * int * int * int * int)
               array ->
    token_buffer:Buffer.t ->
    output_buffer:Buffer.t ->
    (t, string) result

  val execute : t -> token:int -> past_tokens:int -> (int, string) result
end
