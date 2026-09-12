type t = Metal_native.t
type runtime = t

val load_package :
  root:string -> Serving_package.t -> (t, string) result

val load_packages :
  (string * Serving_package.t) list -> (t list, string) result

val device_name : t -> string
val package : t -> Serving_package.t

module Decode_uniforms = Metal_native.Decode_uniforms

module Quant_qkv_tactic : sig
  val kernel_name :
    q_dtype:Ir.Dtype.t ->
    experimental_llama_mv:bool ->
    experimental_mixed_llama_mv:bool ->
    m:int ->
    v_dtype:Ir.Dtype.t ->
    (string, string) result
end

module Paged_attention_tactic : sig
  val kernel_name :
    experimental_gqa3_split8:bool ->
    split8:bool ->
    query_heads:int ->
    kv_heads:int ->
    head_dimension:int ->
    string

  val groups_by_kv_head : string -> bool
end

module Buffer = Metal_native.Buffer

module Cache = Metal_cache

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
  ?decode_uniforms:Decode_uniforms.t ->
  ?cache:Cache.t ->
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

val encode_cache_pack_attention_pair_slice :
  Execution_batch.t ->
  cache:Cache.t ->
  layer:int ->
  slots:Kv_cache.Slot.t array ->
  source_items:int ->
  source_offset:int ->
  key_source:Buffer.t ->
  value_source:Buffer.t ->
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

val bind_schedule_plan :
  ?workspace:Buffer.t ->
  ?memory_plan:Serving_memory_plan.t ->
  t ->
  schedule:Serving_schedule.t ->
  inputs:(string * Buffer.t) list ->
  ((string * Buffer.t list * bytes * int * int * int * int * int * int * bool) array * Execution.t, string) result

val bind_decode_plan :
  ?workspace:Buffer.t ->
  ?memory_plan:Serving_memory_plan.t ->
  decode_uniforms:Decode_uniforms.t ->
  ?cache:Cache.t ->
  ?cache_pack:(Execution.t -> Cache.batch -> (unit, string) result) ->
  t ->
  schedule:Serving_schedule.t ->
  inputs:(string * Buffer.t) list ->
  ((string * Buffer.t list * bytes * int * int * int * int * int * int * bool) array * Execution.t, string) result

val tensor :
  t -> name:string -> (Buffer.t * Weight_archive.Tensor.t, string) result

val dispatch_speculative_attention :
  ?batch:Execution_batch.t ->
  runtime ->
  query:Buffer.t ->
  key:Buffer.t ->
  value:Buffer.t ->
  output:Buffer.t ->
  batches:int ->
  heads:int ->
  query_length:int ->
  key_length:int ->
  head_dimension:int ->
  past_length:int ->
  scale:float ->
  ?kv_heads:int ->
  ?tree_masks:int array ->
  unit ->
  (string, string) result

module Ring_queue = Metal_native.Ring_queue

module Prebaked = Metal_native.Prebaked
