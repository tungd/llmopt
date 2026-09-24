module Attention : sig
  type t = Key | Value
end

type t

type batch = {
  cache : t;
  commands : Metal_native.Batch.t;
  mutable resources : Metal_native.Buffer.t list;
  mutable dispatches : int;
  pack_slots_scratch : Metal_native.Buffer.t;
}

val create : runtime:Metal_native.runtime -> config:Kv_cache.Config.t -> (t, string) result
val format : t -> Kv_cache.Format.t
val token_pool_bytes : t -> int
val checkpoint_pool_bytes : t -> int

val retained_batch : t -> Metal_native.Batch.t -> batch

val with_batch :
  t -> (batch -> ('a, string) result) -> ('a, string) result

val with_batch_async :
  t -> (batch -> ('a, string) result) -> ('a, string) result

val pack_attention :
  t ->
  layer:int ->
  kind:Attention.t ->
  slots:Kv_cache.Slot.t array ->
  source:Metal_native.Buffer.t ->
  (string, string) result

val pack_attention_slice :
  t ->
  layer:int ->
  kind:Attention.t ->
  slots:Kv_cache.Slot.t array ->
  source_items:int ->
  source_offset:int ->
  source:Metal_native.Buffer.t ->
  (string, string) result

val unpack_attention :
  t ->
  layer:int ->
  kind:Attention.t ->
  slots:Kv_cache.Slot.t array ->
  destination:Metal_native.Buffer.t ->
  (string, string) result

val batch_pack_attention :
  batch ->
  layer:int ->
  kind:Attention.t ->
  slots:Kv_cache.Slot.t array ->
  source:Metal_native.Buffer.t ->
  (string, string) result

val batch_pack_attention_pair :
  batch ->
  layer:int ->
  slots:Kv_cache.Slot.t array ->
  key_source:Metal_native.Buffer.t ->
  value_source:Metal_native.Buffer.t ->
  (string, string) result

val batch_pack_attention_slice :
  batch ->
  layer:int ->
  kind:Attention.t ->
  slots:Kv_cache.Slot.t array ->
  source_items:int ->
  source_offset:int ->
  source:Metal_native.Buffer.t ->
  (string, string) result

val batch_pack_attention_pair_slice :
  batch ->
  layer:int ->
  slots:Kv_cache.Slot.t array ->
  source_items:int ->
  source_offset:int ->
  key_source:Metal_native.Buffer.t ->
  value_source:Metal_native.Buffer.t ->
  (string, string) result

val batch_unpack_attention :
  batch ->
  layer:int ->
  kind:Attention.t ->
  slots:Kv_cache.Slot.t array ->
  destination:Metal_native.Buffer.t ->
  (string, string) result

val pack_checkpoint :
  t ->
  layer:int ->
  checkpoint:Kv_cache.Checkpoint.t ->
  source:Metal_native.Buffer.t ->
  (string, string) result

val unpack_checkpoint :
  t ->
  layer:int ->
  checkpoint:Kv_cache.Checkpoint.t ->
  destination:Metal_native.Buffer.t ->
  (string, string) result

val batch_pack_checkpoint :
  batch ->
  layer:int ->
  checkpoint:Kv_cache.Checkpoint.t ->
  source:Metal_native.Buffer.t ->
  (string, string) result

val batch_unpack_checkpoint :
  batch ->
  layer:int ->
  checkpoint:Kv_cache.Checkpoint.t ->
  destination:Metal_native.Buffer.t ->
  (string, string) result

val q8_attention_inputs :
  t ->
  slots:Kv_cache.Slot.t array ->
  ((string * Metal_native.Buffer.t) list, string) result

val update_pack_slot : t -> Kv_cache.Slot.t -> (unit, string) result
