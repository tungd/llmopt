module Config : sig
  type t

  val create :
    state:Model_program.State.t ->
    token_capacity:int ->
    checkpoint_capacity:int ->
    page_size:int ->
    unit ->
    (t, string) result

  val of_state_plan :
    state:Model_program.State.t ->
    token_capacity:int ->
    checkpoint_capacity:int ->
    page_size:int ->
    unit ->
    (t, string) result

  val kv : t -> Kv_cache.Config.t
  val page_size : t -> int
  val state_plan : t -> Model_program.State.t
end

module Match : sig
  type t

  val tokens : t -> int
  val slots : t -> Kv_cache.Slot.t array
  val checkpoint : t -> Kv_cache.Checkpoint.t option
end

module Stats : sig
  type t = {
    radix : Radix_cache.Stats.t;
    kv : Kv_cache.Stats.t;
  }
end

type t

val create : Config.t -> t
val reserve_tokens : t -> int -> (Kv_cache.Slot.t array, Kv_cache.error) result
val reserve_checkpoint : t -> (Kv_cache.Checkpoint.t, Kv_cache.error) result
val release_tokens : t -> Kv_cache.Slot.t array -> (unit, string) result
val release_checkpoint : t -> Kv_cache.Checkpoint.t -> (unit, string) result

val match_prefix :
  t -> ?namespace:string -> reserve_tail:int -> int array -> Match.t

val release_match : t -> Match.t -> (unit, string) result

val insert :
  t ->
  ?namespace:string ->
  tokens:int array ->
  slots:Kv_cache.Slot.t array ->
  checkpoint:Kv_cache.Checkpoint.t ->
  unit ->
  (int, string) result

val validate : t -> (unit, string) result
val stats : t -> Stats.t
