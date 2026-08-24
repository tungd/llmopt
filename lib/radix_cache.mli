module Key : sig
  type t

  val create : ?namespace:string -> int array -> t
  val namespace : t -> string option
  val tokens : t -> int array
  val length : t -> int
end

module Stats : sig
  type t = {
    cached_tokens : int;
    evictable_tokens : int;
    protected_tokens : int;
    checkpoints : int;
    hits : int;
    misses : int;
    evicted_tokens : int;
  }
end

type ('slot, 'checkpoint) t
type ('slot, 'checkpoint) lease

type ('slot, 'checkpoint) insert_result = {
  prefix_tokens : int;
  redundant_values : 'slot array;
  redundant_checkpoint : 'checkpoint option;
  canonical_values : 'slot array;
  retained_checkpoint : 'checkpoint option;
}

type ('slot, 'checkpoint) eviction = {
  values : 'slot array;
  checkpoints : 'checkpoint list;
}

val create : page_size:int -> (('slot, 'checkpoint) t, string) result

val match_prefix :
  ('slot, 'checkpoint) t -> Key.t -> ('slot, 'checkpoint) lease

val matched_tokens : ('slot, 'checkpoint) lease -> int
val matched_values : ('slot, 'checkpoint) lease -> 'slot array
val checkpoint : ('slot, 'checkpoint) lease -> 'checkpoint option

val release :
  ('slot, 'checkpoint) t -> ('slot, 'checkpoint) lease -> (unit, string) result

val insert :
  ('slot, 'checkpoint) t ->
  key:Key.t ->
  values:'slot array ->
  checkpoint:'checkpoint ->
  (('slot, 'checkpoint) insert_result, string) result

val evict :
  ('slot, 'checkpoint) t -> target_tokens:int ->
  ('slot, 'checkpoint) eviction

val stats : ('slot, 'checkpoint) t -> Stats.t
val validate : ('slot, 'checkpoint) t -> (unit, string) result
