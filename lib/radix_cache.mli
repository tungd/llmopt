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

(** Borrowed view of the lease-local matched values. The caller must treat the
    array as read-only and stop using it before releasing the lease. *)
val matched_values_view : ('slot, 'checkpoint) lease -> 'slot array

val checkpoint : ('slot, 'checkpoint) lease -> 'checkpoint option

val release :
  ('slot, 'checkpoint) t -> ('slot, 'checkpoint) lease -> (unit, string) result

val insert :
  ('slot, 'checkpoint) t ->
  key:Key.t ->
  values:'slot array ->
  ?checkpoint:'checkpoint ->
  unit ->
  (('slot, 'checkpoint) insert_result, string) result

(** Insert a complete prefix and acquire its canonical node in the same tree
    traversal. The returned lease must be released with [release]. *)
val insert_and_match :
  ('slot, 'checkpoint) t ->
  key:Key.t ->
  values:'slot array ->
  ?checkpoint:'checkpoint ->
  unit ->
  (('slot, 'checkpoint) insert_result * ('slot, 'checkpoint) lease, string)
  result

(** Extend a live resumable lease by one page-size-one token without walking
    from the radix root. The original and returned leases are independent and
    must each be released. The canonical value and any redundant checkpoint
    identify ownership when another prefix already supplied the child. *)
val extend :
  ('slot, 'checkpoint) t ->
  ('slot, 'checkpoint) lease ->
  token:int ->
  value:'slot ->
  ?checkpoint:'checkpoint ->
  unit ->
  (('slot, 'checkpoint) lease * 'slot * 'checkpoint option, string) result

val evict :
  ('slot, 'checkpoint) t -> target_tokens:int ->
  ('slot, 'checkpoint) eviction

val stats : ('slot, 'checkpoint) t -> Stats.t
val validate : ('slot, 'checkpoint) t -> (unit, string) result

module Speculative : sig
  type ('slot, 'checkpoint) reservation

  val create :
    ?checkpoint:'checkpoint ->
    key:Key.t ->
    slots:'slot array ->
    unit ->
    ('slot, 'checkpoint) reservation

  val key : ('slot, 'checkpoint) reservation -> Key.t
  val slots : ('slot, 'checkpoint) reservation -> 'slot array
  val checkpoint : ('slot, 'checkpoint) reservation -> 'checkpoint option
end

val commit_speculative :
  ('slot, 'checkpoint) t ->
  reservation:('slot, 'checkpoint) Speculative.reservation ->
  accepted_count:int ->
  (('slot, 'checkpoint) insert_result option * 'slot array, string) result
