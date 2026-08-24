module Step : sig
  type t

  val logits : t -> Metal_runtime.Buffer.t
  val tokens : t -> int array
  val cached_prefix : t -> int
  val kernels : t -> string list
end

type t

val validate_packages :
  config:Serving_cache.Config.t ->
  prefill:Serving_package.t ->
  decode:Serving_package.t ->
  (unit, string) result

val create :
  config:Serving_cache.Config.t ->
  prefill:Metal_runtime.t ->
  decode:Metal_runtime.t ->
  (t, string) result

val prefill_tokens : t -> int
val past_tokens : t -> int
val prefill : t -> tokens:int array -> (Step.t, string) result
val decode : t -> prefix:int array -> token:int -> (Step.t, string) result
val stats : t -> Serving_cache.Stats.t
val validate : t -> (unit, string) result
