module Allocation : sig
  type t

  val offset : t -> int
  val bytes : t -> int
end

type t

val create : Serving_schedule.t -> (t, string) result
val plan_concurrent : Serving_schedule.t -> (t, string) result
val allocation : t -> Ir.Value.t -> Allocation.t option
val check_disjoint : t -> Ir.Value.t list -> bool
val workspace_bytes : t -> int
val bytes_without_reuse : t -> int
val allocation_count : t -> int
val value_bytes : Ir.Value.t -> (int, string) result
