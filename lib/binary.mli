module Writer : sig
  type t

  val create : unit -> t
  val raw_string : t -> string -> unit
  val raw_bytes : t -> bytes -> unit
  val u8 : t -> int -> unit
  val u16 : t -> int -> unit
  val u32 : t -> int -> unit
  val u64 : t -> int -> unit
  val i64 : t -> int -> unit
  val float64 : t -> float -> unit
  val bool : t -> bool -> unit
  val string : t -> string -> unit
  val bytes : t -> bytes -> unit
  val contents : t -> bytes
end

module Reader : sig
  type t

  val create : bytes -> t
  val offset : t -> int
  val remaining : t -> int
  val raw_string : t -> length:int -> (string, string) result
  val raw_bytes : t -> length:int -> (bytes, string) result
  val u8 : t -> (int, string) result
  val u16 : t -> (int, string) result
  val u32 : t -> (int, string) result
  val u64 : t -> (int, string) result
  val i64 : t -> (int, string) result
  val float64 : t -> (float, string) result
  val bool : t -> (bool, string) result
  val string : t -> (string, string) result
  val bytes : t -> (bytes, string) result
  val finish : t -> (unit, string) result
end
