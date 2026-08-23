type t
type runtime = t

val load_package :
  root:string -> Serving_package.t -> (t, string) result

val device_name : t -> string

module Buffer : sig
  type t

  val of_bytes : runtime:runtime -> bytes -> (t, string) result
  val create : runtime:runtime -> bytes:int -> (t, string) result
  val contents : t -> (bytes, string) result
  val byte_length : t -> int
  val copy : source:t -> destination:t -> (unit, string) result
end

module Execution : sig
  type t

  val output : t -> name:string -> Buffer.t option
  val outputs : t -> (string * Buffer.t) list
  val kernels : t -> string list
end

val execute :
  t -> inputs:(string * Buffer.t) list -> (Execution.t, string) result

val tensor :
  t -> name:string -> (Buffer.t * Weight_archive.Tensor.t, string) result

val dispatch_q8_linear :
  t ->
  dtype:Ir.Dtype.t ->
  input:Buffer.t ->
  weight:Buffer.t ->
  scale:Buffer.t ->
  bias:Buffer.t option ->
  output:Buffer.t ->
  m:int ->
  n:int ->
  k:int ->
  (string, string) result
