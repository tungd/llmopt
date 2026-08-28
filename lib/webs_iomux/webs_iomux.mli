type t

module Stream : sig
  type t

  val create : ?max_chunks:int -> unit -> t
  val push : t -> string -> bool
  val push_wait : t -> string -> bool
  val replace : t -> string -> bool
  val pop : t -> [ `Data of string | `Wait | `Eof ]
  val close : t -> unit
  val closed : t -> bool
  val response : t -> Webs.Http.Response.t
end

val start :
  ?backlog:int ->
  ?max_header_bytes:int ->
  ?max_body_bytes:int ->
  ?max_keepalive:int ->
  ?header_timeout_ms:int ->
  ?idle_timeout_ms:int ->
  host:string ->
  port:int ->
  (Webs.Http.Request.t -> Webs.Http.Response.t) ->
  (t, string) result

val port : t -> int
val stop : t -> unit

val serve :
  ?addr:string ->
  ?port:int ->
  ?backlog:int ->
  ?max_header_bytes:int ->
  ?max_body_bytes:int ->
  ?max_keepalive:int ->
  ?header_timeout_ms:int ->
  ?idle_timeout_ms:int ->
  (Webs.Http.Request.t -> Webs.Http.Response.t) ->
  unit
