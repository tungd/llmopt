module Stage : sig
  type t = Compiled_graph | Serving

  val to_string : t -> string
end

module Artifact : sig
  type t

  val create : string -> (t, string) result
  val path : t -> string
end

module Files : sig
  type t

  val create : metal_library:Artifact.t -> t
  val metal_library : t -> Artifact.t
end

module Tensor_store : sig
  type t

  val weights : file:Artifact.t -> t
  val file : t -> Artifact.t
end

module Cache : sig
  type t

  val create :
    page_size:int ->
    default_kv:Kv_cache.Format.t ->
    supported_kv:Kv_cache.Format.t list ->
    (t, string) result

  val default : t
  val page_size : t -> int
  val default_kv : t -> Kv_cache.Format.t
  val supported_kv : t -> Kv_cache.Format.t list
end

type t

val compiled_graph :
  ?model:string ->
  files:Files.t ->
  kernels:Kernel_abi.Entry.t list ->
  schedule:Serving_schedule.t ->
  cache:Cache.t ->
  unit ->
  (t, string) result

val serving :
  ?model:string ->
  files:Files.t ->
  kernels:Kernel_abi.Entry.t list ->
  schedule:Serving_schedule.t ->
  tensor_store:Tensor_store.t ->
  cache:Cache.t ->
  unit ->
  (t, string) result

val stage : t -> Stage.t
val model : t -> string option
val files : t -> Files.t
val kernels : t -> Kernel_abi.Entry.t list
val schedule : t -> Serving_schedule.t
val tensor_store : t -> Tensor_store.t option
val cache : t -> Cache.t

val to_bytes : t -> bytes
val of_bytes : bytes -> (t, string) result
val write_file : string -> t -> (unit, string) result
val of_file : string -> (t, string) result
val validate_files : root:string -> t -> (unit, string) result
