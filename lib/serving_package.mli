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

  val create :
    fx:Artifact.t ->
    plan:Artifact.t ->
    metal_source:Artifact.t ->
    metal_library:Artifact.t ->
    llvm_ir:Artifact.t ->
    t

  val fx : t -> Artifact.t
  val plan : t -> Artifact.t
  val metal_source : t -> Artifact.t
  val metal_library : t -> Artifact.t
  val llvm_ir : t -> Artifact.t
end

module Weight : sig
  module Encoding : sig
    type t =
      | Raw
      | Q8_per_output_channel of {
          scale : Artifact.t;
          scale_dtype : Ir.Dtype.t;
          axis : int;
        }
  end

  type t

  val create :
    name:string ->
    data:Artifact.t ->
    dtype:Ir.Dtype.t ->
    shape:int list ->
    encoding:Encoding.t ->
    (t, string) result

  val name : t -> string
  val data : t -> Artifact.t
  val dtype : t -> Ir.Dtype.t
  val shape : t -> int list
  val encoding : t -> Encoding.t
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
  cache:Cache.t ->
  unit ->
  (t, string) result

val serving :
  ?model:string ->
  files:Files.t ->
  kernels:Kernel_abi.Entry.t list ->
  weights:Weight.t list ->
  cache:Cache.t ->
  unit ->
  (t, string) result

val stage : t -> Stage.t
val model : t -> string option
val files : t -> Files.t
val kernels : t -> Kernel_abi.Entry.t list
val weights : t -> Weight.t list
val cache : t -> Cache.t

val to_yojson : t -> Yojson.Basic.t
val of_yojson : Yojson.Basic.t -> (t, string) result
val write_file : string -> t -> (unit, string) result
val of_file : string -> (t, string) result
val validate_files : root:string -> t -> (unit, string) result
