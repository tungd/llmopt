type context_handle
type library_handle
type buffer_handle
type batch_handle
type ring_handle
type prebaked_handle

val protect : (unit -> 'a) -> ('a, string) result

type t = {
  context : context_handle;
  library : library_handle;
  package : Serving_package.t;
  tensor_store : (Weight_archive.t * buffer_handle) option;
}

type runtime = t

module Decode_uniforms : sig
  type t = { past_tokens : int }

  val create : past_tokens:int -> (t, string) result
  val past_tokens : t -> int
end

module Buffer : sig
  type t = buffer_handle

  val of_bytes : runtime:runtime -> bytes -> (t, string) result
  val create : runtime:runtime -> bytes:int -> (t, string) result
  val view : parent:t -> offset:int -> bytes:int -> (t, string) result
  val contents : t -> (bytes, string) result
  val argmax_f16 : t -> vocabulary:int -> (int, string) result
  val byte_length : t -> int
  val copy : source:t -> destination:t -> (unit, string) result
  val set_int64 : t -> offset:int -> int64 -> (unit, string) result
  val set_u32_array : t -> offset:int -> int array -> (unit, string) result
end

module Batch : sig
  type dispatch_item = {
    name : string;
    buffers : buffer_handle list;
    parameters : bytes;
    grid_x : int;
    grid_y : int;
    grid_z : int;
    group_x : int;
    group_y : int;
    group_z : int;
    barrier_before : bool;
  }

  type t = {
    handle : batch_handle;
    context : context_handle;
    mutable pending : (library_handle * dispatch_item) list;
    mutable next_dispatch_barrier : bool;
  }

  val create : runtime -> (t, string) result
  val flush : t -> (unit, string) result
  val dispatch :
    t ->
    runtime:runtime ->
    name:string ->
    buffers:Buffer.t list ->
    parameters:bytes ->
    grid:int * int * int ->
    group:int * int * int ->
    (unit, string) result
  val mark_barrier : t -> unit
  val copy : t -> source:Buffer.t -> destination:Buffer.t -> (unit, string) result
  val barrier : t -> (unit, string) result
  val commit : t -> (unit, string) result
  val commit_async : t -> (unit, string) result
  val abort : t -> (unit, string) result
end

val load_package : root:string -> Serving_package.t -> (t, string) result
val load_packages : (string * Serving_package.t) list -> (t list, string) result
val device_name : t -> string
val package : t -> Serving_package.t
val tensor : t -> name:string -> (Buffer.t * Weight_archive.Tensor.t, string) result

val kernel_entry :
  ?name:string ->
  runtime ->
  operation:Kernel_abi.Operation.t ->
  input_dtype:Ir.Dtype.t ->
  output_dtype:Ir.Dtype.t ->
  (Kernel_abi.Entry.t, string) result

val dispatch :
  ?batch:Batch.t ->
  runtime ->
  Kernel_abi.Entry.t ->
  buffers:Buffer.t list ->
  parameters:bytes ->
  grid:int * int * int ->
  (string, string) result

module Ring_queue : sig
  type t = ring_handle

  val default_pipeline_depth : int
  val pipeline_depth : unit -> int
  val create : unit -> (t, string) result
  val submit :
    t ->
    request_id:int ->
    token:int ->
    past_tokens:int ->
    flags:int ->
    (bool, string) result
  val wait_completion : t -> (int * int * int, string) result
  val poll_completion : t -> ((int * int * int) option, string) result
  val start_worker :
    t ->
    runtime:runtime ->
    dispatches:(string * Buffer.t list * bytes * int * int * int * int * int * int * bool)
               array ->
    token_buffer:Buffer.t ->
    output_buffer:Buffer.t ->
    (unit, string) result
  val destroy : t -> (unit, string) result
end

module Prebaked : sig
  type t = prebaked_handle

  val create :
    runtime:runtime ->
    dispatches:(string * Buffer.t list * bytes * int * int * int * int * int * int * bool)
               array ->
    token_buffer:Buffer.t ->
    output_buffer:Buffer.t ->
    (t, string) result

  val execute :
    t -> token:int -> past_tokens:int -> checkpoint:int -> (int, string) result
end
