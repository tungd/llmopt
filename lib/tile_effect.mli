type input = {
  name : string;
  source : Ir.Input_source.t;
  shape : Shape.t;
  logical_shape : Tensor_shape.t;
  dtype : Ir.Dtype.t;
}
type allocation = {
  shape : Shape.t;
  logical_shape : Tensor_shape.t;
  dtype : Ir.Dtype.t;
  space : Ir.Memory_space.t;
  layout : Ir.Layout.t;
}
type matmul = {
  lhs : Ir.Value.t;
  rhs : Ir.Value.t;
  shape : Shape.t;
  logical_shape : Tensor_shape.t;
}
type linear = {
  input : Ir.Value.t;
  weight : Ir.Value.t;
  bias : Ir.Value.t option;
  shape : Shape.t;
  logical_shape : Tensor_shape.t;
}
type q8_linear = {
  input : Ir.Value.t;
  weight : Ir.Value.t;
  scale : Ir.Value.t;
  bias : Ir.Value.t option;
  shape : Shape.t;
  logical_shape : Tensor_shape.t;
}
type add = {
  lhs : Ir.Value.t;
  rhs : Ir.Value.t;
  shape : Shape.t;
  logical_shape : Tensor_shape.t;
  broadcast : Shape.broadcast;
}
type unary = {
  input : Ir.Value.t;
  shape : Shape.t;
  logical_shape : Tensor_shape.t;
}
type opaque = {
  op : string;
  target : string;
  arguments : Ir.Argument.t list;
  keyword_arguments : (string * Ir.Argument.t) list;
  inputs : Ir.Value.t list;
  shape : Shape.t;
  logical_shape : Tensor_shape.t;
  dtype : Ir.Dtype.t;
}

module Barrier : sig
  type t
  val of_id : int -> t
  val id : t -> int
end

type _ Effect.t +=
  | Input : input -> Ir.Value.t Effect.t
  | Alloc : allocation -> Ir.Value.t Effect.t
  | Copy : { src : Ir.Value.t; dst : Ir.Value.t } -> unit Effect.t
  | Async_copy : { src : Ir.Value.t; dst : Ir.Value.t; barrier : Barrier.t } -> unit Effect.t
  | Matmul : matmul -> Ir.Value.t Effect.t
  | Linear : linear -> Ir.Value.t Effect.t
  | Q8_linear : q8_linear -> Ir.Value.t Effect.t
  | Add : add -> Ir.Value.t Effect.t
  | Gelu : unary -> Ir.Value.t Effect.t
  | Relu : unary -> Ir.Value.t Effect.t
  | Opaque : opaque -> Ir.Value.t Effect.t
  | Output : { name : string; value : Ir.Value.t } -> unit Effect.t
  | Barrier_create : string -> Barrier.t Effect.t
  | Barrier_arrive : Barrier.t -> unit Effect.t
  | Barrier_wait : Barrier.t -> unit Effect.t

val input :
  name:string ->
  source:Ir.Input_source.t ->
  shape:Shape.t ->
  dtype:Ir.Dtype.t ->
  Ir.Value.t
val tensor_input :
  name:string ->
  source:Ir.Input_source.t ->
  shape:Tensor_shape.t ->
  dtype:Ir.Dtype.t ->
  Ir.Value.t
val alloc : allocation -> Ir.Value.t
val copy : src:Ir.Value.t -> dst:Ir.Value.t -> unit
val async_copy : src:Ir.Value.t -> dst:Ir.Value.t -> barrier:Barrier.t -> unit
val matmul : matmul -> Ir.Value.t
val linear : linear -> Ir.Value.t
val q8_linear : q8_linear -> Ir.Value.t
val add : add -> Ir.Value.t
val gelu : unary -> Ir.Value.t
val relu : unary -> Ir.Value.t
val opaque : opaque -> Ir.Value.t
val output : name:string -> value:Ir.Value.t -> unit
val create_barrier : string -> Barrier.t
val barrier_arrive : Barrier.t -> unit
val barrier_wait : Barrier.t -> unit
