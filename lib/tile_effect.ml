type input = {
  name : string;
  source : Ir.Input_source.t;
  shape : Shape.t;
  dtype : Ir.Dtype.t;
}

type allocation = {
  shape : Shape.t;
  dtype : Ir.Dtype.t;
  space : Ir.Memory_space.t;
  layout : Ir.Layout.t;
}

type matmul = { lhs : Ir.Value.t; rhs : Ir.Value.t; shape : Shape.t }

type linear = {
  input : Ir.Value.t;
  weight : Ir.Value.t;
  bias : Ir.Value.t option;
  shape : Shape.t;
}

type q8_linear = {
  input : Ir.Value.t;
  weight : Ir.Value.t;
  scale : Ir.Value.t;
  bias : Ir.Value.t option;
  shape : Shape.t;
}

type add = {
  lhs : Ir.Value.t;
  rhs : Ir.Value.t;
  shape : Shape.t;
  broadcast : Shape.broadcast;
}

type unary = { input : Ir.Value.t; shape : Shape.t }
type opaque = {
  op : string;
  target : string;
  inputs : Ir.Value.t list;
  shape : Shape.t;
  dtype : Ir.Dtype.t;
}

module Barrier = struct
  type t = int
  let of_id id = id
  let id barrier = barrier
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

let input ~name ~source ~shape ~dtype =
  Effect.perform (Input { name; source; shape; dtype })
let alloc allocation = Effect.perform (Alloc allocation)
let copy ~src ~dst = Effect.perform (Copy { src; dst })
let async_copy ~src ~dst ~barrier = Effect.perform (Async_copy { src; dst; barrier })
let matmul request = Effect.perform (Matmul request)
let linear request = Effect.perform (Linear request)
let q8_linear request = Effect.perform (Q8_linear request)
let add request = Effect.perform (Add request)
let gelu request = Effect.perform (Gelu request)
let relu request = Effect.perform (Relu request)
let opaque request = Effect.perform (Opaque request)
let output ~name ~value = Effect.perform (Output { name; value })
let create_barrier name = Effect.perform (Barrier_create name)
let barrier_arrive barrier = Effect.perform (Barrier_arrive barrier)
let barrier_wait barrier = Effect.perform (Barrier_wait barrier)
