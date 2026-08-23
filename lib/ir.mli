module Dtype : sig
  type t = Float32 | Float16 | Bfloat16 | Int64 | Int32 | Int8 | Bool
  val to_string : t -> string
end

module Quantization : sig
  type t = Fp16 | Q8_weight_only
  val to_string : t -> string
end

module Memory_space : sig
  type t = Global | Shared | Register | Private
  val to_string : t -> string
end

module Layout : sig
  type t = Row_major | Col_major | Xor_swizzle of int
  val to_string : t -> string
end

module Input_source : sig
  type t = Runtime | Tensor_store of { key : string }
  val to_string : t -> string
end

module Value_id : sig
  type t
  val compare : t -> t -> int
  val to_int : t -> int
end

module Value : sig
  type t
  val make : id:int -> shape:Shape.t -> dtype:Dtype.t -> t
  val id : t -> Value_id.t
  val shape : t -> Shape.t
  val dtype : t -> Dtype.t
  val equal : t -> t -> bool
end

type node

module Op : sig
  type t =
    | Input of { name : string; source : Input_source.t }
    | Alloc of { space : Memory_space.t; layout : Layout.t }
    | Copy of { asynchronous : bool; barrier : int option }
    | Matmul of { m : int; n : int; k : int }
    | Linear of { m : int; n : int; k : int; bias : bool }
    | Add of { broadcast : Shape.broadcast }
    | Gelu
    | Relu
    | Opaque of { op : string; target : string }
    | Output of { name : string }
    | Barrier_create of { id : int; name : string }
    | Barrier_arrive of int
    | Barrier_wait of int
    | Fused_matmul_bias of { m : int; n : int; k : int }
    | Q8_linear of { m : int; n : int; k : int; bias : bool }

  val to_string : t -> string
end

val node_id : node -> int
val node_op : node -> Op.t
val node_inputs : node -> Value.t list
val node_output : node -> Value.t option
val node_replace : node -> op:Op.t -> inputs:Value.t list -> node

module Graph : sig
  type t

  val create : unit -> t
  val fresh_value : t -> shape:Shape.t -> dtype:Dtype.t -> Value.t
  val input :
    t ->
    name:string ->
    source:Input_source.t ->
    shape:Shape.t ->
    dtype:Dtype.t ->
    Value.t
  val allocate :
    t -> shape:Shape.t -> dtype:Dtype.t -> space:Memory_space.t -> layout:Layout.t -> Value.t
  val append : t -> op:Op.t -> inputs:Value.t list -> output:Value.t option -> unit
  val add_output : t -> name:string -> Value.t -> unit
  val nodes : t -> node list
  val outputs : t -> (string * Value.t) list
  val with_nodes : t -> node list -> t
  val pp : Format.formatter -> t -> unit
end
