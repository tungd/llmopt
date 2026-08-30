module Dtype : sig
  type quant_type =
    | Q8_0
    | Q4_K
    | Q5_K
    | Q6_K
    | Q5_0
    | Q4_0
    | IQ4_XS

  type t =
    | Float32
    | Float16
    | Bfloat16
    | Int64
    | Int32
    | Int8
    | UInt8
    | Bool
    | Quant of quant_type

  val to_string : t -> string
  val quant_to_string : quant_type -> string
  val quant_of_string : string -> quant_type option
end

module Tensor_layout : sig
  type t = Dense of Dtype.t | Block_quantized of Dtype.quant_type

  val of_dtype : Dtype.t -> t
  val block_elements : Dtype.quant_type -> int
  val block_bytes : Dtype.quant_type -> int
  val physical_bytes : t -> Tensor_shape.t -> (int, string) result
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
  val make_tensor : id:int -> shape:Tensor_shape.t -> dtype:Dtype.t -> t
  val id : t -> Value_id.t
  val shape : t -> Shape.t
  val logical_shape : t -> Tensor_shape.t
  val dtype : t -> Dtype.t
  val layout : t -> Tensor_layout.t
  val equal : t -> t -> bool
end

module Linear_storage : sig
  type layout =
    | Dense of Dtype.t
    | Block_quantized of Dtype.quant_type
    | Groupwise_packed of { bits : int; group_elements : int }

  type t = {
    layout : layout;
    weight : Value.t;
    scale : Value.t option;
    bias : Value.t option;
  }

  val classify :
    has_bias:bool ->
    weight:Value.t ->
    parameters:Value.t list ->
    (t, string) result

  val is_quantized : t -> bool
  val has_separate_scale : t -> bool
end

module Scalar : sig
  type t = Bool of bool | Int of int | Float of float
  val to_float : t -> float
  val to_string : t -> string
end

module Pointwise : sig
  type operand = Tensor of Value.t | Scalar of Scalar.t
  type unary =
    | Neg
    | Rsqrt
    | Silu
    | Cos
    | Sin
    | Tanh
    | Exp
    | Sigmoid
    | Softplus
    | Pow of Scalar.t
  type binary =
    | Add
    | Mul
    | Silu_mul
    | Gelu_mul
    | Sigmoid_mul
    | Sub
    | Div
    | Logical_and
    | Equal
    | Not_equal
    | Less_equal
    | Greater
  type t = Unary of unary * Value.t | Binary of binary * operand * operand

  val values : t -> Value.t list
  val to_string : t -> string
end

module Reduction : sig
  type operator = Mean | Sum
  type t = { operator : operator; axes : int list; keepdim : bool }
  val to_string : t -> string
end

module Movement : sig
  type t =
    | View
    | Reshape
    | Transpose of { axis0 : int; axis1 : int }
    | Unsqueeze of int
    | Expand
    | Contiguous
    | Index of Tensor_shape.Index.t
    | Concat of { axis : int }
    | Roll of { axis : int; shift : int }

  val to_string : t -> string
end

module Short_conv : sig
  type t

  val create :
    stride:int ->
    padding:int ->
    dilation:int ->
    groups:int ->
    (t, string) result

  val stride : t -> int
  val padding : t -> int
  val dilation : t -> int
  val groups : t -> int
  val to_string : t -> string
end

module Attention : sig
  type t

  val create : scale:float -> causal:bool -> (t, string) result
  val scale : t -> float
  val causal : t -> bool
  val to_string : t -> string
end

module Rms_rope : sig
  type t

  val create : epsilon:float -> half_dimension:int -> (t, string) result
  val epsilon : t -> float
  val half_dimension : t -> int
  val to_string : t -> string
end

module Short_conv_step : sig
  type t

  val create : channels:int -> window:int -> (t, string) result
  val channels : t -> int
  val window : t -> int
  val to_string : t -> string
end

module Short_conv_prefill : sig
  type t

  val create : channels:int -> window:int -> (t, string) result
  val channels : t -> int
  val window : t -> int
  val to_string : t -> string
end

module Paged_attention_q8 : sig
  type t

  val create :
    scale:float ->
    cache_layer:int ->
    attention_layers:int ->
    kv_heads:int ->
    group_size:int ->
    token_stride:int ->
    (t, string) result

  val scale : t -> float
  val cache_layer : t -> int
  val attention_layers : t -> int
  val kv_heads : t -> int
  val group_size : t -> int
  val token_stride : t -> int
  val to_string : t -> string
end

module Arange : sig
  type t

  val create : start:int -> stop:int -> step:int -> (t, string) result
  val start : t -> int
  val stop : t -> int
  val step : t -> int
  val to_string : t -> string
end

module Diff : sig
  type t

  val create : axis:int -> (t, string) result
  val axis : t -> int
  val to_string : t -> string
end

module Cumsum : sig
  type t

  val create : axis:int -> (t, string) result
  val axis : t -> int
  val to_string : t -> string
end

module Triangular_recurrence : sig
  type t

  val create : axis:int -> start:int -> stop:int -> (t, string) result
  val axis : t -> int
  val start : t -> int
  val stop : t -> int
  val to_string : t -> string
end

module Primitive : sig
  type t =
    | Pointwise of Pointwise.t
    | Cast of Dtype.t
    | Reduce of Reduction.t
    | Movement of Movement.t
    | Short_conv of Short_conv.t
    | Attention of Attention.t
    | Paged_attention_q8 of Paged_attention_q8.t
    | Embedding
    | Arange of Arange.t
    | Diff of Diff.t
    | Cumsum of Cumsum.t
    | Triangular_recurrence of Triangular_recurrence.t
    | Gated_delta
    | L2_norm of { epsilon : float }
    | Fill of Scalar.t
    | Gather2
    | Update_slice of Tensor_shape.Index.t
    | Pad_right_zero of { axis : int }
    | Triangular of { upper : bool; diagonal : int }
    | Masked_fill of Scalar.t
    | Eye
    | Batched_matmul

  val values : t -> Value.t list
  val to_string : t -> string
end

module Argument : sig
  type t =
    | Value of Value.t
    | Null
    | Ellipsis
    | Bool of bool
    | Int of int
    | Float of float
    | String of string
    | Symbol of string
    | List of t list
    | Tuple of t list
    | Mapping of (string * t) list
    | Slice of { start : t; stop : t; step : t }

  val values : t -> Value.t list
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
    | Rms_norm of { epsilon : float }
    | Rms_rope of Rms_rope.t
    | Short_conv_step of Short_conv_step.t
    | Short_conv_step_fused of Short_conv_step.t
    | Short_conv_prefill of Short_conv_prefill.t
    | Primitive of Primitive.t
    | Opaque of {
        op : string;
        target : string;
        arguments : Argument.t list;
        keyword_arguments : (string * Argument.t) list;
      }
    | Output of { name : string }
    | Barrier_create of { id : int; name : string }
    | Barrier_arrive of int
    | Barrier_wait of int
    | Fused_matmul_bias of { m : int; n : int; k : int }
    | W4a16_linear of { m : int; n : int; k : int; bias : bool }
    | W4a16_qkv_linear of {
        m : int;
        k : int;
        n_q : int;
        n_k : int;
        n_v : int;
        extra_outputs : Value.t list;
      }
    | W4a16_swiglu_ffn of { m : int; n : int; k : int; epsilon : float }
    | Rms_rope_qk of {
        q_heads : int;
        k_heads : int;
        width : int;
        half_dimension : int;
        epsilon : float;
        extra_outputs : Value.t list;
      }
    | W4a16_lm_head_argmax of {
        m : int;
        n : int;
        k : int;
        epsilon : float;
        extra_outputs : Value.t list;
      }
  val additional_outputs : t -> Value.t list

  val to_string : t -> string
end

val node_id : node -> int
val node_op : node -> Op.t
val node_inputs : node -> Value.t list
val node_output : node -> Value.t option
val node_replace : node -> op:Op.t -> inputs:Value.t list -> node
val node_reindex : node -> int -> node
val node_create :
  id:int -> op:Op.t -> inputs:Value.t list -> output:Value.t option -> node

module Graph : sig
  type t

  val create : unit -> t
  val fresh_value : t -> shape:Shape.t -> dtype:Dtype.t -> Value.t
  val fresh_tensor_value : t -> shape:Tensor_shape.t -> dtype:Dtype.t -> Value.t
  val input :
    t ->
    name:string ->
    source:Input_source.t ->
    shape:Shape.t ->
    dtype:Dtype.t ->
    Value.t
  val tensor_input :
    t ->
    name:string ->
    source:Input_source.t ->
    shape:Tensor_shape.t ->
    dtype:Dtype.t ->
    Value.t
  val allocate :
    t -> shape:Shape.t -> dtype:Dtype.t -> space:Memory_space.t -> layout:Layout.t -> Value.t
  val append : t -> op:Op.t -> inputs:Value.t list -> output:Value.t option -> unit
  val add_output : t -> name:string -> Value.t -> unit
  val nodes : t -> node list
  val outputs : t -> (string * Value.t) list
  val with_nodes : t -> node list -> t
  val with_nodes_and_outputs : t -> node list -> (string * Value.t) list -> t
  val pp : Format.formatter -> t -> unit
end

module Debug : sig
  val to_sexp_string : Graph.t -> string
end
