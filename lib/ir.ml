module Dtype = struct
  type t = Float32 | Float16 | Bfloat16 | Int64 | Int32 | Int8 | Bool

  let to_string = function
    | Float32 -> "f32"
    | Float16 -> "f16"
    | Bfloat16 -> "bf16"
    | Int64 -> "i64"
    | Int32 -> "i32"
    | Int8 -> "i8"
    | Bool -> "bool"
end

module Quantization = struct
  type t = Fp16 | Q8_weight_only

  let to_string = function
    | Fp16 -> "fp16"
    | Q8_weight_only -> "q8-weight-only"
end

module Memory_space = struct
  type t = Global | Shared | Register | Private

  let to_string = function
    | Global -> "global"
    | Shared -> "shared"
    | Register -> "register"
    | Private -> "private"
end

module Layout = struct
  type t = Row_major | Col_major | Xor_swizzle of int

  let to_string = function
    | Row_major -> "row-major"
    | Col_major -> "col-major"
    | Xor_swizzle mask -> Printf.sprintf "xor-swizzle(0x%x)" mask
end

module Input_source = struct
  type t = Runtime | Tensor_store of { key : string }

  let to_string = function
    | Runtime -> "runtime"
    | Tensor_store { key } -> "tensor:" ^ key
end

module Value_id = struct
  type t = int
  let compare = Int.compare
  let to_int value = value
end

module Value = struct
  type t = {
    id : Value_id.t;
    shape : Shape.t;
    logical_shape : Tensor_shape.t;
    dtype : Dtype.t;
  }

  let make ~id ~shape ~dtype =
    { id; shape; logical_shape = Tensor_shape.of_matrix shape; dtype }

  let make_tensor ~id ~shape ~dtype =
    { id; shape = Tensor_shape.matrix_exn shape; logical_shape = shape; dtype }

  let id value = value.id
  let shape value = value.shape
  let logical_shape value = value.logical_shape
  let dtype value = value.dtype
  let equal left right = left.id = right.id
end

module Scalar = struct
  type t = Bool of bool | Int of int | Float of float

  let to_float = function
    | Bool false -> 0.0
    | Bool true -> 1.0
    | Int value -> Float.of_int value
    | Float value -> value

  let to_string = function
    | Bool value -> string_of_bool value
    | Int value -> string_of_int value
    | Float value -> Printf.sprintf "%.17g" value
end

module Pointwise = struct
  type operand = Tensor of Value.t | Scalar of Scalar.t
  type unary = Neg | Rsqrt | Silu | Cos | Sin | Pow of Scalar.t
  type binary = Add | Mul | Sub | Logical_and | Equal | Not_equal | Less_equal
  type t = Unary of unary * Value.t | Binary of binary * operand * operand

  let values = function
    | Unary (_, value) -> [ value ]
    | Binary (_, left, right) ->
        [ left; right ]
        |> List.filter_map (function Tensor value -> Some value | Scalar _ -> None)

  let unary_to_string = function
    | Neg -> "neg"
    | Rsqrt -> "rsqrt"
    | Silu -> "silu"
    | Cos -> "cos"
    | Sin -> "sin"
    | Pow exponent -> "pow(" ^ Scalar.to_string exponent ^ ")"

  let binary_to_string = function
    | Add -> "add"
    | Mul -> "mul"
    | Sub -> "sub"
    | Logical_and -> "and"
    | Equal -> "eq"
    | Not_equal -> "ne"
    | Less_equal -> "le"

  let to_string = function
    | Unary (operator, _) -> unary_to_string operator
    | Binary (operator, _, _) -> binary_to_string operator
end

module Reduction = struct
  type operator = Mean | Sum
  type t = { operator : operator; axes : int list; keepdim : bool }

  let to_string reduction =
    let axes = reduction.axes |> List.map string_of_int |> String.concat "," in
    let operator = match reduction.operator with Mean -> "mean" | Sum -> "sum" in
    Printf.sprintf "%s(axes=[%s],keepdim=%b)" operator axes reduction.keepdim
end

module Movement = struct
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

  let to_string = function
    | View -> "view"
    | Reshape -> "reshape"
    | Transpose { axis0; axis1 } ->
        Printf.sprintf "transpose(%d,%d)" axis0 axis1
    | Unsqueeze axis -> Printf.sprintf "unsqueeze(%d)" axis
    | Expand -> "expand"
    | Contiguous -> "contiguous"
    | Index index -> Tensor_shape.Index.to_string index
    | Concat { axis } -> Printf.sprintf "concat(axis=%d)" axis
    | Roll { axis; shift } -> Printf.sprintf "roll(axis=%d,shift=%d)" axis shift
end

module Short_conv = struct
  type t = {
    stride : int;
    padding : int;
    dilation : int;
    groups : int;
  }

  let create ~stride ~padding ~dilation ~groups =
    if stride <= 0 then Error "short-conv stride must be positive"
    else if padding < 0 then Error "short-conv padding must be non-negative"
    else if dilation <= 0 then Error "short-conv dilation must be positive"
    else if groups <= 0 then Error "short-conv groups must be positive"
    else Ok { stride; padding; dilation; groups }

  let stride config = config.stride
  let padding config = config.padding
  let dilation config = config.dilation
  let groups config = config.groups

  let to_string config =
    Printf.sprintf "short-conv(stride=%d,padding=%d,dilation=%d,groups=%d)"
      config.stride config.padding config.dilation config.groups
end

module Attention = struct
  type t = { scale : float; causal : bool }

  let create ~scale ~causal =
    if Float.is_finite scale then Ok { scale; causal }
    else Error "attention scale must be finite"

  let scale config = config.scale
  let causal config = config.causal

  let to_string config =
    Printf.sprintf "attention(scale=%.9g,causal=%b)" config.scale config.causal
end

module Rms_rope = struct
  type t = { epsilon : float; half_dimension : int }

  let create ~epsilon ~half_dimension =
    if not (Float.is_finite epsilon) then
      Error "RMSNorm-RoPE epsilon must be finite"
    else if half_dimension <= 0 then
      Error "RMSNorm-RoPE half dimension must be positive"
    else Ok { epsilon; half_dimension }

  let epsilon config = config.epsilon
  let half_dimension config = config.half_dimension

  let to_string config =
    Printf.sprintf "rms-rope(eps=%.9g,half=%d)" config.epsilon
      config.half_dimension
end

module Short_conv_step = struct
  type t = { channels : int; window : int }

  let create ~channels ~window =
    if channels <= 0 then Error "short-conv-step channels must be positive"
    else if window <= 0 then Error "short-conv-step window must be positive"
    else Ok { channels; window }

  let channels config = config.channels
  let window config = config.window

  let to_string config =
    Printf.sprintf "short-conv-step(channels=%d,window=%d)" config.channels
      config.window
end

module Short_conv_prefill = struct
  type t = { channels : int; window : int }

  let create ~channels ~window =
    if channels <= 0 then Error "short-conv-prefill channels must be positive"
    else if window <= 0 then Error "short-conv-prefill window must be positive"
    else Ok { channels; window }

  let channels config = config.channels
  let window config = config.window

  let to_string config =
    Printf.sprintf "short-conv-prefill(channels=%d,window=%d)" config.channels
      config.window
end

module Paged_attention_q8 = struct
  type t = {
    scale : float;
    cache_layer : int;
    attention_layers : int;
    kv_heads : int;
    group_size : int;
    token_stride : int;
  }

  let create ~scale ~cache_layer ~attention_layers ~kv_heads ~group_size
      ~token_stride =
    if not (Float.is_finite scale) then Error "paged Q8 attention scale must be finite"
    else if attention_layers <= 0 then
      Error "paged Q8 attention requires at least one attention layer"
    else if cache_layer < 0 || cache_layer >= attention_layers then
      Error "paged Q8 attention cache layer is outside the attention layout"
    else if kv_heads <= 0 then Error "paged Q8 attention requires positive KV heads"
    else if group_size <= 0 then
      Error "paged Q8 attention requires a positive group size"
    else if token_stride <= 0 then
      Error "paged Q8 attention requires a positive token stride"
    else
      Ok
        {
          scale;
          cache_layer;
          attention_layers;
          kv_heads;
          group_size;
          token_stride;
        }

  let scale config = config.scale
  let cache_layer config = config.cache_layer
  let attention_layers config = config.attention_layers
  let kv_heads config = config.kv_heads
  let group_size config = config.group_size
  let token_stride config = config.token_stride

  let to_string config =
    Printf.sprintf
      "paged-attention-q8(scale=%.9g,layer=%d/%d,kv-heads=%d,group=%d,stride=%d)"
      config.scale config.cache_layer config.attention_layers config.kv_heads
      config.group_size config.token_stride
end

module Arange = struct
  type t = { start : int; stop : int; step : int }

  let create ~start ~stop ~step =
    if step = 0 then Error "arange step must be non-zero"
    else Ok { start; stop; step }

  let start config = config.start
  let stop config = config.stop
  let step config = config.step

  let to_string config =
    Printf.sprintf "arange(start=%d,stop=%d,step=%d)" config.start config.stop
      config.step
end

module Diff = struct
  type t = { axis : int }

  let create ~axis =
    if axis < 0 then Error "diff axis must be normalized"
    else Ok { axis }

  let axis config = config.axis
  let to_string config = Printf.sprintf "diff(axis=%d,prepend=true)" config.axis
end

module Cumsum = struct
  type t = { axis : int }

  let create ~axis =
    if axis < 0 then Error "cumsum axis must be normalized"
    else Ok { axis }

  let axis config = config.axis
  let to_string config = Printf.sprintf "cumsum(axis=%d)" config.axis
end

module Primitive = struct
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
    | Fill of Scalar.t
    | Gather2
    | Update_slice of Tensor_shape.Index.t

  let values = function
    | Pointwise operation -> Pointwise.values operation
    | Cast _ | Reduce _ | Movement _ | Short_conv _ | Attention _
    | Paged_attention_q8 _ | Embedding | Arange _ | Diff _ | Cumsum _ | Fill _
    | Gather2 | Update_slice _ -> []

  let to_string = function
    | Pointwise operation -> Pointwise.to_string operation
    | Cast dtype -> "cast(" ^ Dtype.to_string dtype ^ ")"
    | Reduce reduction -> Reduction.to_string reduction
    | Movement movement -> Movement.to_string movement
    | Short_conv config -> Short_conv.to_string config
    | Attention config -> Attention.to_string config
    | Paged_attention_q8 config -> Paged_attention_q8.to_string config
    | Embedding -> "embedding"
    | Arange config -> Arange.to_string config
    | Diff config -> Diff.to_string config
    | Cumsum config -> Cumsum.to_string config
    | Fill scalar -> "fill(" ^ Scalar.to_string scalar ^ ")"
    | Gather2 -> "gather2"
    | Update_slice index -> "update-" ^ Tensor_shape.Index.to_string index
end

module Argument = struct
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

  let rec values = function
    | Value value -> [ value ]
    | Null | Ellipsis | Bool _ | Int _ | Float _ | String _ | Symbol _ -> []
    | List arguments | Tuple arguments -> List.concat_map values arguments
    | Mapping fields -> fields |> List.map snd |> List.concat_map values
    | Slice { start; stop; step } -> values start @ values stop @ values step
end

module Op = struct
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
    | Q8_linear of { m : int; n : int; k : int; bias : bool }
    | Q8_linear_silu of { m : int; n : int; k : int; bias : bool }
    | Q8_linear_add of { m : int; n : int; k : int; bias : bool }
    | Q8_linear_mul_add of { m : int; n : int; k : int; bias : bool }
    | Q8_dual_linear of { m : int; n1 : int; n2 : int; k : int; bias : bool }
    | Q8_qkv_linear of {
        m : int;
        n_q : int;
        n_kv : int;
        k : int;
        bias : bool;
      }

  let to_string = function
    | Input { name; source } ->
        Printf.sprintf "input(%s,%s)" name (Input_source.to_string source)
    | Alloc { space; layout } ->
        Printf.sprintf "alloc(%s,%s)" (Memory_space.to_string space)
          (Layout.to_string layout)
    | Copy { asynchronous; barrier } ->
        let kind = if asynchronous then "async-copy" else "copy" in
        let barrier =
          match barrier with
          | None -> ""
          | Some id -> Printf.sprintf ",barrier=%d" id
        in
        kind ^ barrier
    | Matmul { m; n; k } -> Printf.sprintf "matmul[%dx%dx%d]" m n k
    | Linear { m; n; k; bias = false } -> Printf.sprintf "linear[%dx%dx%d]" m n k
    | Linear { m; n; k; bias = true } -> Printf.sprintf "linear+bias[%dx%dx%d]" m n k
    | Add { broadcast = Shape.Same } -> "add[same]"
    | Add { broadcast = Shape.Row } -> "add[row-broadcast]"
    | Gelu -> "gelu"
    | Relu -> "relu"
    | Rms_norm { epsilon } -> Printf.sprintf "rms-norm(eps=%.9g)" epsilon
    | Rms_rope config -> Rms_rope.to_string config
    | Short_conv_step config -> Short_conv_step.to_string config
    | Short_conv_step_fused config ->
        "short-conv-step-fused(" ^ Short_conv_step.to_string config ^ ")"
    | Short_conv_prefill config -> Short_conv_prefill.to_string config
    | Primitive primitive -> Primitive.to_string primitive
    | Opaque { op; target; _ } -> Printf.sprintf "opaque(%s,%s)" op target
    | Output { name } -> Printf.sprintf "output(%s)" name
    | Barrier_create { id; name } -> Printf.sprintf "barrier-create(%d,%s)" id name
    | Barrier_arrive id -> Printf.sprintf "barrier-arrive(%d)" id
    | Barrier_wait id -> Printf.sprintf "barrier-wait(%d)" id
    | Fused_matmul_bias { m; n; k } ->
        Printf.sprintf "fused-matmul-bias[%dx%dx%d]" m n k
    | Q8_linear { m; n; k; bias = false } ->
        Printf.sprintf "q8-linear[%dx%dx%d]" m n k
    | Q8_linear { m; n; k; bias = true } ->
        Printf.sprintf "q8-linear+bias[%dx%dx%d]" m n k
    | Q8_linear_silu { m; n; k; bias = false } ->
        Printf.sprintf "q8-linear+silu[%dx%dx%d]" m n k
    | Q8_linear_silu { m; n; k; bias = true } ->
        Printf.sprintf "q8-linear+bias+silu[%dx%dx%d]" m n k
    | Q8_linear_add { m; n; k; bias = false } ->
        Printf.sprintf "q8-linear+add[%dx%dx%d]" m n k
    | Q8_linear_add { m; n; k; bias = true } ->
        Printf.sprintf "q8-linear+bias+add[%dx%dx%d]" m n k
    | Q8_linear_mul_add { m; n; k; bias = false } ->
        Printf.sprintf "q8-linear+mul+add[%dx%dx%d]" m n k
    | Q8_linear_mul_add { m; n; k; bias = true } ->
        Printf.sprintf "q8-linear+bias+mul+add[%dx%dx%d]" m n k
    | Q8_dual_linear { m; n1; n2; k; bias = false } ->
        Printf.sprintf "q8-dual-linear[%dx(%d+%d)x%d]" m n1 n2 k
    | Q8_dual_linear { m; n1; n2; k; bias = true } ->
        Printf.sprintf "q8-dual-linear+bias[%dx(%d+%d)x%d]" m n1 n2 k
    | Q8_qkv_linear { m; n_q; n_kv; k; bias = false } ->
        Printf.sprintf "q8-qkv-linear[%dx(%d+%d+%d)x%d]" m n_q n_kv n_kv k
    | Q8_qkv_linear { m; n_q; n_kv; k; bias = true } ->
        Printf.sprintf "q8-qkv-linear+bias[%dx(%d+%d+%d)x%d]" m n_q n_kv n_kv
          k
end

type node = {
  id : int;
  op : Op.t;
  inputs : Value.t list;
  output : Value.t option;
}

let node_id node = node.id
let node_op node = node.op
let node_inputs node = node.inputs
let node_output node = node.output
let node_replace node ~op ~inputs = { node with op; inputs }
let node_reindex node id = { node with id }
let node_create ~id ~op ~inputs ~output = { id; op; inputs; output }

module Graph = struct
  type t = {
    mutable next_value : int;
    mutable next_node : int;
    mutable nodes_rev : node list;
    mutable outputs_rev : (string * Value.t) list;
  }

  let create () =
    { next_value = 0; next_node = 0; nodes_rev = []; outputs_rev = [] }

  let fresh_value graph ~shape ~dtype =
    let value = Value.make ~id:graph.next_value ~shape ~dtype in
    graph.next_value <- graph.next_value + 1;
    value

  let fresh_tensor_value graph ~shape ~dtype =
    let value = Value.make_tensor ~id:graph.next_value ~shape ~dtype in
    graph.next_value <- graph.next_value + 1;
    value

  let append graph ~op ~inputs ~output =
    let node = { id = graph.next_node; op; inputs; output } in
    graph.next_node <- graph.next_node + 1;
    graph.nodes_rev <- node :: graph.nodes_rev

  let input graph ~name ~source ~shape ~dtype =
    let value = fresh_value graph ~shape ~dtype in
    append graph ~op:(Op.Input { name; source }) ~inputs:[] ~output:(Some value);
    value

  let tensor_input graph ~name ~source ~shape ~dtype =
    let value = fresh_tensor_value graph ~shape ~dtype in
    append graph ~op:(Op.Input { name; source }) ~inputs:[] ~output:(Some value);
    value

  let allocate graph ~shape ~dtype ~space ~layout =
    let value = fresh_value graph ~shape ~dtype in
    append graph ~op:(Op.Alloc { space; layout }) ~inputs:[] ~output:(Some value);
    value

  let add_output graph ~name value =
    append graph ~op:(Op.Output { name }) ~inputs:[ value ] ~output:None;
    graph.outputs_rev <- (name, value) :: graph.outputs_rev

  let nodes graph = List.rev graph.nodes_rev
  let outputs graph = List.rev graph.outputs_rev

  let with_nodes graph nodes = { graph with nodes_rev = List.rev nodes }

  let pp formatter graph =
    let pp_value formatter value =
      Format.fprintf formatter "%%%d:%s:%s"
        (Value_id.to_int (Value.id value))
        (Tensor_shape.to_string (Value.logical_shape value))
        (Dtype.to_string (Value.dtype value))
    in
    List.iter
      (fun node ->
        Format.fprintf formatter "n%d %s (" node.id (Op.to_string node.op);
        List.iteri
          (fun index value ->
            if index > 0 then Format.fprintf formatter ", ";
            pp_value formatter value)
          node.inputs;
        Format.fprintf formatter ")";
        (match node.output with
        | None -> ()
        | Some value -> Format.fprintf formatter " -> %a" pp_value value);
        Format.fprintf formatter "@.")
      (nodes graph)
end
