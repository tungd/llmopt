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
  type operator = Mean
  type t = { operator : operator; axes : int list; keepdim : bool }

  let to_string reduction =
    let axes = reduction.axes |> List.map string_of_int |> String.concat "," in
    Printf.sprintf "mean(axes=[%s],keepdim=%b)" axes reduction.keepdim
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

module Primitive = struct
  type t =
    | Pointwise of Pointwise.t
    | Cast of Dtype.t
    | Reduce of Reduction.t
    | Movement of Movement.t
    | Short_conv of Short_conv.t
    | Attention of Attention.t
    | Embedding

  let values = function
    | Pointwise operation -> Pointwise.values operation
    | Cast _ | Reduce _ | Movement _ | Short_conv _ | Attention _ | Embedding ->
        []

  let to_string = function
    | Pointwise operation -> Pointwise.to_string operation
    | Cast dtype -> "cast(" ^ Dtype.to_string dtype ^ ")"
    | Reduce reduction -> Reduction.to_string reduction
    | Movement movement -> Movement.to_string movement
    | Short_conv config -> Short_conv.to_string config
    | Attention config -> Attention.to_string config
    | Embedding -> "embedding"
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
