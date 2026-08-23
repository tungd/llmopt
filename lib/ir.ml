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
    dtype : Dtype.t;
  }

  let make ~id ~shape ~dtype = { id; shape; dtype }
  let id value = value.id
  let shape value = value.shape
  let dtype value = value.dtype
  let equal left right = left.id = right.id
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
    | Opaque of { op : string; target : string }
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
    | Opaque { op; target } -> Printf.sprintf "opaque(%s,%s)" op target
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

  let append graph ~op ~inputs ~output =
    let node = { id = graph.next_node; op; inputs; output } in
    graph.next_node <- graph.next_node + 1;
    graph.nodes_rev <- node :: graph.nodes_rev

  let input graph ~name ~source ~shape ~dtype =
    let value = fresh_value graph ~shape ~dtype in
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
        (Shape.to_string (Value.shape value))
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
