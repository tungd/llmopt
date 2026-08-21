open Yojson.Basic
open Yojson.Basic.Util

module Node = struct
  type t = {
    name : string;
    op : string;
    target : string;
    inputs : string list;
    shape : int list option;
    dtype : Ir.Dtype.t;
  }

  let name node = node.name
  let op node = node.op
  let target node = node.target
  let inputs node = node.inputs
  let shape node = node.shape
  let dtype node = node.dtype
end

type t = { nodes : Node.t list; outputs : string list }

let nodes graph = graph.nodes
let outputs graph = graph.outputs

let member_opt key json =
  match json |> member key with
  | `Null -> None
  | value -> Some value

let parse_dtype = function
  | "float32" | "torch.float32" | "f32" -> Ok Ir.Dtype.Float32
  | "float16" | "torch.float16" | "half" | "f16" -> Ok Ir.Dtype.Float16
  | "bfloat16" | "torch.bfloat16" | "bf16" -> Ok Ir.Dtype.Bfloat16
  | "int64" | "torch.int64" | "long" | "i64" -> Ok Ir.Dtype.Int64
  | "int32" | "torch.int32" | "int" | "i32" -> Ok Ir.Dtype.Int32
  | "int8" | "torch.int8" | "i8" -> Ok Ir.Dtype.Int8
  | "bool" | "torch.bool" -> Ok Ir.Dtype.Bool
  | value -> Error ("unsupported FX dtype: " ^ value)

let parse_shape json =
  match member_opt "shape" json with
  | None -> Ok None
  | Some (`List values) ->
      let rec collect acc = function
        | [] -> Ok (Some (List.rev acc))
        | `Int value :: rest -> collect (value :: acc) rest
        | `Float value :: rest when Float.is_integer value ->
            collect (int_of_float value :: acc) rest
        | _ -> Ok None
      in
      collect [] values
  | Some _ -> Error "FX node shape must be a list or null"

let parse_inputs json =
  match member_opt "inputs" json with
  | None -> Ok []
  | Some (`List values) ->
      let rec collect acc = function
        | [] -> Ok (List.rev acc)
        | `String value :: rest -> collect (value :: acc) rest
        | _ -> Error "FX node inputs must contain node names"
      in
      collect [] values
  | Some _ -> Error "FX node inputs must be a list"

let parse_node json =
  let name = json |> member "name" |> to_string in
  let op = json |> member "op" |> to_string in
  let target = json |> member "target" |> to_string_option |> Option.value ~default:"" in
  let inputs = parse_inputs json in
  let shape = parse_shape json in
  let dtype =
    json |> member "dtype" |> to_string_option |> Option.value ~default:"float32"
    |> parse_dtype
  in
  match inputs, shape, dtype with
  | Ok inputs, Ok shape, Ok dtype -> Ok { Node.name; op; target; inputs; shape; dtype }
  | Error message, _, _ | _, Error message, _ | _, _, Error message -> Error message

let parse_outputs json =
  match json |> member "outputs" with
  | `List values ->
      let rec collect acc = function
        | [] -> Ok (List.rev acc)
        | `String value :: rest -> collect (value :: acc) rest
        | _ -> Error "FX outputs must contain node names"
      in
      collect [] values
  | _ -> Error "FX manifest outputs must be a list"

let of_json json =
  try
    let nodes_json = json |> member "nodes" |> to_list in
    let rec parse_nodes acc = function
      | [] -> Ok (List.rev acc)
      | value :: rest ->
          (match parse_node value with
          | Ok node -> parse_nodes (node :: acc) rest
          | Error message -> Error message)
    in
    match parse_nodes [] nodes_json, parse_outputs json with
    | Ok nodes, Ok outputs -> Ok { nodes; outputs }
    | Error message, _ | _, Error message -> Error message
  with
  | Yojson.Json_error message -> Error ("invalid FX manifest: " ^ message)
  | Type_error (message, _) -> Error ("invalid FX manifest: " ^ message)

let of_file path =
  try of_json (Yojson.Basic.from_file path)
  with Sys_error message -> Error ("cannot read FX manifest: " ^ message)
