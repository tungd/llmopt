open Yojson.Basic
open Yojson.Basic.Util

let ( let* ) = Result.bind

module Binding = struct
  type t = Computed | Runtime | Tensor_store of { key : string }

  let input_source = function
    | Computed -> None
    | Runtime -> Some Ir.Input_source.Runtime
    | Tensor_store { key } -> Some (Ir.Input_source.Tensor_store { key })
end

module Argument = struct
  type t =
    | Node of string
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

  let rec node_references = function
    | Node name -> [ name ]
    | Null | Ellipsis | Bool _ | Int _ | Float _ | String _ | Symbol _ -> []
    | List values | Tuple values -> List.concat_map node_references values
    | Mapping fields ->
        fields |> List.map snd |> List.concat_map node_references
    | Slice { start; stop; step } ->
        node_references start @ node_references stop @ node_references step
end

module Node = struct
  type t = {
    name : string;
    op : string;
    target : string;
    inputs : string list;
    shape : int list option;
    dtype : Ir.Dtype.t;
    binding : Binding.t;
    arguments : Argument.t list;
    keyword_arguments : (string * Argument.t) list;
  }

  let name node = node.name
  let op node = node.op
  let target node = node.target
  let inputs node = node.inputs
  let shape node = node.shape
  let dtype node = node.dtype
  let binding node = node.binding
  let arguments node = node.arguments
  let keyword_arguments node = node.keyword_arguments
end

type t = { version : int; nodes : Node.t list; outputs : string list }

let version graph = graph.version
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

let parse_binding ~op json =
  match member_opt "binding" json with
  | None ->
      if op = "placeholder" || op = "get_attr" then Ok Binding.Runtime
      else Ok Binding.Computed
  | Some value ->
      (match value |> member "kind" |> to_string with
      | "computed" -> Ok Binding.Computed
      | "runtime" -> Ok Binding.Runtime
      | "tensor-store" ->
          let key = value |> member "key" |> to_string in
          if String.trim key = "" then Error "FX tensor binding key cannot be empty"
          else Ok (Binding.Tensor_store { key })
      | kind -> Error ("unsupported FX binding kind: " ^ kind))

let rec parse_argument json =
  let parse_values values =
    let rec loop acc = function
      | [] -> Ok (List.rev acc)
      | value :: rest ->
          let* value = parse_argument value in
          loop (value :: acc) rest
    in
    loop [] values
  in
  try
    match json |> member "kind" |> to_string with
    | "node" -> Ok (Argument.Node (json |> member "name" |> to_string))
    | "null" -> Ok Argument.Null
    | "ellipsis" -> Ok Argument.Ellipsis
    | "bool" -> Ok (Argument.Bool (json |> member "value" |> to_bool))
    | "int" -> Ok (Argument.Int (json |> member "value" |> to_int))
    | "float" -> Ok (Argument.Float (json |> member "value" |> to_float))
    | "string" -> Ok (Argument.String (json |> member "value" |> to_string))
    | "symbol" -> Ok (Argument.Symbol (json |> member "value" |> to_string))
    | "list" ->
        let* values = json |> member "items" |> to_list |> parse_values in
        Ok (Argument.List values)
    | "tuple" ->
        let* values = json |> member "items" |> to_list |> parse_values in
        Ok (Argument.Tuple values)
    | "mapping" ->
        let rec fields acc = function
          | [] -> Ok (Argument.Mapping (List.rev acc))
          | field :: rest ->
              let name = field |> member "name" |> to_string in
              let* value = field |> member "value" |> parse_argument in
              fields ((name, value) :: acc) rest
        in
        fields [] (json |> member "items" |> to_list)
    | "slice" ->
        let* start = json |> member "start" |> parse_argument in
        let* stop = json |> member "stop" |> parse_argument in
        let* step = json |> member "step" |> parse_argument in
        Ok (Argument.Slice { start; stop; step })
    | kind -> Error ("unsupported FX argument kind: " ^ kind)
  with Type_error (message, _) -> Error ("invalid FX argument: " ^ message)

let parse_arguments ~version json =
  match member_opt "arguments" json with
  | None when version = 1 -> Ok ([], [])
  | None -> Error "FX manifest v2 node is missing typed arguments"
  | Some arguments ->
      let rec positional acc = function
        | [] -> Ok (List.rev acc)
        | value :: rest ->
            let* value = parse_argument value in
            positional (value :: acc) rest
      in
      let rec keywords acc = function
        | [] -> Ok (List.rev acc)
        | field :: rest ->
            let name = field |> member "name" |> to_string in
            let* value = field |> member "value" |> parse_argument in
            keywords ((name, value) :: acc) rest
      in
      let* args = arguments |> member "args" |> to_list |> positional [] in
      let* kwargs =
        arguments |> member "kwargs" |> to_list |> keywords []
      in
      Ok (args, kwargs)

let parse_node ~version json =
  let name = json |> member "name" |> to_string in
  let op = json |> member "op" |> to_string in
  let target = json |> member "target" |> to_string_option |> Option.value ~default:"" in
  let inputs = parse_inputs json in
  let shape = parse_shape json in
  let dtype =
    json |> member "dtype" |> to_string_option |> Option.value ~default:"float32"
    |> parse_dtype
  in
  let binding = parse_binding ~op json in
  let arguments = parse_arguments ~version json in
  match inputs, shape, dtype, binding, arguments with
  | ( Ok inputs,
      Ok shape,
      Ok dtype,
      Ok binding,
      Ok (arguments, keyword_arguments) ) ->
      let argument_inputs =
        arguments @ List.map snd keyword_arguments
        |> List.concat_map Argument.node_references
        |> List.sort_uniq String.compare
      in
      let declared_inputs = List.sort_uniq String.compare inputs in
      if argument_inputs <> [] && argument_inputs <> declared_inputs then
        Error ("FX node argument references disagree with inputs: " ^ name)
      else
        Ok
          {
            Node.name;
            op;
            target;
            inputs;
            shape;
            dtype;
            binding;
            arguments;
            keyword_arguments;
          }
  | Error message, _, _, _, _
  | _, Error message, _, _, _
  | _, _, Error message, _, _
  | _, _, _, Error message, _
  | _, _, _, _, Error message -> Error message

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
    let version =
      json |> member "version" |> to_int_option |> Option.value ~default:1
    in
    if version <> 1 && version <> 2 then
      Error (Printf.sprintf "unsupported FX manifest version: %d" version)
    else
      let nodes_json = json |> member "nodes" |> to_list in
      let rec parse_nodes acc = function
        | [] -> Ok (List.rev acc)
        | value :: rest ->
            (match parse_node ~version value with
            | Ok node -> parse_nodes (node :: acc) rest
            | Error message -> Error message)
      in
      match parse_nodes [] nodes_json, parse_outputs json with
      | Ok nodes, Ok outputs -> Ok { version; nodes; outputs }
      | Error message, _ | _, Error message -> Error message
  with
  | Yojson.Json_error message -> Error ("invalid FX manifest: " ^ message)
  | Type_error (message, _) -> Error ("invalid FX manifest: " ^ message)

let of_file path =
  try of_json (Yojson.Basic.from_file path)
  with Sys_error message -> Error ("cannot read FX manifest: " ^ message)
