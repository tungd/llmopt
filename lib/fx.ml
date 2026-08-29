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

type t = { nodes : Node.t list; outputs : string list }

let binary_magic = "LLMOPTFX"
let nodes graph = graph.nodes
let outputs graph = graph.outputs

let make_node ~name ~op ~target ~inputs ~shape ~dtype ~binding ~arguments
    ~keyword_arguments =
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

let read_n reader count read =
  let rec loop remaining acc =
    if remaining = 0 then Ok (List.rev acc)
    else
      let* value = read reader in
      loop (remaining - 1) (value :: acc)
  in
  loop count []

let read_counted reader read =
  let* count = Binary.Reader.u32 reader in
  read_n reader count read

let dtype_of_tag = function
  | 0 -> Ok Ir.Dtype.Float32
  | 1 -> Ok Ir.Dtype.Float16
  | 2 -> Ok Ir.Dtype.Bfloat16
  | 3 -> Ok Ir.Dtype.Int64
  | 4 -> Ok Ir.Dtype.Int32
  | 5 -> Ok Ir.Dtype.Int8
  | 6 -> Ok Ir.Dtype.Bool
  | 7 -> Ok Ir.Dtype.UInt8
  | 8 -> Ok (Ir.Dtype.Quant Q8_0)
  | 9 -> Ok (Ir.Dtype.Quant Q4_K)
  | 10 -> Ok (Ir.Dtype.Quant Q5_K)
  | 11 -> Ok (Ir.Dtype.Quant Q6_K)
  | 12 -> Ok (Ir.Dtype.Quant Q5_0)
  | 13 -> Ok (Ir.Dtype.Quant Q4_0)
  | 14 -> Ok (Ir.Dtype.Quant IQ4_XS)
  | tag -> Error (Printf.sprintf "unknown binary FX dtype tag: %d" tag)

let read_binary_binding reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 -> Ok Binding.Computed
  | 1 -> Ok Binding.Runtime
  | 2 ->
      let* key = Binary.Reader.string reader in
      if String.trim key = "" then Error "FX tensor binding key cannot be empty"
      else Ok (Binding.Tensor_store { key })
  | tag -> Error (Printf.sprintf "unknown binary FX binding tag: %d" tag)

let read_binary_shape reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 -> Ok None
  | 1 ->
      let* rank = Binary.Reader.u16 reader in
      read_n reader rank Binary.Reader.u64 |> Result.map Option.some
  | tag -> Error (Printf.sprintf "unknown binary FX shape tag: %d" tag)

let rec read_binary_argument ?(depth = 0) reader =
  if depth > 64 then Error "FX argument nesting exceeds 64 levels"
  else
    let nested = read_binary_argument ~depth:(depth + 1) in
    let* tag = Binary.Reader.u8 reader in
    match tag with
    | 0 -> Binary.Reader.string reader |> Result.map (fun name -> Argument.Node name)
    | 1 -> Ok Argument.Null
    | 2 -> Ok Argument.Ellipsis
    | 3 -> Binary.Reader.bool reader |> Result.map (fun value -> Argument.Bool value)
    | 4 -> Binary.Reader.i64 reader |> Result.map (fun value -> Argument.Int value)
    | 5 ->
        let* value = Binary.Reader.float64 reader in
        if Float.is_finite value then Ok (Argument.Float value)
        else Error "binary FX graph contains a non-finite float"
    | 6 ->
        Binary.Reader.string reader |> Result.map (fun value -> Argument.String value)
    | 7 ->
        Binary.Reader.string reader |> Result.map (fun value -> Argument.Symbol value)
    | 8 -> read_counted reader nested |> Result.map (fun values -> Argument.List values)
    | 9 -> read_counted reader nested |> Result.map (fun values -> Argument.Tuple values)
    | 10 ->
        let read_field reader =
          let* name = Binary.Reader.string reader in
          let* value = nested reader in
          Ok (name, value)
        in
        read_counted reader read_field
        |> Result.map (fun fields -> Argument.Mapping fields)
    | 11 ->
        let* start = nested reader in
        let* stop = nested reader in
        let* step = nested reader in
        Ok (Argument.Slice { start; stop; step })
    | tag -> Error (Printf.sprintf "unknown binary FX argument tag: %d" tag)

let read_binary_node reader =
  let* name = Binary.Reader.string reader in
  let* op = Binary.Reader.string reader in
  let* target = Binary.Reader.string reader in
  let* dtype_tag = Binary.Reader.u8 reader in
  let* dtype = dtype_of_tag dtype_tag in
  let* binding = read_binary_binding reader in
  let* shape = read_binary_shape reader in
  let* inputs = read_counted reader Binary.Reader.string in
  let* arguments = read_counted reader read_binary_argument in
  let read_keyword reader =
    let* name = Binary.Reader.string reader in
    let* value = read_binary_argument reader in
    Ok (name, value)
  in
  let* keyword_arguments = read_counted reader read_keyword in
  make_node ~name ~op ~target ~inputs ~shape ~dtype ~binding ~arguments
    ~keyword_arguments

let of_binary bytes =
  let reader = Binary.Reader.create bytes in
  let* magic =
    Binary.Reader.raw_string reader ~length:(String.length binary_magic)
  in
  if magic <> binary_magic then Error "invalid binary FX graph magic"
  else
    let* wire_version = Binary.Reader.u16 reader in
    if wire_version <> 2 then
      Error (Printf.sprintf "unsupported binary FX graph version: %d" wire_version)
    else
      let* manifest_version = Binary.Reader.u16 reader in
      if manifest_version <> 2 then
        Error
          (Printf.sprintf "unsupported FX manifest version: %d" manifest_version)
      else
        let* node_count = Binary.Reader.u32 reader in
        let* output_count = Binary.Reader.u32 reader in
        let* nodes = read_n reader node_count read_binary_node in
        let* outputs = read_n reader output_count Binary.Reader.string in
        let* () = Binary.Reader.finish reader in
        Ok { nodes; outputs }

let of_bytes = of_binary

let of_file path =
  try
    let channel = open_in_bin path in
    let bytes =
      Fun.protect
        ~finally:(fun () -> close_in_noerr channel)
        (fun () -> really_input_string channel (in_channel_length channel))
      |> Bytes.of_string
    in
    of_bytes bytes
  with Sys_error message -> Error ("cannot read FX graph: " ^ message)
