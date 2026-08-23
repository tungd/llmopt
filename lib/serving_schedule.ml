let ( let* ) = Result.bind

module Command = struct
  type t = {
    node_id : int;
    op : Ir.Op.t;
    inputs : Ir.Value.t list;
    output : Ir.Value.t option;
  }

  let node_id command = command.node_id
  let op command = command.op
  let inputs command = command.inputs
  let output command = command.output
end

module Tensor_input = struct
  type t = {
    key : string;
    value : Ir.Value.t;
  }

  let key input = input.key
  let value input = input.value
end

type t = { commands : Command.t list }

let commands schedule = schedule.commands

module Int_set = Set.Make (Int)

let value_id value = Ir.Value.id value |> Ir.Value_id.to_int

let validate_command seen_values command =
  let input_ids = List.map value_id command.Command.inputs in
  match List.find_opt (fun id -> not (Int_set.mem id seen_values)) input_ids with
  | Some id ->
      Error
        (Printf.sprintf "schedule node %d reads undefined value %d"
           command.Command.node_id id)
  | None ->
      let argument_ids =
        match command.Command.op with
        | Ir.Op.Opaque { arguments; keyword_arguments; _ } ->
            arguments @ List.map snd keyword_arguments
            |> List.concat_map Ir.Argument.values |> List.map value_id
            |> List.sort_uniq Int.compare
        | _ -> []
      in
      let declared_ids = List.sort_uniq Int.compare input_ids in
      if argument_ids <> [] && argument_ids <> declared_ids then
        Error
          (Printf.sprintf
             "schedule node %d argument values disagree with its input table"
             command.Command.node_id)
      else
        match command.Command.op, command.Command.inputs, command.Command.output with
        | Ir.Op.Input _, [], Some _ -> Ok ()
        | Ir.Op.Input _, _, _ ->
            Error "schedule input must have no dependencies and one result"
        | Ir.Op.Output _, [ _ ], None -> Ok ()
        | Ir.Op.Output _, _, _ ->
            Error "schedule output must have one dependency and no result"
        | _, _, Some output when Int_set.mem (value_id output) seen_values ->
            Error
              (Printf.sprintf "schedule redefines value %d" (value_id output))
        | _ -> Ok ()

let create commands =
  let rec loop seen_nodes seen_values previous_node_id = function
    | [] -> Ok { commands }
    | command :: rest ->
        if Int_set.mem command.Command.node_id seen_nodes then
          Error
            (Printf.sprintf "schedule repeats node id %d" command.Command.node_id)
        else if command.Command.node_id <= previous_node_id then
          Error "schedule node ids must be strictly increasing"
        else
          let* () = validate_command seen_values command in
          let seen_values =
            match command.Command.output with
            | None -> seen_values
            | Some output -> Int_set.add (value_id output) seen_values
          in
          loop (Int_set.add command.Command.node_id seen_nodes) seen_values
            command.Command.node_id
            rest
  in
  loop Int_set.empty Int_set.empty (-1) commands

let of_graph graph =
  graph |> Ir.Graph.nodes
  |> List.map (fun node ->
         {
           Command.node_id = Ir.node_id node;
           op = Ir.node_op node;
           inputs = Ir.node_inputs node;
           output = Ir.node_output node;
         })
  |> create

let tensor_inputs schedule =
  schedule.commands
  |> List.filter_map (fun command ->
         match command.Command.op, command.Command.output with
         | Ir.Op.Input { source = Ir.Input_source.Tensor_store { key }; _ },
           Some value ->
             Some { Tensor_input.key; value }
         | _ -> None)

let runtime_inputs schedule =
  schedule.commands
  |> List.filter_map (fun command ->
         match command.Command.op, command.Command.output with
         | Ir.Op.Input { name; source = Ir.Input_source.Runtime }, Some value ->
             Some (name, value)
         | _ -> None)

let opaque_count schedule =
  List.fold_left
    (fun count command ->
      match command.Command.op with Ir.Op.Opaque _ -> count + 1 | _ -> count)
    0 schedule.commands

let dtype_tag = function
  | Ir.Dtype.Float32 -> 0
  | Ir.Dtype.Float16 -> 1
  | Ir.Dtype.Bfloat16 -> 2
  | Ir.Dtype.Int64 -> 3
  | Ir.Dtype.Int32 -> 4
  | Ir.Dtype.Int8 -> 5
  | Ir.Dtype.Bool -> 6

let dtype_of_tag = function
  | 0 -> Ok Ir.Dtype.Float32
  | 1 -> Ok Ir.Dtype.Float16
  | 2 -> Ok Ir.Dtype.Bfloat16
  | 3 -> Ok Ir.Dtype.Int64
  | 4 -> Ok Ir.Dtype.Int32
  | 5 -> Ok Ir.Dtype.Int8
  | 6 -> Ok Ir.Dtype.Bool
  | tag -> Error (Printf.sprintf "unknown schedule dtype tag: %d" tag)

let write_shape writer shape =
  let dimensions = Tensor_shape.dimensions shape in
  Binary.Writer.u16 writer (List.length dimensions);
  List.iter (Binary.Writer.u64 writer) dimensions

let read_shape reader =
  let* rank = Binary.Reader.u16 reader in
  let rec dimensions acc remaining =
    if remaining = 0 then
      (match Tensor_shape.create (List.rev acc) with
      | Ok shape -> Ok shape
      | Error error -> Error (Tensor_shape.error_to_string error))
    else
      let* dimension = Binary.Reader.u64 reader in
      dimensions (dimension :: acc) (remaining - 1)
  in
  dimensions [] rank

let write_value writer value =
  Binary.Writer.u32 writer (value_id value);
  Binary.Writer.u8 writer (dtype_tag (Ir.Value.dtype value));
  write_shape writer (Ir.Value.logical_shape value)

let read_value reader =
  let* id = Binary.Reader.u32 reader in
  let* dtype_tag = Binary.Reader.u8 reader in
  let* dtype = dtype_of_tag dtype_tag in
  let* shape = read_shape reader in
  try Ok (Ir.Value.make_tensor ~id ~shape ~dtype)
  with Invalid_argument message -> Error ("invalid schedule value: " ^ message)

let write_memory_space writer = function
  | Ir.Memory_space.Global -> Binary.Writer.u8 writer 0
  | Ir.Memory_space.Shared -> Binary.Writer.u8 writer 1
  | Ir.Memory_space.Register -> Binary.Writer.u8 writer 2
  | Ir.Memory_space.Private -> Binary.Writer.u8 writer 3

let read_memory_space reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 -> Ok Ir.Memory_space.Global
  | 1 -> Ok Ir.Memory_space.Shared
  | 2 -> Ok Ir.Memory_space.Register
  | 3 -> Ok Ir.Memory_space.Private
  | tag -> Error (Printf.sprintf "unknown schedule memory-space tag: %d" tag)

let write_layout writer = function
  | Ir.Layout.Row_major -> Binary.Writer.u8 writer 0
  | Ir.Layout.Col_major -> Binary.Writer.u8 writer 1
  | Ir.Layout.Xor_swizzle mask ->
      Binary.Writer.u8 writer 2;
      Binary.Writer.i64 writer mask

let read_layout reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 -> Ok Ir.Layout.Row_major
  | 1 -> Ok Ir.Layout.Col_major
  | 2 -> Binary.Reader.i64 reader |> Result.map (fun mask -> Ir.Layout.Xor_swizzle mask)
  | _ -> Error (Printf.sprintf "unknown schedule layout tag: %d" tag)

let write_option writer write = function
  | None -> Binary.Writer.u8 writer 0
  | Some value ->
      Binary.Writer.u8 writer 1;
      write writer value

let read_option reader read =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 -> Ok None
  | 1 -> read reader |> Result.map Option.some
  | _ -> Error (Printf.sprintf "unknown schedule option tag: %d" tag)

let rec write_argument writer = function
  | Ir.Argument.Value value ->
      Binary.Writer.u8 writer 0;
      Binary.Writer.u32 writer (value_id value)
  | Ir.Argument.Null -> Binary.Writer.u8 writer 1
  | Ir.Argument.Bool value ->
      Binary.Writer.u8 writer 2;
      Binary.Writer.bool writer value
  | Ir.Argument.Int value ->
      Binary.Writer.u8 writer 3;
      Binary.Writer.i64 writer value
  | Ir.Argument.Float value ->
      Binary.Writer.u8 writer 4;
      Binary.Writer.float64 writer value
  | Ir.Argument.String value ->
      Binary.Writer.u8 writer 5;
      Binary.Writer.string writer value
  | Ir.Argument.Symbol value ->
      Binary.Writer.u8 writer 6;
      Binary.Writer.string writer value
  | Ir.Argument.List values ->
      Binary.Writer.u8 writer 7;
      write_arguments writer values
  | Ir.Argument.Tuple values ->
      Binary.Writer.u8 writer 8;
      write_arguments writer values
  | Ir.Argument.Mapping fields ->
      Binary.Writer.u8 writer 9;
      Binary.Writer.u32 writer (List.length fields);
      List.iter
        (fun (name, value) ->
          Binary.Writer.string writer name;
          write_argument writer value)
        fields
  | Ir.Argument.Slice { start; stop; step } ->
      Binary.Writer.u8 writer 10;
      write_argument writer start;
      write_argument writer stop;
      write_argument writer step

and write_arguments writer values =
  Binary.Writer.u32 writer (List.length values);
  List.iter (write_argument writer) values

let find_value values id =
  match Hashtbl.find_opt values id with
  | Some value -> Ok value
  | None -> Error (Printf.sprintf "schedule argument references undefined value %d" id)

let rec read_argument ~depth values reader =
  if depth > 64 then Error "schedule argument nesting is too deep"
  else
    let* tag = Binary.Reader.u8 reader in
    match tag with
    | 0 ->
        let* id = Binary.Reader.u32 reader in
        find_value values id |> Result.map (fun value -> Ir.Argument.Value value)
    | 1 -> Ok Ir.Argument.Null
    | 2 -> Binary.Reader.bool reader |> Result.map (fun value -> Ir.Argument.Bool value)
    | 3 -> Binary.Reader.i64 reader |> Result.map (fun value -> Ir.Argument.Int value)
    | 4 ->
        let* value = Binary.Reader.float64 reader in
        if Float.is_finite value then Ok (Ir.Argument.Float value)
        else Error "schedule contains a non-finite float argument"
    | 5 -> Binary.Reader.string reader |> Result.map (fun value -> Ir.Argument.String value)
    | 6 -> Binary.Reader.string reader |> Result.map (fun value -> Ir.Argument.Symbol value)
    | 7 ->
        read_arguments ~depth:(depth + 1) values reader
        |> Result.map (fun values -> Ir.Argument.List values)
    | 8 ->
        read_arguments ~depth:(depth + 1) values reader
        |> Result.map (fun values -> Ir.Argument.Tuple values)
    | 9 ->
        let* count = Binary.Reader.u32 reader in
        let rec fields acc remaining =
          if remaining = 0 then Ok (Ir.Argument.Mapping (List.rev acc))
          else
            let* name = Binary.Reader.string reader in
            let* value = read_argument ~depth:(depth + 1) values reader in
            fields ((name, value) :: acc) (remaining - 1)
        in
        fields [] count
    | 10 ->
        let* start = read_argument ~depth:(depth + 1) values reader in
        let* stop = read_argument ~depth:(depth + 1) values reader in
        let* step = read_argument ~depth:(depth + 1) values reader in
        Ok (Ir.Argument.Slice { start; stop; step })
    | _ -> Error (Printf.sprintf "unknown schedule argument tag: %d" tag)

and read_arguments ~depth values reader =
  let* count = Binary.Reader.u32 reader in
  let rec loop acc remaining =
    if remaining = 0 then Ok (List.rev acc)
    else
      let* value = read_argument ~depth values reader in
      loop (value :: acc) (remaining - 1)
  in
  loop [] count

let write_named_arguments writer fields =
  Binary.Writer.u32 writer (List.length fields);
  List.iter
    (fun (name, value) ->
      Binary.Writer.string writer name;
      write_argument writer value)
    fields

let read_named_arguments values reader =
  let* count = Binary.Reader.u32 reader in
  let rec loop acc remaining =
    if remaining = 0 then Ok (List.rev acc)
    else
      let* name = Binary.Reader.string reader in
      let* value = read_argument ~depth:0 values reader in
      loop ((name, value) :: acc) (remaining - 1)
  in
  loop [] count

let write_op writer = function
  | Ir.Op.Input { name; source } ->
      Binary.Writer.u8 writer 0;
      Binary.Writer.string writer name;
      (match source with
      | Ir.Input_source.Runtime -> Binary.Writer.u8 writer 0
      | Ir.Input_source.Tensor_store { key } ->
          Binary.Writer.u8 writer 1;
          Binary.Writer.string writer key)
  | Ir.Op.Alloc { space; layout } ->
      Binary.Writer.u8 writer 1;
      write_memory_space writer space;
      write_layout writer layout
  | Ir.Op.Copy { asynchronous; barrier } ->
      Binary.Writer.u8 writer 2;
      Binary.Writer.bool writer asynchronous;
      write_option writer Binary.Writer.i64 barrier
  | Ir.Op.Matmul { m; n; k } ->
      Binary.Writer.u8 writer 3;
      List.iter (Binary.Writer.u64 writer) [ m; n; k ]
  | Ir.Op.Linear { m; n; k; bias } ->
      Binary.Writer.u8 writer 4;
      List.iter (Binary.Writer.u64 writer) [ m; n; k ];
      Binary.Writer.bool writer bias
  | Ir.Op.Add { broadcast } ->
      Binary.Writer.u8 writer 5;
      Binary.Writer.u8 writer (match broadcast with Shape.Same -> 0 | Shape.Row -> 1)
  | Ir.Op.Gelu -> Binary.Writer.u8 writer 6
  | Ir.Op.Relu -> Binary.Writer.u8 writer 7
  | Ir.Op.Opaque { op; target; arguments; keyword_arguments } ->
      Binary.Writer.u8 writer 8;
      Binary.Writer.string writer op;
      Binary.Writer.string writer target;
      write_arguments writer arguments;
      write_named_arguments writer keyword_arguments
  | Ir.Op.Output { name } ->
      Binary.Writer.u8 writer 9;
      Binary.Writer.string writer name
  | Ir.Op.Barrier_create { id; name } ->
      Binary.Writer.u8 writer 10;
      Binary.Writer.i64 writer id;
      Binary.Writer.string writer name
  | Ir.Op.Barrier_arrive id ->
      Binary.Writer.u8 writer 11;
      Binary.Writer.i64 writer id
  | Ir.Op.Barrier_wait id ->
      Binary.Writer.u8 writer 12;
      Binary.Writer.i64 writer id
  | Ir.Op.Fused_matmul_bias { m; n; k } ->
      Binary.Writer.u8 writer 13;
      List.iter (Binary.Writer.u64 writer) [ m; n; k ]
  | Ir.Op.Q8_linear { m; n; k; bias } ->
      Binary.Writer.u8 writer 14;
      List.iter (Binary.Writer.u64 writer) [ m; n; k ];
      Binary.Writer.bool writer bias

let read_three_dimensions reader =
  let* m = Binary.Reader.u64 reader in
  let* n = Binary.Reader.u64 reader in
  let* k = Binary.Reader.u64 reader in
  Ok (m, n, k)

let read_op values reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 ->
      let* name = Binary.Reader.string reader in
      let* source_tag = Binary.Reader.u8 reader in
      let* source =
        match source_tag with
        | 0 -> Ok Ir.Input_source.Runtime
        | 1 ->
            Binary.Reader.string reader
            |> Result.map (fun key -> Ir.Input_source.Tensor_store { key })
        | _ ->
            Error (Printf.sprintf "unknown schedule input-source tag: %d" source_tag)
      in
      Ok (Ir.Op.Input { name; source })
  | 1 ->
      let* space = read_memory_space reader in
      let* layout = read_layout reader in
      Ok (Ir.Op.Alloc { space; layout })
  | 2 ->
      let* asynchronous = Binary.Reader.bool reader in
      let* barrier = read_option reader Binary.Reader.i64 in
      Ok (Ir.Op.Copy { asynchronous; barrier })
  | 3 ->
      let* m, n, k = read_three_dimensions reader in
      Ok (Ir.Op.Matmul { m; n; k })
  | 4 ->
      let* m, n, k = read_three_dimensions reader in
      let* bias = Binary.Reader.bool reader in
      Ok (Ir.Op.Linear { m; n; k; bias })
  | 5 ->
      let* value = Binary.Reader.u8 reader in
      (match value with
      | 0 -> Ok (Ir.Op.Add { broadcast = Shape.Same })
      | 1 -> Ok (Ir.Op.Add { broadcast = Shape.Row })
      | value -> Error (Printf.sprintf "unknown schedule broadcast tag: %d" value))
  | 6 -> Ok Ir.Op.Gelu
  | 7 -> Ok Ir.Op.Relu
  | 8 ->
      let* op = Binary.Reader.string reader in
      let* target = Binary.Reader.string reader in
      let* arguments = read_arguments ~depth:0 values reader in
      let* keyword_arguments = read_named_arguments values reader in
      Ok (Ir.Op.Opaque { op; target; arguments; keyword_arguments })
  | 9 -> Binary.Reader.string reader |> Result.map (fun name -> Ir.Op.Output { name })
  | 10 ->
      let* id = Binary.Reader.i64 reader in
      let* name = Binary.Reader.string reader in
      Ok (Ir.Op.Barrier_create { id; name })
  | 11 -> Binary.Reader.i64 reader |> Result.map (fun id -> Ir.Op.Barrier_arrive id)
  | 12 -> Binary.Reader.i64 reader |> Result.map (fun id -> Ir.Op.Barrier_wait id)
  | 13 ->
      let* m, n, k = read_three_dimensions reader in
      Ok (Ir.Op.Fused_matmul_bias { m; n; k })
  | 14 ->
      let* m, n, k = read_three_dimensions reader in
      let* bias = Binary.Reader.bool reader in
      Ok (Ir.Op.Q8_linear { m; n; k; bias })
  | _ -> Error (Printf.sprintf "unknown schedule opcode: %d" tag)

let magic = "LLMOSCH\000"

let to_bytes schedule =
  let writer = Binary.Writer.create () in
  Binary.Writer.raw_string writer magic;
  Binary.Writer.u16 writer 1;
  Binary.Writer.u32 writer (List.length schedule.commands);
  List.iter
    (fun command ->
      Binary.Writer.u32 writer command.Command.node_id;
      Binary.Writer.u16 writer (List.length command.Command.inputs);
      List.iter
        (fun value -> Binary.Writer.u32 writer (value_id value))
        command.Command.inputs;
      write_option writer write_value command.Command.output;
      write_op writer command.Command.op)
    schedule.commands;
  Binary.Writer.contents writer

let of_bytes bytes =
  let reader = Binary.Reader.create bytes in
  let* actual_magic = Binary.Reader.raw_string reader ~length:(String.length magic) in
  if actual_magic <> magic then Error "invalid serving schedule magic"
  else
    let* version = Binary.Reader.u16 reader in
    if version <> 1 then
      Error (Printf.sprintf "unsupported serving schedule version: %d" version)
    else
      let* count = Binary.Reader.u32 reader in
      let values = Hashtbl.create count in
      let seen_values = ref Int_set.empty in
      let rec read_values acc remaining =
        if remaining = 0 then Ok (List.rev acc)
        else
          let* node_id = Binary.Reader.u32 reader in
          let* input_count = Binary.Reader.u16 reader in
          let rec inputs acc remaining =
            if remaining = 0 then Ok (List.rev acc)
            else
              let* id = Binary.Reader.u32 reader in
              let* value = find_value values id in
              inputs (value :: acc) (remaining - 1)
          in
          let* inputs = inputs [] input_count in
          let* output = read_option reader read_value in
          let* op = read_op values reader in
          let command = { Command.node_id; op; inputs; output } in
          let* () = validate_command !seen_values command
          in
          Option.iter
            (fun value ->
              let id = value_id value in
              Hashtbl.add values id value;
              seen_values := Int_set.add id !seen_values)
            output;
          read_values (command :: acc) (remaining - 1)
      in
      let* commands = read_values [] count in
      let* () = Binary.Reader.finish reader in
      create commands
