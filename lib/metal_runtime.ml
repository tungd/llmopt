type context_handle
type library_handle
type buffer_handle

let ( let* ) = Result.bind

external create_context_stub : unit -> context_handle
  = "caml_llmopt_metal_create_context"

external device_name_stub : context_handle -> string
  = "caml_llmopt_metal_device_name"

external load_library_stub : context_handle -> string -> library_handle
  = "caml_llmopt_metal_load_library"

external has_function_stub : library_handle -> string -> bool
  = "caml_llmopt_metal_has_function"

external buffer_of_bytes_stub : context_handle -> bytes -> buffer_handle
  = "caml_llmopt_metal_buffer_of_bytes"

external create_buffer_stub : context_handle -> int -> buffer_handle
  = "caml_llmopt_metal_create_buffer"

external map_file_stub : context_handle -> string -> buffer_handle
  = "caml_llmopt_metal_map_file"

external buffer_view_stub : buffer_handle -> int -> int -> buffer_handle
  = "caml_llmopt_metal_buffer_view"

external buffer_contents_stub : buffer_handle -> bytes
  = "caml_llmopt_metal_buffer_contents"

external buffer_length_stub : buffer_handle -> int
  = "caml_llmopt_metal_buffer_length"

external buffer_copy_stub : buffer_handle -> buffer_handle -> unit
  = "caml_llmopt_metal_buffer_copy"

external dispatch_q8_stub :
  library_handle * string * buffer_handle * buffer_handle * buffer_handle
  * buffer_handle * buffer_handle * int * int * int * bool -> unit
  = "caml_llmopt_metal_dispatch_q8"

let protect operation =
  try Ok (operation ()) with
  | Failure message -> Error message
  | Invalid_argument message -> Error message

type t = {
  context : context_handle;
  library : library_handle;
  package : Serving_package.t;
  tensor_store : (Weight_archive.t * buffer_handle) option;
}

type runtime = t

let rec validate_declared_functions library = function
  | [] -> Ok ()
  | entry :: rest ->
      let name = Kernel_abi.Entry.name entry in
      if has_function_stub library name then
        validate_declared_functions library rest
      else Error ("Metal library does not define declared kernel: " ^ name)

let load_package ~root package =
  match Serving_package.validate_files ~root package with
  | Error _ as error -> error
  | Ok () ->
      let archive =
        match Serving_package.tensor_store package with
        | None -> Ok None
        | Some tensor_store ->
            let path =
              tensor_store |> Serving_package.Tensor_store.file
              |> Serving_package.Artifact.path |> Filename.concat root
            in
            Weight_archive.of_file path |> Result.map Option.some
      in
      let* archive = archive in
      let* () = Serving_validation.validate ~package ~archive in
      let* (context, library, tensor_store) =
        protect (fun () ->
            let context = create_context_stub () in
            let library_path =
              Serving_package.files package
              |> Serving_package.Files.metal_library
              |> Serving_package.Artifact.path
              |> Filename.concat root
            in
            let library = load_library_stub context library_path in
            let tensor_store =
              Option.map
                (fun archive ->
                  archive, map_file_stub context (Weight_archive.path archive))
                archive
            in
            context, library, tensor_store)
      in
      let* () =
        validate_declared_functions library (Serving_package.kernels package)
      in
      Ok { context; library; package; tensor_store }

let device_name runtime = device_name_stub runtime.context

module Buffer = struct
  type t = buffer_handle

  let of_bytes ~runtime contents =
    protect (fun () -> buffer_of_bytes_stub runtime.context contents)

  let create ~runtime ~bytes =
    protect (fun () -> create_buffer_stub runtime.context bytes)

  let contents buffer = protect (fun () -> buffer_contents_stub buffer)
  let byte_length = buffer_length_stub

  let copy ~source ~destination =
    protect (fun () -> buffer_copy_stub source destination)
end

let tensor runtime ~name =
  match runtime.tensor_store with
  | None -> Error "serving package has no tensor store"
  | Some (archive, mapped) ->
      (match Weight_archive.find archive name with
      | None -> Error ("tensor store does not contain tensor: " ^ name)
      | Some tensor ->
          protect (fun () ->
              ( buffer_view_stub mapped (Weight_archive.Tensor.offset tensor)
                  (Weight_archive.Tensor.byte_length tensor),
                tensor )))

let q8_kernel runtime dtype =
  let entries = Serving_package.kernels runtime.package in
  List.find_opt
    (fun entry ->
      Kernel_abi.Entry.operation entry = Kernel_abi.Operation.Q8_linear
      && Kernel_abi.Entry.input_dtype entry = dtype
      && Kernel_abi.Entry.output_dtype entry = dtype)
    entries

let dispatch_q8_linear runtime ~dtype ~input ~weight ~scale ~bias ~output ~m ~n
    ~k =
  match dtype with
  | Ir.Dtype.Float16 | Ir.Dtype.Float32 ->
      (match q8_kernel runtime dtype with
      | None ->
          Error
            ("serving package has no Q8 linear kernel for "
            ^ Ir.Dtype.to_string dtype)
      | Some entry ->
          let bias_buffer, has_bias =
            match bias with
            | Some buffer -> buffer, true
            | None -> scale, false
          in
          protect (fun () ->
              dispatch_q8_stub
                ( runtime.library,
                  Kernel_abi.Entry.name entry,
                  input,
                  weight,
                  scale,
                  bias_buffer,
                  output,
                  m,
                  n,
                  k,
                  has_bias );
              Kernel_abi.Entry.name entry))
  | dtype ->
      Error
        ("Q8 Metal dispatch requires f16 or f32 activations, got "
        ^ Ir.Dtype.to_string dtype)

module Value_map = Map.Make (struct
  type t = Ir.Value_id.t
  let compare = Ir.Value_id.compare
end)

module String_map = Map.Make (String)

module Execution = struct
  type t = {
    outputs : (string * Buffer.t) list;
    kernels : string list;
  }

  let output execution ~name = List.assoc_opt name execution.outputs
  let outputs execution = execution.outputs
  let kernels execution = execution.kernels
end

type execution_state = {
  values : Buffer.t Value_map.t;
  outputs_rev : (string * Buffer.t) list;
  kernels_rev : string list;
}

let dtype_byte_width = function
  | Ir.Dtype.Float32 | Ir.Dtype.Int32 -> 4
  | Ir.Dtype.Float16 | Ir.Dtype.Bfloat16 -> 2
  | Ir.Dtype.Int64 -> 8
  | Ir.Dtype.Int8 | Ir.Dtype.Bool -> 1

let value_byte_length value =
  let elements = Tensor_shape.numel (Ir.Value.logical_shape value) in
  let width = dtype_byte_width (Ir.Value.dtype value) in
  if elements > max_int / width then Error "runtime tensor byte length overflows"
  else Ok (elements * width)

let validate_buffer value buffer =
  let* expected = value_byte_length value in
  let actual = Buffer.byte_length buffer in
  if actual = expected then Ok ()
  else
    Error
      (Printf.sprintf "runtime value %d requires %d bytes but received %d"
         (Ir.Value.id value |> Ir.Value_id.to_int) expected actual)

let runtime_input_map inputs =
  List.fold_left
    (fun result (name, buffer) ->
      let* bindings = result in
      if String_map.mem name bindings then
        Error ("runtime input is bound more than once: " ^ name)
      else Ok (String_map.add name buffer bindings))
    (Ok String_map.empty) inputs

let find_value state value =
  match Value_map.find_opt (Ir.Value.id value) state.values with
  | Some buffer -> Ok buffer
  | None ->
      Error
        (Printf.sprintf "runtime value %d is not bound"
           (Ir.Value.id value |> Ir.Value_id.to_int))

let bind_value state value buffer =
  { state with values = Value_map.add (Ir.Value.id value) buffer state.values }

let execute runtime ~inputs =
  let* runtime_inputs = runtime_input_map inputs in
  let schedule = Serving_package.schedule runtime.package in
  let rec run state = function
    | [] ->
        Ok
          {
            Execution.outputs = List.rev state.outputs_rev;
            kernels = List.rev state.kernels_rev;
          }
    | command :: rest ->
        let node_id = Serving_schedule.Command.node_id command in
        let op = Serving_schedule.Command.op command in
        let command_inputs = Serving_schedule.Command.inputs command in
        let command_output = Serving_schedule.Command.output command in
        let continue state = run state rest in
        let unsupported () =
          Error
            (Printf.sprintf "native schedule node %d is not executable: %s"
               node_id (Ir.Op.to_string op))
        in
        (match op, command_inputs, command_output with
        | Ir.Op.Input { name; source = Ir.Input_source.Runtime }, [], Some output ->
            (match String_map.find_opt name runtime_inputs with
            | None -> Error ("runtime input is not bound: " ^ name)
            | Some buffer ->
                let* () = validate_buffer output buffer in
                continue (bind_value state output buffer))
        | ( Ir.Op.Input
              { source = Ir.Input_source.Tensor_store { key }; name = _ },
            [],
            Some output ) ->
            let* buffer, _tensor = tensor runtime ~name:key in
            let* () = validate_buffer output buffer in
            continue (bind_value state output buffer)
        | Ir.Op.Alloc _, [], Some output ->
            let* bytes = value_byte_length output in
            let* buffer = Buffer.create ~runtime ~bytes in
            continue (bind_value state output buffer)
        | Ir.Op.Copy _, [ source; destination ], None ->
            let* source = find_value state source in
            let* destination = find_value state destination in
            let* () = Buffer.copy ~source ~destination in
            continue state
        | ( Ir.Op.Primitive
              (Ir.Primitive.Movement
                (Ir.Movement.View | Ir.Movement.Reshape
                | Ir.Movement.Unsqueeze _)),
            [ input ],
            Some output ) ->
            let* buffer = find_value state input in
            let* () = validate_buffer output buffer in
            continue (bind_value state output buffer)
        | Ir.Op.Primitive (Ir.Primitive.Cast dtype), [ input ], Some output
          when dtype = Ir.Value.dtype input ->
            let* buffer = find_value state input in
            let* () = validate_buffer output buffer in
            continue (bind_value state output buffer)
        | Ir.Op.Q8_linear { m; n; k; bias = has_bias }, values, Some output ->
            let* input, weight, scale, bias =
              match values, has_bias with
              | [ input; weight; scale ], false ->
                  let* input = find_value state input in
                  let* weight = find_value state weight in
                  let* scale = find_value state scale in
                  Ok (input, weight, scale, None)
              | [ input; weight; scale; bias ], true ->
                  let* input = find_value state input in
                  let* weight = find_value state weight in
                  let* scale = find_value state scale in
                  let* bias = find_value state bias in
                  Ok (input, weight, scale, Some bias)
              | _ -> Error "Q8 schedule command has inconsistent bias inputs"
            in
            let* bytes = value_byte_length output in
            let* output_buffer = Buffer.create ~runtime ~bytes in
            let* kernel =
              dispatch_q8_linear runtime ~dtype:(Ir.Value.dtype output) ~input
                ~weight ~scale ~bias ~output:output_buffer ~m ~n ~k
            in
            let state = bind_value state output output_buffer in
            continue { state with kernels_rev = kernel :: state.kernels_rev }
        | Ir.Op.Output { name }, [ input ], None ->
            let* buffer = find_value state input in
            continue
              { state with outputs_rev = (name, buffer) :: state.outputs_rev }
        | _ -> unsupported ())
  in
  run { values = Value_map.empty; outputs_rev = []; kernels_rev = [] }
    (Serving_schedule.commands schedule)
