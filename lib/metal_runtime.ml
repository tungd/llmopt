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

external dispatch_stub :
  library_handle * string * buffer_handle list * bytes * int * int * int * int
  * int * int -> unit
  = "caml_llmopt_metal_dispatch"

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

let kernel_entry ?name runtime ~operation ~input_dtype ~output_dtype =
  Serving_package.kernels runtime.package
  |> List.find_opt (fun entry ->
         Kernel_abi.Entry.operation entry = operation
         && Kernel_abi.Entry.input_dtype entry = input_dtype
         && Kernel_abi.Entry.output_dtype entry = output_dtype
         &&
         match name with
         | None -> true
         | Some expected -> Kernel_abi.Entry.name entry = expected)
  |> function
  | Some entry -> Ok entry
  | None ->
      Error
        (Printf.sprintf "serving package has no %s%s kernel for %s -> %s"
           (Kernel_abi.Operation.to_string operation)
           (match name with None -> "" | Some value -> " " ^ value)
           (Ir.Dtype.to_string input_dtype) (Ir.Dtype.to_string output_dtype))

let dispatch runtime entry ~buffers ~parameters ~grid =
  let name = Kernel_abi.Entry.name entry in
  let group_x, group_y, group_z = Kernel_abi.Entry.threadgroup entry in
  let grid_x, grid_y, grid_z = grid in
  protect (fun () ->
      dispatch_stub
        ( runtime.library,
          name,
          buffers,
          parameters,
          grid_x,
          grid_y,
          grid_z,
          group_x,
          group_y,
          group_z );
      name)

module Parameters = struct
  let max_rank = 8
  let max_u32 = 0xffff_ffffL

  let set_u32 bytes offset value =
    let encoded = Int64.of_int value in
    if value < 0 || Int64.compare encoded max_u32 > 0 then
      Error (Printf.sprintf "Metal parameter does not fit uint32: %d" value)
    else (
      Bytes.set_int32_le bytes offset (Int64.to_int32 encoded);
      Ok ())

  let u32s values =
    let bytes = Bytes.make (4 * List.length values) '\000' in
    let rec write offset = function
      | [] -> Ok bytes
      | value :: rest ->
          let* () = set_u32 bytes offset value in
          write (offset + 4) rest
    in
    write 0 values

  let rms_norm ~rows ~width ~epsilon =
    let bytes = Bytes.make 12 '\000' in
    let* () = set_u32 bytes 0 rows in
    let* () = set_u32 bytes 4 width in
    Bytes.set_int32_le bytes 8 (Int32.bits_of_float epsilon);
    Ok bytes

  let arange ~count ~start ~step =
    let bytes = Bytes.make 24 '\000' in
    let* () = set_u32 bytes 0 count in
    Bytes.set_int64_le bytes 8 (Int64.of_int start);
    Bytes.set_int64_le bytes 16 (Int64.of_int step);
    Ok bytes

  let fill ~count scalar =
    let bytes = Bytes.make 8 '\000' in
    let* () = set_u32 bytes 0 count in
    (match scalar with
    | Ir.Scalar.Bool value ->
        Bytes.set_int32_le bytes 4 (if value then 1l else 0l)
    | Ir.Scalar.Float value ->
        Bytes.set_int32_le bytes 4 (Int32.bits_of_float value)
    | Ir.Scalar.Int value -> Bytes.set_int32_le bytes 4 (Int32.of_int value));
    Ok bytes

  let attention ~batches ~heads ~query_length ~key_length ~head_dimension
      ~mask_batches ~mask_heads ~causal ~scale =
    let bytes = Bytes.make 36 '\000' in
    let values =
      [ batches; heads; query_length; key_length; head_dimension; mask_batches;
        mask_heads; if causal then 1 else 0 ]
    in
    let rec write offset = function
      | [] -> Ok ()
      | value :: rest ->
          let* () = set_u32 bytes offset value in
          write (offset + 4) rest
    in
    let* () = write 0 values in
    Bytes.set_int32_le bytes 32 (Int32.bits_of_float scale);
    Ok bytes

  let scalar_i64 = function
    | Ir.Scalar.Bool value -> if value then 1L else 0L
    | Ir.Scalar.Int value -> Int64.of_int value
    | Ir.Scalar.Float value -> Int64.of_float value

  let scalar_f32 scalar =
    scalar |> Ir.Scalar.to_float |> Int32.bits_of_float

  let rec write_shape bytes offset = function
    | [] -> Ok ()
    | dimension :: rest ->
        let* () = set_u32 bytes offset dimension in
        write_shape bytes (offset + 4) rest

  let dimensions ~operation shape =
    let dimensions = Tensor_shape.dimensions shape in
    if List.length dimensions > max_rank then
      Error
        (Printf.sprintf "Metal %s supports rank at most %d" operation max_rank)
    else Ok dimensions

  let pad_shape dimensions =
    dimensions @ List.init (max_rank - List.length dimensions) (Fun.const 1)

  let write_i64s bytes offset values =
    List.iteri
      (fun index value ->
        Bytes.set_int64_le bytes (offset + (8 * index)) (Int64.of_int value))
      values

  let movement ~input_dimensions ~output_dimensions ~axis0 ~axis1 =
    let rank = List.length output_dimensions in
    let bytes = Bytes.make 80 '\000' in
    let* () = set_u32 bytes 0 (List.fold_left ( * ) 1 output_dimensions) in
    let* () = set_u32 bytes 4 rank in
    let* () = set_u32 bytes 8 axis0 in
    let* () = set_u32 bytes 12 axis1 in
    let* () = write_shape bytes 16 (pad_shape input_dimensions) in
    let* () = write_shape bytes 48 (pad_shape output_dimensions) in
    Ok bytes

  let transpose ~input_shape ~output_shape ~axis0 ~axis1 =
    let* input_dimensions = dimensions ~operation:"transpose" input_shape in
    let* output_dimensions = dimensions ~operation:"transpose" output_shape in
    let* axis0 =
      Tensor_shape.normalize_axis input_shape axis0
      |> Result.map_error Tensor_shape.error_to_string
    in
    let* axis1 =
      Tensor_shape.normalize_axis input_shape axis1
      |> Result.map_error Tensor_shape.error_to_string
    in
    let* inferred =
      Tensor_shape.transpose input_shape ~axis0 ~axis1
      |> Result.map_error Tensor_shape.error_to_string
    in
    if not (Tensor_shape.equal inferred output_shape) then
      Error "Metal transpose output shape is inconsistent"
    else
      movement ~input_dimensions ~output_dimensions ~axis0 ~axis1

  let expand ~input_shape ~output_shape =
    let* input_dimensions = dimensions ~operation:"expand" input_shape in
    let* output_dimensions = dimensions ~operation:"expand" output_shape in
    let* _ =
      Tensor_shape.expand input_shape ~target:output_shape
      |> Result.map_error Tensor_shape.error_to_string
    in
    let rank = List.length output_dimensions in
    let aligned_input =
      List.init (rank - List.length input_dimensions) (Fun.const 1)
      @ input_dimensions
    in
    movement ~input_dimensions:aligned_input ~output_dimensions ~axis0:0 ~axis1:0

  let index ~input_shape ~output_shape index =
    let* input_dimensions = dimensions ~operation:"index" input_shape in
    let* output_dimensions = dimensions ~operation:"index" output_shape in
    let selectors = Tensor_shape.Index.selectors index in
    if List.length selectors > max_rank then
      Error
        (Printf.sprintf "Metal index supports at most %d selectors" max_rank)
    else
      let selector_kind, starts, steps =
        List.fold_left
          (fun (kinds, starts, steps) selector ->
            match selector with
            | Tensor_shape.Index.At value ->
                0 :: kinds, value :: starts, 0 :: steps
            | Tensor_shape.Index.Slice { start; step; length = _ } ->
                1 :: kinds, start :: starts, step :: steps
            | Tensor_shape.Index.New_axis ->
                2 :: kinds, 0 :: starts, 0 :: steps)
          ([], [], []) selectors
      in
      let selector_kind = List.rev selector_kind in
      let starts = List.rev starts in
      let steps = List.rev steps in
      let pad values =
        values @ List.init (max_rank - List.length values) (Fun.const 0)
      in
      let bytes = Bytes.make 240 '\000' in
      let* () = set_u32 bytes 0 (Tensor_shape.numel output_shape) in
      let* () = set_u32 bytes 4 (List.length input_dimensions) in
      let* () = set_u32 bytes 8 (List.length output_dimensions) in
      let* () = set_u32 bytes 12 (List.length selectors) in
      let* () = write_shape bytes 16 (pad_shape input_dimensions) in
      let* () = write_shape bytes 48 (pad_shape output_dimensions) in
      let* () = write_shape bytes 80 (pad selector_kind) in
      write_i64s bytes 112 (pad starts);
      write_i64s bytes 176 (pad steps);
      Ok bytes

  let concat ~left_shape ~right_shape ~output_shape ~axis =
    let* left_dimensions = dimensions ~operation:"concat" left_shape in
    let* _right_dimensions = dimensions ~operation:"concat" right_shape in
    let* output_dimensions = dimensions ~operation:"concat" output_shape in
    let* axis =
      Tensor_shape.normalize_axis output_shape axis
      |> Result.map_error Tensor_shape.error_to_string
    in
    let* inferred =
      Tensor_shape.concat [ left_shape; right_shape ] ~axis
      |> Result.map_error Tensor_shape.error_to_string
    in
    if not (Tensor_shape.equal inferred output_shape) then
      Error "Metal concat output shape is inconsistent"
    else
      let left_dimensions = Array.of_list left_dimensions in
      let bytes = Bytes.make 48 '\000' in
      let* () = set_u32 bytes 0 (Tensor_shape.numel output_shape) in
      let* () = set_u32 bytes 4 (List.length output_dimensions) in
      let* () = set_u32 bytes 8 axis in
      let* () = set_u32 bytes 12 left_dimensions.(axis) in
      let* () = write_shape bytes 16 (pad_shape output_dimensions) in
      Ok bytes

  let roll ~shape ~axis ~shift =
    let* shape_dimensions = dimensions ~operation:"roll" shape in
    let* axis =
      Tensor_shape.normalize_axis shape axis
      |> Result.map_error Tensor_shape.error_to_string
    in
    let encoded_shift = Int64.of_int shift in
    if
      Int64.compare encoded_shift (Int64.of_int32 Int32.min_int) < 0
      || Int64.compare encoded_shift (Int64.of_int32 Int32.max_int) > 0
    then Error "Metal roll shift does not fit int32"
    else
      let bytes = Bytes.make 48 '\000' in
      let* () = set_u32 bytes 0 (Tensor_shape.numel shape) in
      let* () = set_u32 bytes 4 (List.length shape_dimensions) in
      let* () = set_u32 bytes 8 axis in
      Bytes.set_int32_le bytes 12 (Int64.to_int32 encoded_shift);
      let* () = write_shape bytes 16 (pad_shape shape_dimensions) in
      Ok bytes

  let align_broadcast_shape ~output_dimensions = function
    | None -> Ok (List.init (List.length output_dimensions) (Fun.const 1))
    | Some input_dimensions ->
        let output_rank = List.length output_dimensions in
        let input_rank = List.length input_dimensions in
        if input_rank > output_rank then
          Error "pointwise input rank exceeds output rank"
        else
          let aligned =
            List.init (output_rank - input_rank) (Fun.const 1)
            @ input_dimensions
          in
          if
            List.for_all2
              (fun input output -> input = 1 || input = output)
              aligned output_dimensions
          then Ok aligned
          else Error "pointwise input shape is not broadcastable to output"

  let pointwise ~output_shape ~left_shape ~right_shape ~left_scalar
      ~right_scalar =
    let output_dimensions = Tensor_shape.dimensions output_shape in
    let rank = List.length output_dimensions in
    if rank > 8 then Error "Metal pointwise kernels support rank at most eight"
    else
      let* left_dimensions =
        align_broadcast_shape ~output_dimensions left_shape
      in
      let* right_dimensions =
        align_broadcast_shape ~output_dimensions right_shape
      in
      let pad dimensions = dimensions @ List.init (8 - rank) (Fun.const 1) in
      let bytes = Bytes.make 136 '\000' in
      let count = Tensor_shape.numel output_shape in
      let* () = set_u32 bytes 0 count in
      let* () = set_u32 bytes 4 rank in
      let* () = set_u32 bytes 8 (if Option.is_some left_scalar then 1 else 0) in
      let* () = set_u32 bytes 12 (if Option.is_some right_scalar then 1 else 0) in
      let* () = write_shape bytes 16 (pad output_dimensions) in
      let* () = write_shape bytes 48 (pad left_dimensions) in
      let* () = write_shape bytes 80 (pad right_dimensions) in
      Option.iter
        (fun scalar ->
          Bytes.set_int64_le bytes 112 (scalar_i64 scalar);
          Bytes.set_int32_le bytes 128 (scalar_f32 scalar))
        left_scalar;
      Option.iter
        (fun scalar ->
          Bytes.set_int64_le bytes 120 (scalar_i64 scalar);
          Bytes.set_int32_le bytes 132 (scalar_f32 scalar))
        right_scalar;
      Ok bytes
end

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

let find_values state values =
  let rec collect acc = function
    | [] -> Ok (List.rev acc)
    | value :: rest ->
        let* buffer = find_value state value in
        collect (buffer :: acc) rest
  in
  collect [] values

let bind_value state value buffer =
  { state with values = Value_map.add (Ir.Value.id value) buffer state.values }

let round_up value multiple =
  if value > max_int - (multiple - 1) then
    Error "Metal grid dimension overflows"
  else Ok (((value + multiple - 1) / multiple) * multiple)

let split_axis shape axis =
  let rec loop outer remaining = function
    | [] -> Error "Metal axis is outside the tensor rank"
    | width :: rest when remaining = 0 ->
        Ok (List.fold_left ( * ) 1 outer, width, List.fold_left ( * ) 1 rest)
    | dimension :: rest -> loop (dimension :: outer) (remaining - 1) rest
  in
  loop [] axis (Tensor_shape.dimensions shape)

let dispatch_output ?name runtime state output ~operation ~input_dtype ~buffers
    ~parameters ~grid =
  let output_dtype = Ir.Value.dtype output in
  let* entry = kernel_entry ?name runtime ~operation ~input_dtype ~output_dtype in
  let* bytes = value_byte_length output in
  let* output_buffer = Buffer.create ~runtime ~bytes in
  let* kernel =
    dispatch runtime entry ~buffers:(buffers @ [ output_buffer ]) ~parameters
      ~grid
  in
  Ok (bind_value state output output_buffer, kernel)

let pointwise_kernel operation output =
  let output_dtype = Ir.Value.dtype output in
  let tensor_dtypes =
    operation |> Ir.Pointwise.values |> List.map Ir.Value.dtype
  in
  let* input_dtype =
    match tensor_dtypes with
    | [] -> Error "Metal pointwise operation requires at least one tensor operand"
    | dtype :: rest when List.for_all (( = ) dtype) rest -> Ok dtype
    | _ -> Error "Metal pointwise operands must use one tensor dtype"
  in
  let name =
    match operation, input_dtype, output_dtype with
    | Ir.Pointwise.Binary (Ir.Pointwise.Add, _, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_add_f16"
    | Ir.Pointwise.Binary (Ir.Pointwise.Add, _, _), Ir.Dtype.Int64,
      Ir.Dtype.Int64 ->
        Some "llmopt_add_i64"
    | Ir.Pointwise.Binary (Ir.Pointwise.Mul, _, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_mul_f16"
    | Ir.Pointwise.Binary (Ir.Pointwise.Mul, _, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_mul_f32"
    | Ir.Pointwise.Binary (Ir.Pointwise.Less_equal, _, _), Ir.Dtype.Int64,
      Ir.Dtype.Bool ->
        Some "llmopt_le_i64"
    | Ir.Pointwise.Unary (Ir.Pointwise.Neg, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_neg_f16"
    | Ir.Pointwise.Unary (Ir.Pointwise.Silu, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_silu_f16"
    | Ir.Pointwise.Unary (Ir.Pointwise.Cos, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_cos_f32"
    | Ir.Pointwise.Unary (Ir.Pointwise.Sin, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_sin_f32"
    | _ -> None
  in
  match name with
  | Some name -> Ok (name, input_dtype)
  | None ->
      Error
        (Printf.sprintf "unsupported Metal pointwise kernel: %s %s -> %s"
           (Ir.Pointwise.to_string operation)
           (Ir.Dtype.to_string input_dtype) (Ir.Dtype.to_string output_dtype))

let pointwise_operand_shape = function
  | Ir.Pointwise.Tensor value ->
      Some (Ir.Value.logical_shape value |> Tensor_shape.dimensions)
  | Ir.Pointwise.Scalar _ -> None

let pointwise_operand_scalar = function
  | Ir.Pointwise.Tensor _ -> None
  | Ir.Pointwise.Scalar scalar -> Some scalar

let movement_kernel movement input output =
  let input_dtype = Ir.Value.dtype input in
  let output_dtype = Ir.Value.dtype output in
  if input_dtype <> output_dtype then
    Error "Metal movement input and output dtypes differ"
  else
    let name =
      match movement, input_dtype with
      | Ir.Movement.Transpose _, Ir.Dtype.Float16 -> Some "llmopt_transpose_f16"
      | Ir.Movement.Transpose _, Ir.Dtype.Float32 -> Some "llmopt_transpose_f32"
      | Ir.Movement.Index _, Ir.Dtype.Float16 -> Some "llmopt_index_f16"
      | Ir.Movement.Index _, Ir.Dtype.Float32 -> Some "llmopt_index_f32"
      | Ir.Movement.Index _, Ir.Dtype.Int64 -> Some "llmopt_index_i64"
      | Ir.Movement.Expand, Ir.Dtype.Float16 -> Some "llmopt_expand_f16"
      | Ir.Movement.Expand, Ir.Dtype.Float32 -> Some "llmopt_expand_f32"
      | Ir.Movement.Expand, Ir.Dtype.Bool -> Some "llmopt_expand_bool"
      | Ir.Movement.Concat _, Ir.Dtype.Float16 -> Some "llmopt_concat_f16"
      | Ir.Movement.Concat _, Ir.Dtype.Float32 -> Some "llmopt_concat_f32"
      | Ir.Movement.Roll _, Ir.Dtype.Float16 -> Some "llmopt_roll_f16"
      | _ -> None
    in
    match name with
    | Some name -> Ok (name, input_dtype)
    | None ->
        Error
          (Printf.sprintf "unsupported Metal movement kernel: %s %s"
             (Ir.Movement.to_string movement)
             (Ir.Dtype.to_string input_dtype))

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
        let dispatched result =
          let* state, kernel = result in
          continue { state with kernels_rev = kernel :: state.kernels_rev }
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
                | Ir.Movement.Unsqueeze _ | Ir.Movement.Contiguous)),
            [ input ],
            Some output ) ->
            let* buffer = find_value state input in
            let* () = validate_buffer output buffer in
            continue (bind_value state output buffer)
        | ( Ir.Op.Primitive
              (Ir.Primitive.Movement
                (Ir.Movement.Transpose { axis0; axis1 } as movement)),
            [ input ],
            Some output ) ->
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* name, input_dtype = movement_kernel movement input output in
            let* input_buffer = find_value state input in
            let* parameters =
              Parameters.transpose ~input_shape:(Ir.Value.logical_shape input)
                ~output_shape:(Ir.Value.logical_shape output) ~axis0 ~axis1
            in
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Movement ~input_dtype
                 ~buffers:[ input_buffer ] ~parameters ~grid:(count, 1, 1))
        | ( Ir.Op.Primitive
              (Ir.Primitive.Movement
                (Ir.Movement.Index index as movement)),
            [ input ],
            Some output ) ->
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* name, input_dtype = movement_kernel movement input output in
            let* input_buffer = find_value state input in
            let* parameters =
              Parameters.index ~input_shape:(Ir.Value.logical_shape input)
                ~output_shape:(Ir.Value.logical_shape output) index
            in
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Movement ~input_dtype
                 ~buffers:[ input_buffer ] ~parameters ~grid:(count, 1, 1))
        | ( Ir.Op.Primitive
              (Ir.Primitive.Movement (Ir.Movement.Expand as movement)),
            [ input ],
            Some output ) ->
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* name, input_dtype = movement_kernel movement input output in
            let* input_buffer = find_value state input in
            let* parameters =
              Parameters.expand ~input_shape:(Ir.Value.logical_shape input)
                ~output_shape:(Ir.Value.logical_shape output)
            in
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Movement ~input_dtype
                 ~buffers:[ input_buffer ] ~parameters ~grid:(count, 1, 1))
        | ( Ir.Op.Primitive
              (Ir.Primitive.Movement
                (Ir.Movement.Concat { axis } as movement)),
            [ left; right ],
            Some output ) ->
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* name, input_dtype = movement_kernel movement left output in
            let* buffers = find_values state [ left; right ] in
            let* parameters =
              Parameters.concat ~left_shape:(Ir.Value.logical_shape left)
                ~right_shape:(Ir.Value.logical_shape right)
                ~output_shape:(Ir.Value.logical_shape output) ~axis
            in
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Movement ~input_dtype ~buffers
                 ~parameters ~grid:(count, 1, 1))
        | ( Ir.Op.Primitive
              (Ir.Primitive.Movement
                (Ir.Movement.Roll { axis; shift } as movement)),
            [ input ],
            Some output ) ->
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* name, input_dtype = movement_kernel movement input output in
            let* input_buffer = find_value state input in
            let* parameters =
              Parameters.roll ~shape:(Ir.Value.logical_shape input) ~axis ~shift
            in
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Movement ~input_dtype
                 ~buffers:[ input_buffer ] ~parameters ~grid:(count, 1, 1))
        | Ir.Op.Primitive (Ir.Primitive.Cast dtype), [ input ], Some output
          when dtype = Ir.Value.dtype input ->
            let* buffer = find_value state input in
            let* () = validate_buffer output buffer in
            continue (bind_value state output buffer)
        | Ir.Op.Primitive (Ir.Primitive.Cast _), [ input ], Some output ->
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* buffers = find_values state [ input ] in
            let* parameters = Parameters.u32s [ count ] in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Cast
                 ~input_dtype:(Ir.Value.dtype input) ~buffers ~parameters
                 ~grid:(count, 1, 1))
        | ( Ir.Op.Primitive (Ir.Primitive.Pointwise operation),
            _,
            Some output ) ->
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* name, input_dtype = pointwise_kernel operation output in
            (match operation with
            | Ir.Pointwise.Unary (_, input) ->
                let* input_buffer = find_value state input in
                let* parameters = Parameters.u32s [ count ] in
                dispatched
                  (dispatch_output ~name runtime state output
                     ~operation:Kernel_abi.Operation.Pointwise ~input_dtype
                     ~buffers:[ input_buffer ] ~parameters ~grid:(count, 1, 1))
            | Ir.Pointwise.Binary (_, left, right) ->
                let tensor_values = Ir.Pointwise.values operation in
                let* dummy =
                  match tensor_values with
                  | value :: _ -> find_value state value
                  | [] -> Error "Metal pointwise operation has no tensor buffer"
                in
                let operand_buffer = function
                  | Ir.Pointwise.Tensor value -> find_value state value
                  | Ir.Pointwise.Scalar _ -> Ok dummy
                in
                let* left_buffer = operand_buffer left in
                let* right_buffer = operand_buffer right in
                let* parameters =
                  Parameters.pointwise
                    ~output_shape:(Ir.Value.logical_shape output)
                    ~left_shape:(pointwise_operand_shape left)
                    ~right_shape:(pointwise_operand_shape right)
                    ~left_scalar:(pointwise_operand_scalar left)
                    ~right_scalar:(pointwise_operand_scalar right)
                in
                dispatched
                  (dispatch_output ~name runtime state output
                     ~operation:Kernel_abi.Operation.Pointwise ~input_dtype
                     ~buffers:[ left_buffer; right_buffer ] ~parameters
                     ~grid:(count, 1, 1)))
        | Ir.Op.Matmul { m; n; k = _ }, [ lhs; rhs ], Some output ->
            let* buffers = find_values state [ lhs; rhs ] in
            let* grid_x = round_up n 16 in
            let* grid_y = round_up m 16 in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Matmul
                 ~input_dtype:(Ir.Value.dtype lhs) ~buffers
                 ~parameters:(Bytes.create 0) ~grid:(grid_x, grid_y, 1))
        | Ir.Op.Fused_matmul_bias { m; n; k = _ }, [ lhs; rhs; bias ], Some output ->
            let* buffers = find_values state [ lhs; rhs; bias ] in
            let* grid_x = round_up n 16 in
            let* grid_y = round_up m 16 in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Fused_linear
                 ~input_dtype:(Ir.Value.dtype lhs) ~buffers
                 ~parameters:(Bytes.create 0) ~grid:(grid_x, grid_y, 1))
        | Ir.Op.Linear { m; n; k = _; bias }, values, Some output ->
            let* values =
              match values, bias with
              | [ input; weight ], false -> Ok (input, [ input; weight ])
              | [ input; weight; bias_value ], true ->
                  Ok (input, [ input; weight; bias_value ])
              | _ -> Error "linear schedule command has inconsistent bias inputs"
            in
            let input, values = values in
            let* buffers = find_values state values in
            let* grid_x = round_up n 16 in
            let* grid_y = round_up m 16 in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Linear
                 ~input_dtype:(Ir.Value.dtype input) ~buffers
                 ~parameters:(Bytes.create 0) ~grid:(grid_x, grid_y, 1))
        | Ir.Op.Rms_norm { epsilon }, [ input; weight ], Some output ->
            let input_shape = Ir.Value.logical_shape input in
            let dimensions = Tensor_shape.dimensions input_shape in
            let* width =
              match List.rev dimensions with
              | width :: _ when width > 0 -> Ok width
              | _ -> Error "RMSNorm requires a non-empty final dimension"
            in
            let rows = Tensor_shape.numel input_shape / width in
            let* buffers = find_values state [ input; weight ] in
            let* parameters = Parameters.rms_norm ~rows ~width ~epsilon in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Rms_norm
                 ~input_dtype:(Ir.Value.dtype input) ~buffers ~parameters
                 ~grid:(rows, 1, 1))
        | ( Ir.Op.Primitive Ir.Primitive.Embedding,
            [ indices; weight ],
            Some output ) ->
            let* vocabulary, width =
              match Tensor_shape.dimensions (Ir.Value.logical_shape weight) with
              | [ vocabulary; width ] -> Ok (vocabulary, width)
              | _ -> Error "embedding weight must have rank two"
            in
            let tokens = Tensor_shape.numel (Ir.Value.logical_shape indices) in
            let* buffers = find_values state [ indices; weight ] in
            let* parameters = Parameters.u32s [ tokens; vocabulary; width ] in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Embedding
                 ~input_dtype:(Ir.Value.dtype indices) ~buffers ~parameters
                 ~grid:(tokens * width, 1, 1))
        | Ir.Op.Primitive (Ir.Primitive.Arange config), [], Some output ->
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* parameters =
              Parameters.arange ~count ~start:(Ir.Arange.start config)
                ~step:(Ir.Arange.step config)
            in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Arange
                 ~input_dtype:(Ir.Value.dtype output) ~buffers:[] ~parameters
                 ~grid:(count, 1, 1))
        | Ir.Op.Primitive (Ir.Primitive.Fill scalar), [], Some output ->
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* parameters = Parameters.fill ~count scalar in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Fill
                 ~input_dtype:(Ir.Value.dtype output) ~buffers:[] ~parameters
                 ~grid:(count, 1, 1))
        | ( Ir.Op.Primitive (Ir.Primitive.Short_conv config),
            [ input; weight ],
            Some output ) ->
            let* batches, channels, input_width =
              match Tensor_shape.dimensions (Ir.Value.logical_shape input) with
              | [ batches; channels; input_width ] ->
                  Ok (batches, channels, input_width)
              | _ -> Error "ShortConv input must have rank three"
            in
            let* kernel_width =
              match Tensor_shape.dimensions (Ir.Value.logical_shape weight) with
              | [ _channels; 1; kernel_width ] -> Ok kernel_width
              | _ -> Error "ShortConv weight must have shape [channels,1,width]"
            in
            let* output_width =
              match Tensor_shape.dimensions (Ir.Value.logical_shape output) with
              | [ _batches; _channels; output_width ] -> Ok output_width
              | _ -> Error "ShortConv output must have rank three"
            in
            let* buffers = find_values state [ input; weight ] in
            let* parameters =
              Parameters.u32s
                [ batches; channels; input_width; output_width; kernel_width;
                  Ir.Short_conv.stride config; Ir.Short_conv.padding config;
                  Ir.Short_conv.dilation config ]
            in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Short_conv
                 ~input_dtype:(Ir.Value.dtype input) ~buffers ~parameters
                 ~grid:(Tensor_shape.numel (Ir.Value.logical_shape output), 1, 1))
        | ( Ir.Op.Primitive (Ir.Primitive.Attention config),
            [ query; key; value; mask ],
            Some output ) ->
            let* batches, heads, query_length, head_dimension =
              match Tensor_shape.dimensions (Ir.Value.logical_shape query) with
              | [ batches; heads; query_length; head_dimension ] ->
                  Ok (batches, heads, query_length, head_dimension)
              | _ -> Error "attention query must have rank four"
            in
            let* key_length =
              match Tensor_shape.dimensions (Ir.Value.logical_shape key) with
              | [ _batches; _heads; key_length; _dimension ] -> Ok key_length
              | _ -> Error "attention key must have rank four"
            in
            let* mask_batches, mask_heads =
              match Tensor_shape.dimensions (Ir.Value.logical_shape mask) with
              | [ mask_batches; mask_heads; _query_length; _key_length ] ->
                  Ok (mask_batches, mask_heads)
              | _ -> Error "attention mask must have rank four"
            in
            let* buffers = find_values state [ query; key; value; mask ] in
            let* parameters =
              Parameters.attention ~batches ~heads ~query_length ~key_length
                ~head_dimension ~mask_batches ~mask_heads
                ~causal:(Ir.Attention.causal config)
                ~scale:(Ir.Attention.scale config)
            in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Attention
                 ~input_dtype:(Ir.Value.dtype query) ~buffers ~parameters
                 ~grid:(batches * heads * query_length, 1, 1))
        | ( Ir.Op.Primitive (Ir.Primitive.Diff config),
            [ source; prepend ],
            Some output ) ->
            let* outer, source_width, inner =
              split_axis (Ir.Value.logical_shape source) (Ir.Diff.axis config)
            in
            let* _prepend_outer, prepend_width, _prepend_inner =
              split_axis (Ir.Value.logical_shape prepend) (Ir.Diff.axis config)
            in
            let* buffers = find_values state [ source; prepend ] in
            let* parameters =
              Parameters.u32s [ outer; source_width; prepend_width; inner ]
            in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Diff
                 ~input_dtype:(Ir.Value.dtype source) ~buffers ~parameters
                 ~grid:(Tensor_shape.numel (Ir.Value.logical_shape output), 1, 1))
        | Ir.Op.Primitive (Ir.Primitive.Cumsum config), [ input ], Some output ->
            let* outer, width, inner =
              split_axis (Ir.Value.logical_shape input) (Ir.Cumsum.axis config)
            in
            let* buffers = find_values state [ input ] in
            let* parameters = Parameters.u32s [ outer; width; inner ] in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Cumsum
                 ~input_dtype:(Ir.Value.dtype input) ~buffers ~parameters
                 ~grid:(outer * inner, 1, 1))
        | ( Ir.Op.Primitive Ir.Primitive.Gather2,
            [ source; first_index; second_index ],
            Some output ) ->
            let* rows, cols =
              match Tensor_shape.dimensions (Ir.Value.logical_shape source) with
              | [ rows; cols ] -> Ok (rows, cols)
              | _ -> Error "Gather2 source must have rank two"
            in
            let* output_shape, first_shape, second_shape =
              match
                ( Tensor_shape.dimensions (Ir.Value.logical_shape output),
                  Tensor_shape.dimensions (Ir.Value.logical_shape first_index),
                  Tensor_shape.dimensions (Ir.Value.logical_shape second_index) )
              with
              | ([ o0; o1; o2; o3 ], [ f0; f1; f2; f3 ], [ s0; s1; s2; s3 ]) ->
                  Ok
                    ( [ o0; o1; o2; o3 ], [ f0; f1; f2; f3 ],
                      [ s0; s1; s2; s3 ] )
              | _ -> Error "Gather2 indices and output must have rank four"
            in
            let* buffers = find_values state [ source; first_index; second_index ] in
            let* parameters =
              Parameters.u32s
                ([ Tensor_shape.numel (Ir.Value.logical_shape output); rows; cols ]
                @ output_shape @ first_shape @ second_shape)
            in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Gather2
                 ~input_dtype:(Ir.Value.dtype source) ~buffers ~parameters
                 ~grid:(Tensor_shape.numel (Ir.Value.logical_shape output), 1, 1))
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
