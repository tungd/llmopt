type context_handle
type library_handle
type buffer_handle
type batch_handle

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

external begin_batch_stub : library_handle -> batch_handle
  = "caml_llmopt_metal_begin_batch"

external batch_dispatch_stub :
  batch_handle * library_handle * string * buffer_handle list * bytes * int
  * int * int * int * int * int -> unit
  = "caml_llmopt_metal_batch_dispatch"

external batch_copy_stub :
  batch_handle -> buffer_handle -> buffer_handle -> unit
  = "caml_llmopt_metal_batch_copy"

external batch_barrier_stub : batch_handle -> unit
  = "caml_llmopt_metal_batch_barrier"

external commit_batch_stub : batch_handle -> unit
  = "caml_llmopt_metal_commit_batch"

external abort_batch_stub : batch_handle -> unit
  = "caml_llmopt_metal_abort_batch"

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

module Batch = struct
  type t = {
    handle : batch_handle;
    context : context_handle;
  }

  let create (runtime : runtime) =
    protect (fun () ->
        {
          handle = begin_batch_stub runtime.library;
          context = runtime.context;
        })

  let dispatch batch ~(runtime : runtime) ~name ~buffers ~parameters ~grid
      ~group =
    let grid_x, grid_y, grid_z = grid in
    let group_x, group_y, group_z = group in
    if batch.context != runtime.context then
      Error "Metal batch and dispatch runtime use different device contexts"
    else
      protect (fun () ->
          batch_dispatch_stub
            ( batch.handle,
              runtime.library,
              name,
              buffers,
              parameters,
              grid_x,
              grid_y,
              grid_z,
              group_x,
              group_y,
              group_z ))

  let copy batch ~source ~destination =
    protect (fun () -> batch_copy_stub batch.handle source destination)

  let barrier batch =
    protect (fun () -> batch_barrier_stub batch.handle)

  let commit batch = protect (fun () -> commit_batch_stub batch.handle)
  let abort batch = protect (fun () -> abort_batch_stub batch.handle)
end

type prepared_package = {
  root : string;
  manifest : Serving_package.t;
  archive : Weight_archive.t option;
}

let rec validate_declared_functions library = function
  | [] -> Ok ()
  | entry :: rest ->
      let name = Kernel_abi.Entry.name entry in
      if has_function_stub library name then
        validate_declared_functions library rest
      else Error ("Metal library does not define declared kernel: " ^ name)

let prepare_package (root, package) =
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
      Ok { root; manifest = package; archive }

let file_identity path =
  try
    let stats = Unix.stat path in
    Ok (stats.st_dev, stats.st_ino)
  with Unix.Unix_error (error, operation, target) ->
    Error
      (Printf.sprintf "%s %s: %s" operation target (Unix.error_message error))

let load_packages sources =
  let rec prepare acc = function
    | [] -> Ok (List.rev acc)
    | source :: rest ->
        let* package = prepare_package source in
        prepare (package :: acc) rest
  in
  let* prepared = prepare [] sources in
  if prepared = [] then Error "cannot load an empty serving-package set"
  else
    let* context = protect (fun () -> create_context_stub ()) in
    let rec load mappings runtimes = function
      | [] -> Ok (List.rev runtimes)
      | prepared :: rest ->
          let library_path =
            Serving_package.files prepared.manifest
            |> Serving_package.Files.metal_library
            |> Serving_package.Artifact.path
            |> Filename.concat prepared.root
          in
          let* library = protect (fun () -> load_library_stub context library_path) in
          let* tensor_store, mappings =
            match prepared.archive with
            | None -> Ok (None, mappings)
            | Some archive ->
                let path = Weight_archive.path archive in
                let* identity = file_identity path in
                (match List.assoc_opt identity mappings with
                | Some mapped -> Ok (Some (archive, mapped), mappings)
                | None ->
                    let* mapped =
                      protect (fun () -> map_file_stub context path)
                    in
                    Ok (Some (archive, mapped), (identity, mapped) :: mappings))
          in
          let* () =
            validate_declared_functions library
              (Serving_package.kernels prepared.manifest)
          in
          let runtime =
            { context; library; package = prepared.manifest; tensor_store }
          in
          load mappings (runtime :: runtimes) rest
    in
    load [] [] prepared

let load_package ~root package =
  let* runtimes = load_packages [ root, package ] in
  match runtimes with
  | [ runtime ] -> Ok runtime
  | _ -> Error "single serving-package load returned an invalid runtime count"

let device_name runtime = device_name_stub runtime.context
let package runtime = runtime.package

module Buffer = struct
  type t = buffer_handle

  let of_bytes ~runtime contents =
    protect (fun () -> buffer_of_bytes_stub runtime.context contents)

  let create ~runtime ~bytes =
    protect (fun () -> create_buffer_stub runtime.context bytes)

  let view ~parent ~offset ~bytes =
    protect (fun () -> buffer_view_stub parent offset bytes)

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

let q8_kernel ?name runtime ~operation dtype =
  let entries = Serving_package.kernels runtime.package in
  List.find_opt
    (fun entry ->
      Kernel_abi.Entry.operation entry = operation
      && Kernel_abi.Entry.input_dtype entry = dtype
      && Kernel_abi.Entry.output_dtype entry = dtype
      &&
      match name with
      | None -> true
      | Some expected -> Kernel_abi.Entry.name entry = expected)
    entries

type q8_epilogue = Identity | Silu | Add | Mul_add

module Q8_decode_layout = struct
  type t = Scalar | Simd_single | Simd_pair | Simdgroup_gemm
end

let q8_operation = function
  | Identity -> Kernel_abi.Operation.Q8_linear
  | Silu -> Kernel_abi.Operation.Q8_linear_silu
  | Add -> Kernel_abi.Operation.Q8_linear_add
  | Mul_add -> Kernel_abi.Operation.Q8_linear_mul_add

let q8_kernel_name dtype ~m ~epilogue =
  match dtype, m, epilogue with
  | Ir.Dtype.Float16, 1, Identity -> Ok "llmopt_q8_gemv_simd"
  | Ir.Dtype.Float32, 1, Identity -> Ok "llmopt_q8_gemv_simd_f32"
  | Ir.Dtype.Float16, _, Identity -> Ok "llmopt_q8_linear"
  | Ir.Dtype.Float32, _, Identity -> Ok "llmopt_q8_linear_f32"
  | Ir.Dtype.Float16, 1, Silu -> Ok "llmopt_q8_gemv_silu_simd"
  | Ir.Dtype.Float32, 1, Silu -> Ok "llmopt_q8_gemv_silu_simd_f32"
  | Ir.Dtype.Float16, _, Silu -> Ok "llmopt_q8_linear_silu"
  | Ir.Dtype.Float32, _, Silu -> Ok "llmopt_q8_linear_silu_f32"
  | Ir.Dtype.Float16, 1, Add -> Ok "llmopt_q8_gemv_add_simd"
  | Ir.Dtype.Float32, 1, Add -> Ok "llmopt_q8_gemv_add_simd_f32"
  | Ir.Dtype.Float16, _, Add -> Ok "llmopt_q8_linear_add"
  | Ir.Dtype.Float32, _, Add -> Ok "llmopt_q8_linear_add_f32"
  | Ir.Dtype.Float16, 1, Mul_add -> Ok "llmopt_q8_gemv_mul_add_simd"
  | Ir.Dtype.Float32, 1, Mul_add -> Ok "llmopt_q8_gemv_mul_add_simd_f32"
  | Ir.Dtype.Float16, _, Mul_add -> Ok "llmopt_q8_linear_mul_add"
  | Ir.Dtype.Float32, _, Mul_add -> Ok "llmopt_q8_linear_mul_add_f32"
  | dtype, _, _ ->
      Error
        ("Q8 Metal dispatch requires f16 or f32 activations, got "
        ^ Ir.Dtype.to_string dtype)

let q8_pair_kernel_name dtype = function
  | epilogue ->
      let* suffix =
        match dtype with
        | Ir.Dtype.Float16 -> Ok ""
        | Ir.Dtype.Float32 -> Ok "_f32"
        | dtype ->
            Error
              ("paired Q8 Metal dispatch requires f16 or f32 activations, got "
              ^ Ir.Dtype.to_string dtype)
      in
      let base =
        match epilogue with
        | Identity -> "llmopt_q8_gemv_pair_simd"
        | Silu -> "llmopt_q8_gemv_silu_pair_simd"
        | Add -> "llmopt_q8_gemv_add_pair_simd"
        | Mul_add -> "llmopt_q8_gemv_mul_add_pair_simd"
      in
      Ok (base ^ suffix)

let q8_legacy_gemv_name dtype = function
  | Identity ->
      (match dtype with
      | Ir.Dtype.Float16 -> Some "llmopt_q8_gemv"
      | Ir.Dtype.Float32 -> Some "llmopt_q8_gemv_f32"
      | _ -> None)
  | Silu ->
      (match dtype with
      | Ir.Dtype.Float16 -> Some "llmopt_q8_gemv_silu"
      | Ir.Dtype.Float32 -> Some "llmopt_q8_gemv_silu_f32"
      | _ -> None)
  | Add ->
      (match dtype with
      | Ir.Dtype.Float16 -> Some "llmopt_q8_gemv_add"
      | Ir.Dtype.Float32 -> Some "llmopt_q8_gemv_add_f32"
      | _ -> None)
  | Mul_add ->
      (match dtype with
      | Ir.Dtype.Float16 -> Some "llmopt_q8_gemv_mul_add"
      | Ir.Dtype.Float32 -> Some "llmopt_q8_gemv_mul_add_f32"
      | _ -> None)

let dispatch_q8_linear runtime ~dtype ~input ~weight ~scale ~bias ~output ~m ~n
    ~k =
  match dtype with
  | Ir.Dtype.Float16 | Ir.Dtype.Float32 ->
      (match
         q8_kernel runtime ~operation:Kernel_abi.Operation.Q8_linear dtype
       with
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

let dispatch_q8_gemm runtime ~dtype ~input ~weight ~scale ~bias ~output ~m ~n ~k =
  dispatch_q8_linear runtime ~dtype ~input ~weight ~scale ~bias ~output ~m ~n ~k

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

let dispatch ?batch runtime entry ~buffers ~parameters ~grid =
  let name = Kernel_abi.Entry.name entry in
  let group_x, group_y, group_z = Kernel_abi.Entry.threadgroup entry in
  let grid_x, grid_y, grid_z = grid in
  let* () =
    match batch with
    | None ->
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
                group_z ))
    | Some batch ->
        Batch.dispatch batch ~runtime ~name ~buffers ~parameters
          ~grid:(grid_x, grid_y, grid_z) ~group:(group_x, group_y, group_z)
  in
  Ok name

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

  let q8_linear_add_norm ~m ~n ~k ~epsilon =
    let bytes = Bytes.make 16 '\000' in
    let* encoded = u32s [ m; n; k ] in
    Bytes.blit encoded 0 bytes 0 12;
    Bytes.set_int32_le bytes 12 (Int32.bits_of_float epsilon);
    Ok bytes

  let q8_dual_linear ~m ~n1 ~n2 ~k = u32s [ m; n1; n2; k ]

  let q8_qkv_linear ~m ~n_q ~n_kv ~k = u32s [ m; n_q; n_kv; k ]

  let q8_lm_head_argmax ~m ~n ~k ~epsilon =
    let bytes = Bytes.make 16 '\000' in
    let* encoded = u32s [ m; n; k ] in
    Bytes.blit encoded 0 bytes 0 12;
    Bytes.set_int32_le bytes 12 (Int32.bits_of_float epsilon);
    Ok bytes

  let q8_fused_swiglu_ffn ~m ~n ~k ~epsilon =
    let bytes = Bytes.make 16 '\000' in
    let* encoded = u32s [ m; n; k ] in
    Bytes.blit encoded 0 bytes 0 12;
    Bytes.set_int32_le bytes 12 (Int32.bits_of_float epsilon);
    Ok bytes

  let q8_fused_short_conv ~m ~channels ~window ~k ~epsilon =
    let bytes = Bytes.make 20 '\000' in
    let* encoded = u32s [ m; channels; window; k ] in
    Bytes.blit encoded 0 bytes 0 16;
    Bytes.set_int32_le bytes 16 (Int32.bits_of_float epsilon);
    Ok bytes

  let rms_rope ~batches ~tokens ~heads ~width ~half_dimension ~trig_batches
      ~epsilon =
    let bytes = Bytes.make 28 '\000' in
    let* () =
      u32s [ batches; tokens; heads; width; half_dimension; trig_batches ]
      |> Result.map (fun encoded -> Bytes.blit encoded 0 bytes 0 24)
    in
    Bytes.set_int32_le bytes 24 (Int32.bits_of_float epsilon);
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

  let paged_attention_q8 ~batches ~query_heads ~kv_heads ~past_length
      ~head_dimension ~mask_batches ~mask_heads ~cache_layer ~attention_layers
      ~group_size ~token_stride ~scale =
    let bytes = Bytes.make 48 '\000' in
    let values =
      [ batches; query_heads; kv_heads; past_length; head_dimension; mask_batches;
        mask_heads; cache_layer; attention_layers; group_size; token_stride ]
    in
    let rec write offset = function
      | [] -> Ok ()
      | value :: rest ->
          let* () = set_u32 bytes offset value in
          write (offset + 4) rest
    in
    let* () = write 0 values in
    Bytes.set_int32_le bytes 44 (Int32.bits_of_float scale);
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

  let index_parameters ~count ~input_shape ~output_shape index =
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
      let* () = set_u32 bytes 0 count in
      let* () = set_u32 bytes 4 (List.length input_dimensions) in
      let* () = set_u32 bytes 8 (List.length output_dimensions) in
      let* () = set_u32 bytes 12 (List.length selectors) in
      let* () = write_shape bytes 16 (pad_shape input_dimensions) in
      let* () = write_shape bytes 48 (pad_shape output_dimensions) in
      let* () = write_shape bytes 80 (pad selector_kind) in
      write_i64s bytes 112 (pad starts);
      write_i64s bytes 176 (pad steps);
      Ok bytes

  let index ~input_shape ~output_shape index =
    index_parameters ~count:(Tensor_shape.numel output_shape) ~input_shape
      ~output_shape index

  let update_slice ~destination_shape ~source_shape index =
    let* inferred =
      Tensor_shape.apply_index destination_shape index
      |> Result.map_error Tensor_shape.error_to_string
    in
    if not (Tensor_shape.equal inferred source_shape) then
      Error "Metal slice-update source shape is inconsistent"
    else
      index_parameters ~count:(Tensor_shape.numel destination_shape)
        ~input_shape:destination_shape ~output_shape:source_shape index

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

module Cache = struct
  module Attention = struct
    type t = Key | Value
  end

  module Pack_layout = struct
    type t = Scalar | Simdgroup
  end

  module Unpack_layout = struct
    type t = Scalar | Vec4
  end

  type pack_kernel = {
    entry : Kernel_abi.Entry.t;
    dispatch_layout : Pack_layout.t;
  }

  type unpack_kernel = {
    entry : Kernel_abi.Entry.t;
    unpack_layout : Unpack_layout.t;
  }

  type kernels = {
    pack_attention : pack_kernel;
    unpack_attention : unpack_kernel;
    pack_checkpoint : pack_kernel;
    unpack_checkpoint : unpack_kernel;
  }

  type t = {
    runtime : runtime;
    config : Kv_cache.Config.t;
    layout : Kv_cache.Layout.t;
    token_pool : Buffer.t;
    checkpoint_pool : Buffer.t;
    kernels : kernels;
  }

  type batch = {
    cache : t;
    commands : Batch.t;
    mutable resources : Buffer.t list;
    mutable dispatches : int;
  }

  let release_batch_resources batch = batch.resources <- []

  let with_batch cache operation =
    let* commands = Batch.create cache.runtime in
    let batch = { cache; commands; resources = []; dispatches = 0 } in
    match operation batch with
    | Error message ->
        ignore (Batch.abort commands);
        release_batch_resources batch;
        Error message
    | Ok value ->
        let completion =
          if batch.dispatches = 0 then Batch.abort commands
          else Batch.commit commands
        in
        release_batch_resources batch;
        let* () = completion in
        Ok value
    | exception exception_value ->
        ignore (Batch.abort commands);
        release_batch_resources batch;
        raise exception_value

  let format cache = Kv_cache.Layout.format cache.layout
  let token_pool_bytes cache = Kv_cache.Config.token_pool_bytes cache.config

  let checkpoint_pool_bytes cache =
    Kv_cache.Config.checkpoint_pool_bytes cache.config

  let kernel_names format =
    let suffix =
      match format with
      | Kv_cache.Format.F16 -> "f16"
      | Kv_cache.Format.Q8 _ -> "q8"
    in
    ( "llmopt_cache_pack_attention_" ^ suffix,
      "llmopt_cache_unpack_attention_" ^ suffix,
      "llmopt_cache_pack_checkpoint_" ^ suffix,
      "llmopt_cache_unpack_checkpoint_" ^ suffix )

  let kernel_dtypes format =
    match format with
    | Kv_cache.Format.F16 ->
        ( (Ir.Dtype.Float16, Ir.Dtype.Float16),
          (Ir.Dtype.Float16, Ir.Dtype.Float16) )
    | Kv_cache.Format.Q8 _ ->
        ( (Ir.Dtype.Float16, Ir.Dtype.Int8),
          (Ir.Dtype.Int8, Ir.Dtype.Float16) )

  let cache_entry runtime name (input_dtype, output_dtype) =
    kernel_entry ~name runtime ~operation:Kernel_abi.Operation.Cache
      ~input_dtype ~output_dtype

  let cache_entry_opt runtime name (input_dtype, output_dtype) =
    Serving_package.kernels runtime.package
    |> List.find_opt (fun entry ->
           Kernel_abi.Entry.name entry = name
           && Kernel_abi.Entry.operation entry = Kernel_abi.Operation.Cache
           && Kernel_abi.Entry.input_dtype entry = input_dtype
           && Kernel_abi.Entry.output_dtype entry = output_dtype)

  let select_pack_kernel runtime format scalar_name dtypes =
    match format with
    | Kv_cache.Format.F16 ->
        let* entry = cache_entry runtime scalar_name dtypes in
        Ok { entry; dispatch_layout = Pack_layout.Scalar }
    | Kv_cache.Format.Q8 _ ->
        let simd_name = scalar_name ^ "_simd" in
        (match cache_entry_opt runtime simd_name dtypes with
        | Some entry -> Ok { entry; dispatch_layout = Pack_layout.Simdgroup }
        | None ->
            let* entry = cache_entry runtime scalar_name dtypes in
            Ok { entry; dispatch_layout = Pack_layout.Scalar })

  let select_unpack_kernel runtime format scalar_name dtypes =
    match format with
    | Kv_cache.Format.Q8 { group_size } when group_size mod 4 = 0 ->
        let vector_name = scalar_name ^ "_vec4" in
        (match cache_entry_opt runtime vector_name dtypes with
        | Some entry -> Ok { entry; unpack_layout = Unpack_layout.Vec4 }
        | None ->
            let* entry = cache_entry runtime scalar_name dtypes in
            Ok { entry; unpack_layout = Unpack_layout.Scalar })
    | Kv_cache.Format.Q8 _ | Kv_cache.Format.F16 ->
        let* entry = cache_entry runtime scalar_name dtypes in
        Ok { entry; unpack_layout = Unpack_layout.Scalar }

  let create ~runtime ~config =
    if Serving_package.stage runtime.package <> Serving_package.Stage.Serving then
      Error "physical Metal cache requires a serving-stage package"
    else
      let layout = Kv_cache.Config.layout config in
      let format = Kv_cache.Layout.format layout in
      let supported =
        runtime.package |> Serving_package.cache
        |> Serving_package.Cache.supported_kv
      in
      if not (List.mem format supported) then
        Error
          ("serving package does not support physical cache format: "
          ^ Kv_cache.Format.to_string format)
      else
        let pack_attention, unpack_attention, pack_checkpoint,
            unpack_checkpoint =
          kernel_names format
        in
        let pack_dtype, unpack_dtype = kernel_dtypes format in
        let* pack_attention =
          select_pack_kernel runtime format pack_attention pack_dtype
        in
        let* unpack_attention =
          select_unpack_kernel runtime format unpack_attention unpack_dtype
        in
        let* pack_checkpoint =
          select_pack_kernel runtime format pack_checkpoint pack_dtype
        in
        let* unpack_checkpoint =
          select_unpack_kernel runtime format unpack_checkpoint unpack_dtype
        in
        let* token_pool =
          Buffer.create ~runtime ~bytes:(Kv_cache.Config.token_pool_bytes config)
        in
        let* checkpoint_pool =
          Buffer.create ~runtime
            ~bytes:(Kv_cache.Config.checkpoint_pool_bytes config)
        in
        Ok
          {
            runtime;
            config;
            layout;
            token_pool;
            checkpoint_pool;
            kernels =
              {
                pack_attention;
                unpack_attention;
                pack_checkpoint;
                unpack_checkpoint;
              };
          }

  let checked_product left right label =
    if left < 0 || right < 0 || (right <> 0 && left > max_int / right) then
      Error ("physical cache " ^ label ^ " overflows")
    else Ok (left * right)

  let pack_grid kernel logical_groups =
    match kernel.dispatch_layout with
    | Pack_layout.Scalar -> Ok logical_groups
    | Pack_layout.Simdgroup ->
        let simdgroups_per_threadgroup = 8 in
        if logical_groups > max_int - (simdgroups_per_threadgroup - 1) then
          Error "physical cache SIMD pack grid overflows"
        else
          let threadgroups =
            (logical_groups + simdgroups_per_threadgroup - 1)
            / simdgroups_per_threadgroup
          in
          checked_product threadgroups 256 "SIMD pack grid"

  let unpack_grid kernel elements =
    match kernel.unpack_layout with
    | Unpack_layout.Scalar -> elements
    | Unpack_layout.Vec4 -> elements / 4

  let exact_buffer_length buffer expected label =
    let actual = Buffer.byte_length buffer in
    if actual = expected then Ok ()
    else
      Error
        (Printf.sprintf "physical cache %s requires %d bytes but received %d"
           label expected actual)

  let validate_layer layer count label =
    if layer < 0 || layer >= count then
      Error
        (Printf.sprintf "physical cache %s layer %d is outside [0,%d)" label
           layer count)
    else Ok ()

  let group_size format =
    Kv_cache.Format.group_size format |> Option.value ~default:1

  let slots_buffer cache slots =
    let count = Array.length slots in
    if count = 0 then Error "physical attention cache requires at least one slot"
    else
      let capacity = Kv_cache.Config.token_capacity cache.config in
      let seen = Hashtbl.create count in
      let rec validate index values =
        if index = count then Parameters.u32s (List.rev values)
        else
          let slot = Kv_cache.Slot.to_int slots.(index) in
          if slot < 0 || slot >= capacity then
            Error
              (Printf.sprintf "physical cache token slot %d is outside [0,%d)"
                 slot capacity)
          else if Hashtbl.mem seen slot then
            Error (Printf.sprintf "physical cache token slot %d is duplicated" slot)
          else (
            Hashtbl.add seen slot ();
            validate (index + 1) (slot :: values))
      in
      let* bytes = validate 0 [] in
      Buffer.of_bytes ~runtime:cache.runtime bytes

  let attention_parameters cache ~layer ~kind ~items ~source_items
      ~source_offset =
    let layout = cache.layout in
    let heads = Kv_cache.Layout.kv_heads layout in
    let head_dim = Kv_cache.Layout.head_dim layout in
    let group_size = group_size (format cache) in
    let token_elements =
      2 * Kv_cache.Layout.attention_layers layout * heads * head_dim
    in
    let token_groups =
      Kv_cache.Format.groups_for_elements (format cache)
        ~elements:token_elements
    in
    let segment =
      (2 * layer)
      + match kind with Attention.Key -> 0 | Attention.Value -> 1
    in
    Parameters.u32s
      [ items; segment; heads; head_dim; group_size; token_elements; token_groups;
        Kv_cache.Layout.bytes_per_token layout; source_items; source_offset ]

  let attention_counts cache items =
    let layout = cache.layout in
    let* segment_elements =
      checked_product (Kv_cache.Layout.kv_heads layout)
        (Kv_cache.Layout.head_dim layout) "attention segment size"
    in
    let* elements =
      checked_product items segment_elements "attention transfer size"
    in
    let groups =
      match format cache with
      | Kv_cache.Format.F16 -> elements
      | Kv_cache.Format.Q8 { group_size } ->
          elements / group_size
    in
    Ok (elements, groups)

  let attention ?batch cache ~pack ~layer ~kind ~slots ~source_items
      ~source_offset buffer =
    let* () =
      validate_layer layer (Kv_cache.Layout.attention_layers cache.layout)
        "attention"
    in
    let items = Array.length slots in
    let* elements, groups = attention_counts cache items in
    let* expected_elements =
      if pack then
        if source_items <= 0 then
          Error "physical attention cache source_items must be positive"
        else if source_offset < 0 || source_offset > source_items - items then
          Error "physical attention cache source slice is out of range"
        else
          let* source_elements =
            checked_product source_items
              (Kv_cache.Layout.kv_heads cache.layout
              * Kv_cache.Layout.head_dim cache.layout)
              "attention source size"
          in
          Ok source_elements
      else Ok elements
    in
    let* expected_bytes =
      checked_product expected_elements 2 "attention byte length"
    in
    let* () =
      exact_buffer_length buffer expected_bytes
        (if pack then "attention source" else "attention destination")
    in
    let* slots_buffer = slots_buffer cache slots in
    let* parameters =
      attention_parameters cache ~layer ~kind ~items ~source_items
        ~source_offset
    in
    let entry =
      if pack then cache.kernels.pack_attention.entry
      else cache.kernels.unpack_attention.entry
    in
    let buffers =
      if pack then [ buffer; slots_buffer; cache.token_pool ]
      else [ cache.token_pool; slots_buffer; buffer ]
    in
    let* grid =
      if pack then pack_grid cache.kernels.pack_attention groups
      else Ok (unpack_grid cache.kernels.unpack_attention elements)
    in
    Option.iter
      (fun batch -> batch.resources <- slots_buffer :: batch.resources)
      batch;
    let* kernel =
      dispatch ?batch:(Option.map (fun batch -> batch.commands) batch)
        cache.runtime entry ~buffers ~parameters ~grid:(grid, 1, 1)
    in
    Option.iter
      (fun batch -> batch.dispatches <- batch.dispatches + 1)
      batch;
    Ok kernel

  let pack_attention cache ~layer ~kind ~slots ~source =
    attention cache ~pack:true ~layer ~kind ~slots
      ~source_items:(Array.length slots) ~source_offset:0 source

  let pack_attention_slice cache ~layer ~kind ~slots ~source_items
      ~source_offset ~source =
    if Serving_package.abi_version cache.runtime.package < 8 then
      Error "attention cache source slicing requires serving-package ABI v8"
    else
      attention cache ~pack:true ~layer ~kind ~slots ~source_items ~source_offset
        source

  let unpack_attention cache ~layer ~kind ~slots ~destination =
    attention cache ~pack:false ~layer ~kind ~slots
      ~source_items:(Array.length slots) ~source_offset:0 destination

  let batch_pack_attention batch ~layer ~kind ~slots ~source =
    attention ~batch batch.cache ~pack:true ~layer ~kind ~slots
      ~source_items:(Array.length slots) ~source_offset:0 source

  let batch_pack_attention_slice batch ~layer ~kind ~slots ~source_items
      ~source_offset ~source =
    if Serving_package.abi_version batch.cache.runtime.package < 8 then
      Error "attention cache source slicing requires serving-package ABI v8"
    else
      attention ~batch batch.cache ~pack:true ~layer ~kind ~slots ~source_items
        ~source_offset source

  let batch_unpack_attention batch ~layer ~kind ~slots ~destination =
    attention ~batch batch.cache ~pack:false ~layer ~kind ~slots
      ~source_items:(Array.length slots) ~source_offset:0 destination

  let checkpoint_parameters cache ~layer ~checkpoint =
    let layout = cache.layout in
    let layer_elements =
      Kv_cache.Layout.recurrent_width layout
      * Kv_cache.Layout.recurrent_window layout
    in
    let checkpoint_elements =
      Kv_cache.Layout.recurrent_layers layout * layer_elements
    in
    let checkpoint_groups =
      Kv_cache.Format.groups_for_elements (format cache)
        ~elements:checkpoint_elements
    in
    Parameters.u32s
      [ Kv_cache.Checkpoint.to_int checkpoint; layer; layer_elements;
        group_size (format cache); checkpoint_elements; checkpoint_groups;
        Kv_cache.Layout.bytes_per_checkpoint layout ]

  let transfer_checkpoint ?batch cache ~pack ~layer ~checkpoint buffer =
    let* () =
      validate_layer layer (Kv_cache.Layout.recurrent_layers cache.layout)
        "checkpoint"
    in
    let checkpoint_id = Kv_cache.Checkpoint.to_int checkpoint in
    let capacity = Kv_cache.Config.checkpoint_capacity cache.config in
    if checkpoint_id < 0 || checkpoint_id >= capacity then
      Error
        (Printf.sprintf "physical cache checkpoint %d is outside [0,%d)"
           checkpoint_id capacity)
    else
      let* layer_elements =
        checked_product (Kv_cache.Layout.recurrent_width cache.layout)
          (Kv_cache.Layout.recurrent_window cache.layout)
          "checkpoint layer size"
      in
      let* expected_bytes =
        checked_product layer_elements 2 "checkpoint byte length"
      in
      let* () =
        exact_buffer_length buffer expected_bytes
          (if pack then "checkpoint source" else "checkpoint destination")
      in
      let* parameters = checkpoint_parameters cache ~layer ~checkpoint in
      let entry =
        if pack then cache.kernels.pack_checkpoint.entry
        else cache.kernels.unpack_checkpoint.entry
      in
      let buffers =
        if pack then [ buffer; cache.checkpoint_pool ]
        else [ cache.checkpoint_pool; buffer ]
      in
      let* grid =
        match pack, format cache with
        | true, Kv_cache.Format.Q8 { group_size } ->
            pack_grid cache.kernels.pack_checkpoint
              (layer_elements / group_size)
        | true, Kv_cache.Format.F16 ->
            pack_grid cache.kernels.pack_checkpoint layer_elements
        | false, _ ->
            Ok (unpack_grid cache.kernels.unpack_checkpoint layer_elements)
      in
      let* kernel =
        dispatch ?batch:(Option.map (fun batch -> batch.commands) batch)
          cache.runtime entry ~buffers ~parameters ~grid:(grid, 1, 1)
      in
      Option.iter
        (fun batch -> batch.dispatches <- batch.dispatches + 1)
        batch;
      Ok kernel

  let pack_checkpoint cache ~layer ~checkpoint ~source =
    transfer_checkpoint cache ~pack:true ~layer ~checkpoint source

  let unpack_checkpoint cache ~layer ~checkpoint ~destination =
    transfer_checkpoint cache ~pack:false ~layer ~checkpoint destination

  let batch_pack_checkpoint batch ~layer ~checkpoint ~source =
    transfer_checkpoint ~batch batch.cache ~pack:true ~layer ~checkpoint source

  let batch_unpack_checkpoint batch ~layer ~checkpoint ~destination =
    transfer_checkpoint ~batch batch.cache ~pack:false ~layer ~checkpoint
      destination

  let q8_attention_inputs cache ~slots =
    match format cache with
    | Kv_cache.Format.F16 ->
        Error "direct paged attention requires a grouped-Q8 physical cache"
    | Kv_cache.Format.Q8 _ ->
        let* slots = slots_buffer cache slots in
        Ok
          [ Serving_schedule.Lfm25.q8_attention_pool_input, cache.token_pool;
            Serving_schedule.Lfm25.q8_attention_slots_input, slots ]
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
    workspace_bytes : int;
  }

  let output execution ~name = List.assoc_opt name execution.outputs
  let outputs execution = execution.outputs
  let kernels execution = execution.kernels
  let workspace_bytes execution = execution.workspace_bytes
end

module Execution_batch = struct
  type t = {
    runtime : runtime;
    commands : Batch.t;
    mutable resources : Buffer.t list;
    mutable schedules : int;
  }
end

let with_execution_batch runtime operation =
  let* commands = Batch.create runtime in
  let batch =
    { Execution_batch.runtime; commands; resources = []; schedules = 0 }
  in
  let release_resources () = batch.resources <- [] in
  match operation batch with
  | Error message ->
      ignore (Batch.abort commands);
      release_resources ();
      Error message
  | Ok value ->
      let completion =
        if batch.schedules = 0 then Batch.abort commands
        else Batch.commit commands
      in
      release_resources ();
      let* () = completion in
      Ok value
  | exception exception_value ->
      ignore (Batch.abort commands);
      release_resources ();
      raise exception_value

let encode_cache execution_batch cache operation =
  let batch =
    {
      Cache.cache;
      commands = execution_batch.Execution_batch.commands;
      resources = [];
      dispatches = 0;
    }
  in
  let retain_resources () =
    execution_batch.resources <- batch.resources @ execution_batch.resources
  in
  match operation batch with
  | Error _ as error ->
      retain_resources ();
      error
  | Ok value ->
      retain_resources ();
      execution_batch.schedules <-
        execution_batch.schedules + batch.dispatches;
      Ok value

let encode_cache_pack_attention_slice execution_batch ~cache ~layer ~kind
    ~slots ~source_items ~source_offset ~source =
  encode_cache execution_batch cache (fun batch ->
      Cache.batch_pack_attention_slice batch ~layer ~kind ~slots ~source_items
        ~source_offset ~source)

let encode_cache_pack_checkpoint execution_batch ~cache ~layer ~checkpoint
    ~source =
  encode_cache execution_batch cache (fun batch ->
      Cache.batch_pack_checkpoint batch ~layer ~checkpoint ~source)

type execution_state = {
  values : Buffer.t Value_map.t;
  outputs_rev : (string * Buffer.t) list;
  kernels_rev : string list;
  memory_plan : Serving_memory_plan.t;
  workspace : Buffer.t option;
}

let value_byte_length = Serving_memory_plan.value_bytes

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

let workspace_buffer state value =
  match
    Serving_memory_plan.allocation state.memory_plan value,
    state.workspace
  with
  | Some allocation, Some workspace ->
      Buffer.view ~parent:workspace
        ~offset:(Serving_memory_plan.Allocation.offset allocation)
        ~bytes:(Serving_memory_plan.Allocation.bytes allocation)
  | Some _, None -> Error "workspace allocation exists without a Metal buffer"
  | None, _ ->
      Error
        (Printf.sprintf "runtime value %d has no workspace allocation"
           (Ir.Value.id value |> Ir.Value_id.to_int))

let round_up value multiple =
  if value > max_int - (multiple - 1) then
    Error "Metal grid dimension overflows"
  else Ok (((value + multiple - 1) / multiple) * multiple)

let q8_simd_grid ~outputs_per_simdgroup columns =
  let simdgroups_per_threadgroup = 8 in
  let simd_width = 32 in
  let* rounded_columns = round_up columns outputs_per_simdgroup in
  let simdgroups = rounded_columns / outputs_per_simdgroup in
  let* rounded_simdgroups = round_up simdgroups simdgroups_per_threadgroup in
  if rounded_simdgroups > max_int / simd_width then
    Error "Metal Q8 decode grid dimension overflows"
  else Ok (rounded_simdgroups * simd_width)

let q8_decode_grid columns = function
  | Q8_decode_layout.Scalar -> round_up columns 256
  | Q8_decode_layout.Simd_single ->
      q8_simd_grid ~outputs_per_simdgroup:1 columns
  | Q8_decode_layout.Simd_pair ->
      q8_simd_grid ~outputs_per_simdgroup:2 columns
  | Q8_decode_layout.Simdgroup_gemm -> round_up columns 8

let linear_f16_grid columns =
  let simdgroups_per_threadgroup = 8 in
  let simd_width = 32 in
  let* rounded_columns = round_up columns simdgroups_per_threadgroup in
  if rounded_columns > max_int / simd_width then
    Error "Metal float16 linear grid dimension overflows"
  else Ok (rounded_columns * simd_width)

let simd_rows_grid rows = linear_f16_grid rows

let rms_norm_kernel_name = function
  | Ir.Dtype.Float32 -> Ok "llmopt_rms_norm_f32_f16_simd"
  | Ir.Dtype.Float16 -> Ok "llmopt_rms_norm_f16_simd"
  | dtype ->
      Error
        ("unsupported Metal RMSNorm input dtype: " ^ Ir.Dtype.to_string dtype)

let legacy_rms_norm_kernel_name = function
  | Ir.Dtype.Float32 -> Some "llmopt_rms_norm_f32_f16"
  | Ir.Dtype.Float16 -> Some "llmopt_rms_norm_f16"
  | _ -> None

let select_rms_norm_kernel runtime input_dtype output_dtype =
  let* preferred = rms_norm_kernel_name input_dtype in
  match
    kernel_entry ~name:preferred runtime
      ~operation:Kernel_abi.Operation.Rms_norm ~input_dtype ~output_dtype
  with
  | Ok entry -> Ok (entry, true)
  | Error preferred_error -> (
      match legacy_rms_norm_kernel_name input_dtype with
      | None -> Error preferred_error
      | Some legacy ->
          kernel_entry ~name:legacy runtime
            ~operation:Kernel_abi.Operation.Rms_norm ~input_dtype ~output_dtype
          |> Result.map (fun entry -> entry, false))

let select_attention_kernel runtime input_dtype output_dtype head_dimension =
  let candidates =
    if head_dimension <= 64 then
      [ "llmopt_attention_f16_simd_h64", true;
        "llmopt_attention_f16", false ]
    else [ "llmopt_attention_f16", false ]
  in
  let rec select = function
    | [] ->
        Error
          (Printf.sprintf "serving package has no attention kernel for %s -> %s"
             (Ir.Dtype.to_string input_dtype)
             (Ir.Dtype.to_string output_dtype))
    | (name, simd) :: rest -> (
        match
          kernel_entry ~name runtime ~operation:Kernel_abi.Operation.Attention
            ~input_dtype ~output_dtype
        with
        | Ok entry -> Ok (entry, simd)
        | Error _ -> select rest)
  in
  select candidates

let validate_linear_shapes ~m ~n ~k input weight output =
  let input_elements = Tensor_shape.numel (Ir.Value.logical_shape input) in
  let output_elements = Tensor_shape.numel (Ir.Value.logical_shape output) in
  let weight_dimensions =
    Tensor_shape.dimensions (Ir.Value.logical_shape weight)
  in
  if input_elements <> m * k then
    Error "linear input shape is inconsistent with m and k"
  else if weight_dimensions <> [ n; k ] then
    Error "linear weight shape is inconsistent with n and k"
  else if output_elements <> m * n then
    Error "linear output shape is inconsistent with m and n"
  else Ok ()

let same_value_metadata left right =
  Ir.Value.dtype left = Ir.Value.dtype right
  && Tensor_shape.equal
       (Ir.Value.logical_shape left)
       (Ir.Value.logical_shape right)

let dispatch_q8_linear_batched batch runtime ~selection ~epilogue ~dtype ~input_value
    ~input_right_value ~weight_value ~input ~input_right ~weight ~scale ~bias
    ~residual ~output_value ~output ~m ~n ~k =
  let* () = validate_linear_shapes ~m ~n ~k input_value weight_value output_value in
  let* () =
    match input_right_value with
    | None -> Ok ()
    | Some right when same_value_metadata input_value right -> Ok ()
    | Some _ -> Error "Q8 multiplied inputs have different metadata"
  in
  let operation = q8_operation epilogue in
  let* preferred_name = q8_kernel_name dtype ~m ~epilogue in
  let legacy_candidates =
    if m = 1 then
      let* pair_name = q8_pair_kernel_name dtype epilogue in
      Ok
        (match q8_legacy_gemv_name dtype epilogue with
        | Some legacy ->
            [ pair_name, Q8_decode_layout.Simd_pair, None;
              preferred_name, Q8_decode_layout.Simd_single, None;
              legacy, Q8_decode_layout.Scalar, None ]
        | None ->
            [ pair_name, Q8_decode_layout.Simd_pair, None;
              preferred_name, Q8_decode_layout.Simd_single, None ])
    else Ok [ preferred_name, Q8_decode_layout.Scalar, None ]
  in
  let selected_candidate =
    match selection with
    | None -> Ok None
    | Some selected ->
        let mode = Kernel_cost_model.mode selected in
        let* name, layout, tile =
          match mode with
          | Kernel_cost_model.Gemv_pair ->
              q8_pair_kernel_name dtype epilogue
              |> Result.map (fun name ->
                     (name, Q8_decode_layout.Simd_pair, None))
          | Kernel_cost_model.Gemv_single ->
              q8_kernel_name dtype ~m:1 ~epilogue
              |> Result.map (fun name ->
                     (name, Q8_decode_layout.Simd_single, None))
          | Kernel_cost_model.Gemm ->
              (match epilogue with
              | Identity ->
                  let name = Kernel_cost_model.kernel_name selected in
                  let name =
                    match dtype with
                    | Ir.Dtype.Float32 -> name ^ "_f32"
                    | Ir.Dtype.Float16 -> name
                    | _ -> name
                  in
                  Ok
                    ( name,
                      Q8_decode_layout.Scalar,
                      Some (Kernel_cost_model.tile selected) )
              | (Silu | Add | Mul_add) ->
                  q8_kernel_name dtype ~m ~epilogue
                  |> Result.map (fun name ->
                         (name, Q8_decode_layout.Scalar, None)))
        in
        Ok (Some (name, layout, tile))
  in
  let* selected_candidate = selected_candidate in
  let* legacy_candidates = legacy_candidates in
  let candidates =
    let base_candidates =
      match selected_candidate with
      | None -> legacy_candidates
      | Some selected -> selected :: legacy_candidates
    in
    if m >= 8 && m mod 8 = 0 && n mod 8 = 0 && k mod 8 = 0 && epilogue = Identity then
      let simdgroup_gemm_name =
        match dtype with
        | Ir.Dtype.Float16 -> "llmopt_q8_gemm_simdgroup_f16"
        | Ir.Dtype.Float32 -> "llmopt_q8_gemm_simdgroup_f32"
        | _ -> preferred_name
      in
      (simdgroup_gemm_name, Q8_decode_layout.Simdgroup_gemm, Some (8, 8, 8))
      :: base_candidates
    else base_candidates
  in
  let rec select_kernel = function
    | [] ->
        Error
          ("serving package has no Q8 linear kernel for "
          ^ Ir.Dtype.to_string dtype)
    | (name, layout, tile) :: rest ->
        (match q8_kernel ~name runtime ~operation dtype with
        | Some entry -> Ok (entry, layout, tile)
        | None -> select_kernel rest)
  in
  let* entry, layout, tile = select_kernel candidates in
  let bias_buffer, has_bias =
    match bias with Some buffer -> buffer, true | None -> scale, false
  in
  let tile_m, tile_n =
    match tile with
    | Some (tile_m, tile_n, _) -> tile_m, tile_n
    | None -> 16, 16
  in
  let* parameters = Parameters.u32s [ m; n; k; if has_bias then 1 else 0 ] in
  let* grid_x =
    if m = 1 then q8_decode_grid n layout
    else
      match layout with
      | Q8_decode_layout.Simdgroup_gemm ->
          let* blocks_n = round_up n tile_n in
          Ok ((blocks_n / tile_n) * 32)
      | _ -> round_up n tile_n
  in
  let* grid_y =
    if m = 1 then Ok 1
    else
      match layout with
      | Q8_decode_layout.Simdgroup_gemm ->
          let* blocks_m = round_up m tile_m in
          Ok (blocks_m / tile_m)
      | _ -> round_up m tile_m
  in
  let* buffers =
    match epilogue, input_right, residual with
    | (Identity | Silu), None, None ->
        Ok [ input; weight; scale; bias_buffer; output ]
    | Add, None, Some residual ->
        Ok [ input; weight; scale; bias_buffer; residual; output ]
    | Mul_add, Some right, Some residual ->
        Ok [ input; right; weight; scale; bias_buffer; residual; output ]
    | (Add | Mul_add), _, None ->
        Error "Q8 residual epilogue has no residual buffer"
    | Mul_add, None, _ -> Error "Q8 multiplied epilogue has no right input"
    | (Identity | Silu | Add), Some _, _ ->
        Error "Q8 non-multiplied epilogue received a right input"
    | (Identity | Silu), None, Some _ ->
        Error "Q8 non-residual epilogue received a residual buffer"
  in
  dispatch ~batch runtime entry ~buffers ~parameters
    ~grid:(grid_x, grid_y, 1)

let dispatch_q8_command batch runtime state ~selection ~epilogue ~m ~n ~k ~has_bias
    values output =
  let* values, residual_value =
    match epilogue, List.rev values with
    | (Add | Mul_add), residual :: reversed ->
        Ok (List.rev reversed, Some residual)
    | (Add | Mul_add), [] ->
        Error "Q8 residual schedule command has no residual input"
    | (Identity | Silu), _ -> Ok (values, None)
  in
  let* input_value, input_right_value, weight_value, input, input_right, weight,
      scale, bias =
    match epilogue, values, has_bias with
    | (Identity | Silu | Add),
      [ input_value; weight_value; scale_value ], false ->
        let* input = find_value state input_value in
        let* weight = find_value state weight_value in
        let* scale = find_value state scale_value in
        Ok
          ( input_value,
            None,
            weight_value,
            input,
            None,
            weight,
            scale,
            None )
    | (Identity | Silu | Add),
      [ input_value; weight_value; scale_value; bias_value ], true ->
        let* input = find_value state input_value in
        let* weight = find_value state weight_value in
        let* scale = find_value state scale_value in
        let* bias = find_value state bias_value in
        Ok
          ( input_value,
            None,
            weight_value,
            input,
            None,
            weight,
            scale,
            Some bias )
    | Mul_add,
      [ input_value; right_value; weight_value; scale_value ], false ->
        let* input = find_value state input_value in
        let* right = find_value state right_value in
        let* weight = find_value state weight_value in
        let* scale = find_value state scale_value in
        Ok
          ( input_value,
            Some right_value,
            weight_value,
            input,
            Some right,
            weight,
            scale,
            None )
    | Mul_add,
      [ input_value; right_value; weight_value; scale_value; bias_value ], true ->
        let* input = find_value state input_value in
        let* right = find_value state right_value in
        let* weight = find_value state weight_value in
        let* scale = find_value state scale_value in
        let* bias = find_value state bias_value in
        Ok
          ( input_value,
            Some right_value,
            weight_value,
            input,
            Some right,
            weight,
            scale,
            Some bias )
    | _ -> Error "Q8 schedule command has inconsistent bias inputs"
  in
  let* residual =
    match residual_value with
    | None -> Ok None
    | Some value ->
        if
          Ir.Value.dtype value <> Ir.Value.dtype output
          || not
               (Tensor_shape.equal
                  (Ir.Value.logical_shape value)
                  (Ir.Value.logical_shape output))
        then Error "Q8 residual input metadata differs from its output"
        else find_value state value |> Result.map Option.some
  in
  let* output_buffer = workspace_buffer state output in
  let* kernel =
    dispatch_q8_linear_batched batch runtime ~selection ~epilogue
      ~dtype:(Ir.Value.dtype output) ~input_value ~input_right_value ~weight_value
      ~input ~input_right ~weight ~scale ~bias ~residual ~output_value:output
      ~output:output_buffer ~m ~n ~k
  in
  Ok (bind_value state output output_buffer, kernel)

let dispatch_q8_gemm_command = dispatch_q8_command

let q8_macro_kernel_name dtype ~base ~has_bias =
  match dtype with
  | Ir.Dtype.Float16 -> Ok (base ^ "_f16" ^ if has_bias then "_bias" else "")
  | Ir.Dtype.Float32 -> Ok (base ^ "_f32" ^ if has_bias then "_bias" else "")
  | dtype ->
      Error
        (Printf.sprintf "Q8 macro dispatch requires f16 or f32 activations, got %s"
           (Ir.Dtype.to_string dtype))

let dispatch_q8_dual_command batch runtime state ~m ~n1 ~n2 ~k ~bias ~silu_first
    ~extra_outputs values output =
  let* output2 =
    match extra_outputs with
    | [ output2 ] -> Ok output2
    | _ -> Error "Q8 dual-linear command requires one secondary output"
  in
  let* input_value, weight1_value, scale1_value, bias1_value, weight2_value,
      scale2_value, bias2_value =
    match values, bias with
    | [ input; weight1; scale1; weight2; scale2 ], false ->
        Ok (input, weight1, scale1, None, weight2, scale2, None)
    | [ input; weight1; scale1; bias1; weight2; scale2; bias2 ], true ->
        Ok (input, weight1, scale1, Some bias1, weight2, scale2, Some bias2)
    | _ -> Error "Q8 dual-linear command has inconsistent bias inputs"
  in
  let* input = find_value state input_value in
  let* weight1 = find_value state weight1_value in
  let* scale1 = find_value state scale1_value in
  let* weight2 = find_value state weight2_value in
  let* scale2 = find_value state scale2_value in
  let* bias1 = Option.fold ~none:(Ok None)
      ~some:(fun value -> find_value state value |> Result.map Option.some)
      bias1_value
  in
  let* bias2 = Option.fold ~none:(Ok None)
      ~some:(fun value -> find_value state value |> Result.map Option.some)
      bias2_value
  in
  let* output_buffer = workspace_buffer state output in
  let* output2_buffer = workspace_buffer state output2 in
  let* () = validate_linear_shapes ~m ~n:n1 ~k input_value weight1_value output in
  let* () = validate_linear_shapes ~m ~n:n2 ~k input_value weight2_value output2 in
  let* kernel_name =
    q8_macro_kernel_name (Ir.Value.dtype input_value)
      ~base:(if silu_first then "llmopt_q8_dual_linear_silu"
             else "llmopt_q8_dual_linear")
      ~has_bias:bias
  in
  let* entry =
    kernel_entry ~name:kernel_name runtime
      ~operation:Kernel_abi.Operation.Q8_linear
      ~input_dtype:(Ir.Value.dtype input_value)
      ~output_dtype:(Ir.Value.dtype output)
  in
  let* parameters = Parameters.q8_dual_linear ~m ~n1 ~n2 ~k in
  let* bias1 =
    match bias1 with
    | Some bias -> Ok bias
    | None when not bias -> Ok scale1
    | None -> Error "Q8 dual-linear command is missing bias1"
  in
  let* bias2 =
    match bias2 with
    | Some bias -> Ok bias
    | None when not bias -> Ok scale2
    | None -> Error "Q8 dual-linear command is missing bias2"
  in
  let buffers =
    if bias then
      [ input; weight1; scale1; bias1; weight2; scale2; bias2; output_buffer;
        output2_buffer ]
    else [ input; weight1; scale1; weight2; scale2; output_buffer; output2_buffer ]
  in
  let* kernel =
    dispatch ~batch runtime entry ~buffers ~parameters
      ~grid:(n1 + n2, m, 1)
  in
  Ok (bind_value (bind_value state output output_buffer) output2 output2_buffer, kernel)

let dispatch_q8_qkv_command batch runtime state ~m ~n_q ~n_kv ~k ~bias
    ~extra_outputs values output =
  let* key_output, value_output =
    match extra_outputs with
    | [ key_output; value_output ] -> Ok (key_output, value_output)
    | _ -> Error "Q8 QKV command requires key and value secondary outputs"
  in
  let* input_value, weight_q_value, scale_q_value, bias_q_value, weight_k_value,
      scale_k_value, bias_k_value, weight_v_value, scale_v_value, bias_v_value =
    match values, bias with
    | [ input; weight_q; scale_q; weight_k; scale_k; weight_v; scale_v ], false ->
        Ok (input, weight_q, scale_q, None, weight_k, scale_k, None, weight_v,
          scale_v, None)
    | [ input; weight_q; scale_q; bias_q; weight_k; scale_k; bias_k; weight_v;
        scale_v; bias_v ], true ->
        Ok (input, weight_q, scale_q, Some bias_q, weight_k, scale_k, Some bias_k,
          weight_v, scale_v, Some bias_v)
    | _ -> Error "Q8 QKV command has inconsistent bias inputs"
  in
  let* input = find_value state input_value in
  let* weight_q = find_value state weight_q_value in
  let* scale_q = find_value state scale_q_value in
  let* weight_k = find_value state weight_k_value in
  let* scale_k = find_value state scale_k_value in
  let* weight_v = find_value state weight_v_value in
  let* scale_v = find_value state scale_v_value in
  let* bias_q = Option.fold ~none:(Ok None)
      ~some:(fun value -> find_value state value |> Result.map Option.some)
      bias_q_value
  in
  let* bias_k = Option.fold ~none:(Ok None)
      ~some:(fun value -> find_value state value |> Result.map Option.some)
      bias_k_value
  in
  let* bias_v = Option.fold ~none:(Ok None)
      ~some:(fun value -> find_value state value |> Result.map Option.some)
      bias_v_value
  in
  let* output_buffer = workspace_buffer state output in
  let* key_buffer = workspace_buffer state key_output in
  let* value_buffer = workspace_buffer state value_output in
  let* () = validate_linear_shapes ~m ~n:n_q ~k input_value weight_q_value output in
  let* () = validate_linear_shapes ~m ~n:n_kv ~k input_value weight_k_value key_output in
  let* () = validate_linear_shapes ~m ~n:n_kv ~k input_value weight_v_value value_output in
  let* kernel_name =
    q8_macro_kernel_name (Ir.Value.dtype input_value)
      ~base:"llmopt_q8_qkv_linear" ~has_bias:bias
  in
  let* entry =
    kernel_entry ~name:kernel_name runtime
      ~operation:Kernel_abi.Operation.Q8_linear
      ~input_dtype:(Ir.Value.dtype input_value)
      ~output_dtype:(Ir.Value.dtype output)
  in
  let* parameters = Parameters.q8_qkv_linear ~m ~n_q ~n_kv ~k in
  let* bias_q =
    match bias_q with
    | Some bias -> Ok bias
    | None when not bias -> Ok scale_q
    | None -> Error "Q8 QKV command is missing bias_q"
  in
  let* bias_k =
    match bias_k with
    | Some bias -> Ok bias
    | None when not bias -> Ok scale_k
    | None -> Error "Q8 QKV command is missing bias_k"
  in
  let* bias_v =
    match bias_v with
    | Some bias -> Ok bias
    | None when not bias -> Ok scale_v
    | None -> Error "Q8 QKV command is missing bias_v"
  in
  let buffers =
    if bias then
      [ input; weight_q; scale_q; bias_q; weight_k; scale_k; bias_k; weight_v;
        scale_v; bias_v; output_buffer; key_buffer; value_buffer ]
    else
      [ input; weight_q; scale_q; weight_k; scale_k; weight_v; scale_v;
        output_buffer; key_buffer; value_buffer ]
  in
  let* kernel =
    dispatch ~batch runtime entry ~buffers ~parameters
      ~grid:(n_q + (2 * n_kv), m, 1)
  in
  let state = bind_value state output output_buffer in
  let state = bind_value state key_output key_buffer in
  Ok (bind_value state value_output value_buffer, kernel)

let dispatch_q8_lm_head_argmax_command batch runtime state ~m ~n ~k ~epsilon
    ~extra_outputs values output =
  let* input_value, norm_weight_value, weight_value, scale_value =
    match values with
    | [ input; norm_weight; weight; scale ] ->
        Ok (input, norm_weight, weight, scale)
    | _ -> Error "Q8 LM-head argmax command has inconsistent inputs"
  in
  let* input = find_value state input_value in
  let* norm_weight = find_value state norm_weight_value in
  let* weight = find_value state weight_value in
  let* scale = find_value state scale_value in
  let* output_buffer = workspace_buffer state output in
  let* extra_output, extra_output_buffer =
    match extra_outputs with
    | [] -> Ok (None, None)
    | [ extra_output ] ->
        workspace_buffer state extra_output
        |> Result.map (fun buffer -> (Some extra_output, Some buffer))
    | _ -> Error "Q8 LM-head argmax command has too many secondary outputs"
  in
  let* () =
    if
      Tensor_shape.numel (Ir.Value.logical_shape input_value) = m * k
      && Tensor_shape.dimensions (Ir.Value.logical_shape norm_weight_value) = [ k ]
      && Tensor_shape.dimensions (Ir.Value.logical_shape weight_value) = [ n; k ]
      && Tensor_shape.dimensions (Ir.Value.logical_shape scale_value) = [ n ]
      && Tensor_shape.dimensions (Ir.Value.logical_shape output) = [ m ]
      && (Ir.Value.dtype input_value = Ir.Dtype.Float16
          || Ir.Value.dtype input_value = Ir.Dtype.Float32)
      && Ir.Value.dtype norm_weight_value = Ir.Dtype.Float16
      && Ir.Value.dtype weight_value = Ir.Dtype.Int8
      && Ir.Value.dtype scale_value = Ir.Dtype.Float16
      && Ir.Value.dtype output = Ir.Dtype.Int32
      && (match extra_outputs with
         | [] -> true
         | [ logits ] ->
             Tensor_shape.numel (Ir.Value.logical_shape logits) = m * n
             && (match
                   List.rev (Tensor_shape.dimensions (Ir.Value.logical_shape logits))
                 with
                 | last_dimension :: _ -> last_dimension = n
                 | [] -> false)
             && Ir.Value.dtype logits = Ir.Dtype.Float16
         | _ -> false)
    then Ok ()
    else Error "Q8 LM-head argmax input metadata is inconsistent"
  in
  let* kernel_name =
    q8_macro_kernel_name (Ir.Value.dtype input_value)
      ~base:
        (if Option.is_some extra_output then
           "llmopt_q8_lm_head_argmax_extra"
         else "llmopt_q8_lm_head_argmax")
      ~has_bias:false
  in
  let* entry =
    kernel_entry ~name:kernel_name runtime
      ~operation:Kernel_abi.Operation.Q8_lm_head_argmax
      ~input_dtype:(Ir.Value.dtype input_value)
      ~output_dtype:Ir.Dtype.Int32
  in
  let* parameters = Parameters.q8_lm_head_argmax ~m ~n ~k ~epsilon in
  let* grid =
    if m > max_int / 256 then Error "LM-head argmax grid dimension overflows"
    else Ok (m * 256, 1, 1)
  in
  let* kernel =
    let buffers =
      [ input; norm_weight; weight; scale; output_buffer ]
      @ Option.to_list extra_output_buffer
    in
    dispatch ~batch runtime entry
      ~buffers
      ~parameters ~grid
  in
  let state = bind_value state output output_buffer in
  let state =
    match extra_output, extra_output_buffer with
    | Some value, Some buffer -> bind_value state value buffer
    | _ -> state
  in
  Ok (state, kernel)

let dispatch_q8_linear_add_norm_command batch runtime state ~m ~n ~k ~epsilon
    ~extra_outputs values output =
  let* input_value, weight_value, scale_value, residual_value, norm_weight_value =
    match values with
    | [ input; weight; scale; residual; norm_weight ] ->
        Ok (input, weight, scale, residual, norm_weight)
    | _ ->
        Error "Q8 linear-add-norm schedule command has inconsistent inputs"
  in
  let* input = find_value state input_value in
  let* weight = find_value state weight_value in
  let* scale = find_value state scale_value in
  let* residual = find_value state residual_value in
  let* norm_weight = find_value state norm_weight_value in
  let* output_buffer = workspace_buffer state output in
  let* extra_output, extra_output_buffer =
    match extra_outputs with
    | [] -> Ok (None, None)
    | [ extra_output ] ->
        workspace_buffer state extra_output
        |> Result.map (fun buffer -> (Some extra_output, Some buffer))
    | _ -> Error "Q8 linear-add-norm command has too many secondary outputs"
  in
  let* kernel_name =
    match Ir.Value.dtype input_value with
    | Ir.Dtype.Float16 ->
        Ok
          (if Option.is_some extra_output then
             "llmopt_q8_linear_add_norm_extra_f16"
           else "llmopt_q8_linear_add_norm_f16")
    | Ir.Dtype.Float32 ->
        Ok
          (if Option.is_some extra_output then
             "llmopt_q8_linear_add_norm_extra_f32"
           else "llmopt_q8_linear_add_norm_f32")
    | dtype ->
        Error
          (Printf.sprintf "unsupported Q8 linear-add-norm input dtype: %s"
             (Ir.Dtype.to_string dtype))
  in
  let* entry =
    kernel_entry ~name:kernel_name runtime
      ~operation:Kernel_abi.Operation.Q8_linear_add
      ~input_dtype:(Ir.Value.dtype input_value)
      ~output_dtype:(Ir.Value.dtype output)
  in
  let* parameters = Parameters.q8_linear_add_norm ~m ~n ~k ~epsilon in
  let* () =
    if
      Ir.Value.dtype residual_value = Ir.Value.dtype input_value
      && Tensor_shape.equal
           (Ir.Value.logical_shape residual_value)
           (Ir.Value.logical_shape output)
      && Tensor_shape.dimensions (Ir.Value.logical_shape norm_weight_value) = [ n ]
    then Ok ()
    else Error "Q8 linear-add-norm input metadata is inconsistent"
  in
  let* kernel =
    let buffers =
      [ input; weight; scale; residual; norm_weight; output_buffer ]
      @ Option.to_list extra_output_buffer
    in
    dispatch ~batch runtime entry
      ~buffers
      ~parameters ~grid:(m, 1, 1)
  in
  let state = bind_value state output output_buffer in
  let state =
    match extra_output, extra_output_buffer with
    | Some value, Some buffer -> bind_value state value buffer
    | _ -> state
  in
  Ok (state, kernel)

let dispatch_q8_fused_swiglu_ffn_command batch runtime state ~m ~n ~k ~epsilon
    values output =
  let* input_val, residual_val, weight1_val, scale1_val, weight3_val, scale3_val, weight2_val, scale2_val, norm_weight_val =
    match values with
    | [ x; r; w1; s1; w3; s3; w2; s2; nw ] -> Ok (x, r, w1, s1, w3, s3, w2, s2, nw)
    | _ -> Error "Q8 fused SwiGLU FFN schedule command has inconsistent inputs"
  in
  let* input = find_value state input_val in
  let* residual = find_value state residual_val in
  let* weight1 = find_value state weight1_val in
  let* scale1 = find_value state scale1_val in
  let* weight3 = find_value state weight3_val in
  let* scale3 = find_value state scale3_val in
  let* weight2 = find_value state weight2_val in
  let* scale2 = find_value state scale2_val in
  let* norm_weight = find_value state norm_weight_val in
  let* output_buffer = workspace_buffer state output in
  let* kernel_name =
    match Ir.Value.dtype input_val with
    | Ir.Dtype.Float16 -> Ok "llmopt_q8_fused_swiglu_ffn_f16"
    | Ir.Dtype.Float32 -> Ok "llmopt_q8_fused_swiglu_ffn_f32"
    | dtype ->
        Error
          (Printf.sprintf "unsupported Q8 fused SwiGLU FFN input dtype: %s"
             (Ir.Dtype.to_string dtype))
  in
  let* entry =
    kernel_entry ~name:kernel_name runtime
      ~operation:Kernel_abi.Operation.Q8_fused_swiglu_ffn
      ~input_dtype:(Ir.Value.dtype input_val)
      ~output_dtype:(Ir.Value.dtype output)
  in
  let* parameters = Parameters.q8_fused_swiglu_ffn ~m ~n ~k ~epsilon in
  let* () =
    if
      Ir.Value.dtype residual_val = Ir.Value.dtype input_val
      && Tensor_shape.equal
           (Ir.Value.logical_shape residual_val)
           (Ir.Value.logical_shape output)
      && Tensor_shape.dimensions (Ir.Value.logical_shape norm_weight_val) = [ k ]
      && Tensor_shape.dimensions (Ir.Value.logical_shape weight1_val) = [ n; k ]
      && Tensor_shape.dimensions (Ir.Value.logical_shape scale1_val) = [ n ]
      && Tensor_shape.dimensions (Ir.Value.logical_shape weight3_val) = [ n; k ]
      && Tensor_shape.dimensions (Ir.Value.logical_shape scale3_val) = [ n ]
      && Tensor_shape.dimensions (Ir.Value.logical_shape weight2_val) = [ k; n ]
      && Tensor_shape.dimensions (Ir.Value.logical_shape scale2_val) = [ k ]
    then Ok ()
    else Error "Q8 fused SwiGLU FFN input metadata is inconsistent"
  in
  let buffers =
    [ input; residual; weight1; scale1; weight3; scale3; weight2; scale2; norm_weight; output_buffer ]
  in
  let* kernel =
    dispatch ~batch runtime entry
      ~buffers
      ~parameters ~grid:(m * 256, 1, 1)
  in
  let state = bind_value state output output_buffer in
  Ok (state, kernel)

let dispatch_q8_fused_short_conv_command batch runtime state ~m ~channels ~window ~k
    ~epsilon values output =
  let* input_val, residual_val, weight_in_val, scale_in_val, conv_state_val, conv_weight_val, weight_out_val, scale_out_val, norm_weight_val =
    match values with
    | [ x; r; win; sin; cs; cw; wout; sout; nw ] ->
        Ok (x, r, win, sin, cs, cw, wout, sout, nw)
    | _ -> Error "Q8 fused ShortConv schedule command has inconsistent inputs"
  in
  let* input = find_value state input_val in
  let* residual = find_value state residual_val in
  let* weight_in = find_value state weight_in_val in
  let* scale_in = find_value state scale_in_val in
  let* conv_state = find_value state conv_state_val in
  let* conv_weight = find_value state conv_weight_val in
  let* weight_out = find_value state weight_out_val in
  let* scale_out = find_value state scale_out_val in
  let* norm_weight = find_value state norm_weight_val in
  let* output_buffer = workspace_buffer state output in
  let* kernel_name =
    match Ir.Value.dtype input_val with
    | Ir.Dtype.Float16 -> Ok "llmopt_q8_fused_short_conv_f16"
    | Ir.Dtype.Float32 -> Ok "llmopt_q8_fused_short_conv_f32"
    | dtype ->
        Error
          (Printf.sprintf "unsupported Q8 fused ShortConv input dtype: %s"
             (Ir.Dtype.to_string dtype))
  in
  let* entry =
    kernel_entry ~name:kernel_name runtime
      ~operation:Kernel_abi.Operation.Q8_fused_short_conv
      ~input_dtype:(Ir.Value.dtype input_val)
      ~output_dtype:(Ir.Value.dtype output)
  in
  let* parameters =
    Parameters.q8_fused_short_conv ~m ~channels ~window ~k ~epsilon
  in
  let* () =
    if
      Ir.Value.dtype residual_val = Ir.Value.dtype input_val
      && Tensor_shape.equal
           (Ir.Value.logical_shape residual_val)
           (Ir.Value.logical_shape output)
      && Tensor_shape.dimensions (Ir.Value.logical_shape norm_weight_val) = [ k ]
      && Tensor_shape.dimensions (Ir.Value.logical_shape weight_in_val) = [ 3 * channels; k ]
      && Tensor_shape.dimensions (Ir.Value.logical_shape scale_in_val) = [ 3 * channels ]
      && Ir.Value.dtype conv_state_val = Ir.Dtype.Float16
      && Ir.Value.dtype conv_weight_val = Ir.Dtype.Float16
      && Tensor_shape.dimensions (Ir.Value.logical_shape weight_out_val) = [ k; channels ]
      && Tensor_shape.dimensions (Ir.Value.logical_shape scale_out_val) = [ k ]
    then Ok ()
    else Error "Q8 fused ShortConv input metadata is inconsistent"
  in
  let buffers =
    [
      input;
      residual;
      weight_in;
      scale_in;
      conv_state;
      conv_weight;
      weight_out;
      scale_out;
      norm_weight;
      output_buffer;
    ]
  in
  let* kernel =
    dispatch ~batch runtime entry
      ~buffers
      ~parameters ~grid:(m * 256, 1, 1)
  in
  let state = bind_value state output output_buffer in
  Ok (state, kernel)

let split_axis shape axis =
  let rec loop outer remaining = function
    | [] -> Error "Metal axis is outside the tensor rank"
    | width :: rest when remaining = 0 ->
        Ok (List.fold_left ( * ) 1 outer, width, List.fold_left ( * ) 1 rest)
    | dimension :: rest -> loop (dimension :: outer) (remaining - 1) rest
  in
  loop [] axis (Tensor_shape.dimensions shape)

let dispatch_output ?name ?batch runtime state output ~operation ~input_dtype
    ~buffers ~parameters ~grid =
  let output_dtype = Ir.Value.dtype output in
  let* entry = kernel_entry ?name runtime ~operation ~input_dtype ~output_dtype in
  let* output_buffer = workspace_buffer state output in
  let* kernel =
    dispatch ?batch runtime entry ~buffers:(buffers @ [ output_buffer ])
      ~parameters ~grid
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

let reduction_kernel reduction input output =
  match
    reduction.Ir.Reduction.operator,
    reduction.axes,
    Ir.Value.dtype input,
    Ir.Value.dtype output
  with
  | Ir.Reduction.Sum, [ axis ], Ir.Dtype.Float16, Ir.Dtype.Float16 ->
      Ok ("llmopt_sum_f16", axis)
  | _ ->
      Error
        (Printf.sprintf "unsupported Metal reduction kernel: %s %s -> %s"
           (Ir.Reduction.to_string reduction)
           (Ir.Value.dtype input |> Ir.Dtype.to_string)
           (Ir.Value.dtype output |> Ir.Dtype.to_string))

let update_slice_kernel destination source output =
  match
    Ir.Value.dtype destination,
    Ir.Value.dtype source,
    Ir.Value.dtype output
  with
  | Ir.Dtype.Float16, Ir.Dtype.Float16, Ir.Dtype.Float16 ->
      Ok "llmopt_update_slice_f16"
  | destination_dtype, source_dtype, output_dtype ->
      Error
        (Printf.sprintf "unsupported Metal slice-update kernel: %s + %s -> %s"
           (Ir.Dtype.to_string destination_dtype)
           (Ir.Dtype.to_string source_dtype)
           (Ir.Dtype.to_string output_dtype))

let encode_schedule execution_batch ~schedule ~inputs =
  let runtime = execution_batch.Execution_batch.runtime in
  let batch = execution_batch.commands in
  let* runtime_inputs = runtime_input_map inputs in
  let* memory_plan = Serving_memory_plan.create schedule in
  let workspace_bytes = Serving_memory_plan.workspace_bytes memory_plan in
  let* workspace =
    if workspace_bytes = 0 then Ok None
    else Buffer.create ~runtime ~bytes:workspace_bytes |> Result.map Option.some
  in
  execution_batch.resources <-
    List.map snd inputs @ Option.to_list workspace @ execution_batch.resources;
  let dispatch_output ?name = dispatch_output ?name ~batch in
  let rec run state = function
    | [] ->
        Ok
          {
            Execution.outputs = List.rev state.outputs_rev;
            kernels = List.rev state.kernels_rev;
            workspace_bytes;
          }
    | command :: rest ->
        let node_id = Serving_schedule.Command.node_id command in
        let selection =
          Serving_schedule.q8_selection schedule ~node_id
        in
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
            let* buffer = workspace_buffer state output in
            continue (bind_value state output buffer)
        | Ir.Op.Copy _, [ source; destination ], None ->
            let* source = find_value state source in
            let* destination = find_value state destination in
            let* () = Batch.copy batch ~source ~destination in
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
        | Ir.Op.Primitive (Ir.Primitive.Reduce reduction), [ input ], Some output ->
            let* name, axis = reduction_kernel reduction input output in
            let* outer, width, inner =
              split_axis (Ir.Value.logical_shape input) axis
            in
            let* input_buffer = find_value state input in
            let* parameters = Parameters.u32s [ outer; width; inner ] in
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Reduction
                 ~input_dtype:(Ir.Value.dtype input) ~buffers:[ input_buffer ]
                 ~parameters ~grid:(outer * inner, 1, 1))
        | ( Ir.Op.Primitive (Ir.Primitive.Update_slice index),
            [ destination; source ],
            Some output ) ->
            let* name = update_slice_kernel destination source output in
            let* buffers = find_values state [ destination; source ] in
            let* parameters =
              Parameters.update_slice
                ~destination_shape:(Ir.Value.logical_shape destination)
                ~source_shape:(Ir.Value.logical_shape source) index
            in
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Update_slice
                 ~input_dtype:(Ir.Value.dtype destination) ~buffers ~parameters
                 ~grid:
                   ( Tensor_shape.numel (Ir.Value.logical_shape output),
                     1,
                     1 ))
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
        | Ir.Op.Matmul { m; n; k }, [ lhs; rhs ], Some output ->
            let* buffers = find_values state [ lhs; rhs ] in
            let* parameters = Parameters.u32s [ m; n; k ] in
            let* grid_x = round_up n 16 in
            let* grid_y = round_up m 16 in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Matmul
                 ~input_dtype:(Ir.Value.dtype lhs) ~buffers
                 ~parameters ~grid:(grid_x, grid_y, 1))
        | Ir.Op.Fused_matmul_bias { m; n; k }, [ lhs; rhs; bias ], Some output ->
            let* buffers = find_values state [ lhs; rhs; bias ] in
            let* parameters = Parameters.u32s [ m; n; k ] in
            let* grid_x = round_up n 16 in
            let* grid_y = round_up m 16 in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Fused_linear
                 ~input_dtype:(Ir.Value.dtype lhs) ~buffers
                 ~parameters ~grid:(grid_x, grid_y, 1))
        | Ir.Op.Linear { m; n; k; bias }, values, Some output ->
            let* values =
              match values, bias with
              | [ input; weight ], false ->
                  Ok (input, weight, None, [ input; weight ])
              | [ input; weight; bias_value ], true ->
                  Ok
                    ( input,
                      weight,
                      Some bias_value,
                      [ input; weight; bias_value ] )
              | _ -> Error "linear schedule command has inconsistent bias inputs"
            in
            let input, weight, bias_value, values = values in
            let* () = validate_linear_shapes ~m ~n ~k input weight output in
            let* buffers = find_values state values in
            (match
               Ir.Value.dtype input,
               Ir.Value.dtype weight,
               Ir.Value.dtype output,
               bias_value
             with
            | Ir.Dtype.Float16, Ir.Dtype.Float16, Ir.Dtype.Float16, None ->
                let* parameters = Parameters.u32s [ m; n; k ] in
                let* grid_x = linear_f16_grid n in
                dispatched
                  (dispatch_output ~name:"llmopt_linear_f16" runtime state output
                     ~operation:Kernel_abi.Operation.Linear
                     ~input_dtype:Ir.Dtype.Float16 ~buffers ~parameters
                     ~grid:(grid_x, 1, 1))
            | Ir.Dtype.Float32, Ir.Dtype.Float32, Ir.Dtype.Float32, _ ->
                let* parameters = Parameters.u32s [ m; n; k ] in
                let* grid_x = round_up n 16 in
                let* grid_y = round_up m 16 in
                dispatched
                  (dispatch_output runtime state output
                     ~operation:Kernel_abi.Operation.Linear
                     ~input_dtype:(Ir.Value.dtype input) ~buffers
                     ~parameters ~grid:(grid_x, grid_y, 1))
            | input_dtype, weight_dtype, output_dtype, _ ->
                Error
                  (Printf.sprintf "unsupported Metal linear kernel: %s + %s -> %s"
                     (Ir.Dtype.to_string input_dtype)
                     (Ir.Dtype.to_string weight_dtype)
                     (Ir.Dtype.to_string output_dtype)))
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
            let input_dtype = Ir.Value.dtype input in
            let* entry, simd =
              select_rms_norm_kernel runtime input_dtype (Ir.Value.dtype output)
            in
            let* grid_x = if simd then simd_rows_grid rows else Ok rows in
            let* output_buffer = workspace_buffer state output in
            let* kernel =
              dispatch ~batch runtime entry ~buffers:(buffers @ [ output_buffer ])
                ~parameters ~grid:(grid_x, 1, 1)
            in
            dispatched (Ok (bind_value state output output_buffer, kernel))
        | ( Ir.Op.Rms_rope config,
            [ input; weight; cosine; sine ],
            Some output ) ->
            let* batches, tokens, heads, width =
              match Tensor_shape.dimensions (Ir.Value.logical_shape input) with
              | [ batches; tokens; heads; width ]
                when batches > 0 && tokens > 0 && heads > 0 && width > 0 ->
                  Ok (batches, tokens, heads, width)
              | _ -> Error "Metal RMSNorm-RoPE input must have rank four"
            in
            let* trig_batches =
              match
                Tensor_shape.dimensions (Ir.Value.logical_shape cosine),
                Tensor_shape.dimensions (Ir.Value.logical_shape sine)
              with
              | ( [ cosine_batches; 1; cosine_tokens; cosine_width ],
                  [ sine_batches; 1; sine_tokens; sine_width ] )
                when (cosine_batches = 1 || cosine_batches = batches)
                     && sine_batches = cosine_batches
                     && cosine_tokens = tokens && sine_tokens = tokens
                     && cosine_width = width && sine_width = width ->
                  Ok cosine_batches
              | _ ->
                  Error "Metal RMSNorm-RoPE trigonometric tables are inconsistent"
            in
            let half_dimension = Ir.Rms_rope.half_dimension config in
            let output_shape =
              Tensor_shape.dimensions (Ir.Value.logical_shape output)
            in
            if width <> 2 * half_dimension then
              Error "Metal RMSNorm-RoPE width is inconsistent with its half dimension"
            else if
              Tensor_shape.dimensions (Ir.Value.logical_shape weight) <> [ width ]
              || output_shape <> [ batches; heads; tokens; width ]
              || Ir.Value.dtype input <> Ir.Dtype.Float16
              || Ir.Value.dtype weight <> Ir.Dtype.Float16
              || Ir.Value.dtype cosine <> Ir.Dtype.Float16
              || Ir.Value.dtype sine <> Ir.Dtype.Float16
              || Ir.Value.dtype output <> Ir.Dtype.Float16
            then Error "Metal RMSNorm-RoPE tensor metadata is inconsistent"
            else
              let* buffers = find_values state [ input; weight; cosine; sine ] in
              let* parameters =
                Parameters.rms_rope ~batches ~tokens ~heads ~width
                  ~half_dimension ~trig_batches
                  ~epsilon:(Ir.Rms_rope.epsilon config)
              in
              let* entry =
                kernel_entry ~name:"llmopt_rms_rope_f16_simd_h64" runtime
                  ~operation:Kernel_abi.Operation.Rms_rope
                  ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
              in
              let* grid_x = simd_rows_grid (batches * heads * tokens) in
              let* output_buffer = workspace_buffer state output in
              let* kernel =
                dispatch ~batch runtime entry
                  ~buffers:(buffers @ [ output_buffer ]) ~parameters
                  ~grid:(grid_x, 1, 1)
              in
              dispatched (Ok (bind_value state output output_buffer, kernel))
        | ( (Ir.Op.Short_conv_step config | Ir.Op.Short_conv_step_fused config),
            [ in_proj; conv_state; conv_weight ],
            Some output ) ->
            let channels = Ir.Short_conv_step.channels config in
            let* in_proj_buf = find_value state in_proj in
            let* state_buf = find_value state conv_state in
            let* weight_buf = find_value state conv_weight in
            let* output_buf = workspace_buffer state output in
            let* parameters = Parameters.u32s [ channels ] in
            let kernel_name =
              match op with
              | Ir.Op.Short_conv_step _ -> "llmopt_short_conv_step_f16"
              | Ir.Op.Short_conv_step_fused _ ->
                  "llmopt_short_conv_step_fused_f16"
              | _ -> assert false
            in
            let* entry =
              kernel_entry ~name:kernel_name runtime
                ~operation:Kernel_abi.Operation.Short_conv_step
                ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
            in
            let* kernel =
              dispatch ~batch runtime entry
                ~buffers:[ in_proj_buf; state_buf; weight_buf; output_buf ]
                ~parameters ~grid:(channels, 1, 1)
            in
            dispatched (Ok (bind_value state output output_buf, kernel))
        | ( Ir.Op.Short_conv_prefill config,
            [ in_proj; conv_weight; conv_state_out ],
            Some output ) ->
            let channels = Ir.Short_conv_prefill.channels config in
            let* tokens =
              match Tensor_shape.dimensions (Ir.Value.logical_shape in_proj) with
              | [ 1; tokens; _ ] -> Ok tokens
              | _ ->
                  Error
                    "Metal ShortConv prefill in_proj must have shape [1, tokens, \
                     3*channels]"
            in
            let* in_proj_buf = find_value state in_proj in
            let* weight_buf = find_value state conv_weight in
            let* output_buf = workspace_buffer state output in
            let* state_out_buf = find_value state conv_state_out in
            let* parameters = Parameters.u32s [ tokens; channels ] in
            let* entry =
              kernel_entry ~name:"llmopt_short_conv_prefill_f16" runtime
                ~operation:Kernel_abi.Operation.Short_conv_prefill
                ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
            in
            let* kernel =
              dispatch ~batch runtime entry
                ~buffers:[ in_proj_buf; weight_buf; output_buf; state_out_buf ]
                ~parameters ~grid:(channels, 1, 1)
            in
            dispatched (Ok (bind_value state output output_buf, kernel))
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
            let rows = batches * heads * query_length in
            let input_dtype = Ir.Value.dtype query in
            let* entry, simd =
              select_attention_kernel runtime input_dtype (Ir.Value.dtype output)
                head_dimension
            in
            let* grid_x = if simd then simd_rows_grid rows else Ok rows in
            let* output_buffer = workspace_buffer state output in
            let* kernel =
              dispatch ~batch runtime entry ~buffers:(buffers @ [ output_buffer ])
                ~parameters ~grid:(grid_x, 1, 1)
            in
            dispatched (Ok (bind_value state output output_buffer, kernel))
        | ( Ir.Op.Primitive (Ir.Primitive.Paged_attention_q8 config),
            [ query; current_key; current_value; pool; slots; mask ],
            Some output ) ->
            let* batches, query_heads, head_dimension =
              match Tensor_shape.dimensions (Ir.Value.logical_shape query) with
              | [ batches; query_heads; 1; head_dimension ] ->
                  Ok (batches, query_heads, head_dimension)
              | _ -> Error "paged Q8 attention query must have shape [b,h,1,d]"
            in
            let* kv_heads =
              match Tensor_shape.dimensions (Ir.Value.logical_shape current_key) with
              | [ _batches; kv_heads; 1; _head_dimension ] -> Ok kv_heads
              | _ ->
                  Error "paged Q8 attention current key must have shape [b,h,1,d]"
            in
            let* past_length =
              match Tensor_shape.dimensions (Ir.Value.logical_shape slots) with
              | [ past_length ] -> Ok past_length
              | _ -> Error "paged Q8 attention slots must have rank one"
            in
            let* mask_batches, mask_heads =
              match Tensor_shape.dimensions (Ir.Value.logical_shape mask) with
              | [ mask_batches; mask_heads; 1; _key_length ] ->
                  Ok (mask_batches, mask_heads)
              | _ -> Error "paged Q8 attention mask must have rank four"
            in
            let* buffers =
              find_values state
                [ query; current_key; current_value; pool; slots; mask ]
            in
            let* parameters =
              Parameters.paged_attention_q8 ~batches ~query_heads ~kv_heads
                ~past_length ~head_dimension ~mask_batches ~mask_heads
                ~cache_layer:(Ir.Paged_attention_q8.cache_layer config)
                ~attention_layers:
                  (Ir.Paged_attention_q8.attention_layers config)
                ~group_size:(Ir.Paged_attention_q8.group_size config)
                ~token_stride:(Ir.Paged_attention_q8.token_stride config)
                ~scale:(Ir.Paged_attention_q8.scale config)
            in
            let* entry =
              kernel_entry ~name:"llmopt_attention_q8_paged_simd_h64" runtime
                ~operation:Kernel_abi.Operation.Attention
                ~input_dtype:Ir.Dtype.Float16
                ~output_dtype:(Ir.Value.dtype output)
            in
            let* grid_x = simd_rows_grid (batches * query_heads) in
            let* output_buffer = workspace_buffer state output in
            let* kernel =
              dispatch ~batch runtime entry
                ~buffers:(buffers @ [ output_buffer ]) ~parameters
                ~grid:(grid_x, 1, 1)
            in
            dispatched (Ok (bind_value state output output_buffer, kernel))
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
        | Ir.Op.W4a16_linear { m; n; k; bias = has_bias }, values, Some output ->
            let* input, weight, scale, bias_buffer =
              match values, has_bias with
              | [ input; weight; scale ], false ->
                  let* input_buffer = find_value state input in
                  let* weight_buffer = find_value state weight in
                  let* scale_buffer = find_value state scale in
                  Ok (input_buffer, weight_buffer, scale_buffer, scale_buffer)
              | [ input; weight; scale; bias_value ], true ->
                  let* input_buffer = find_value state input in
                  let* weight_buffer = find_value state weight in
                  let* scale_buffer = find_value state scale in
                  let* bias_buffer = find_value state bias_value in
                  Ok (input_buffer, weight_buffer, scale_buffer, bias_buffer)
              | _ ->
                  Error
                    "W4A16 schedule command has inconsistent bias inputs"
            in
            let* parameters =
              Parameters.u32s [ m; n; k; if has_bias then 1 else 0 ]
            in
            dispatched
              (dispatch_output ~name:"llmopt_w4a16_linear_f16_g64" runtime
                 state output ~operation:Kernel_abi.Operation.W4a16_linear
                 ~input_dtype:Ir.Dtype.Float16
                 ~buffers:[ input; weight; scale; bias_buffer ] ~parameters
                 ~grid:(m * n, 1, 1))
        | Ir.Op.Q8_linear { m; n; k; bias = has_bias }, values, Some output ->
            dispatched
              (dispatch_q8_gemm_command batch runtime state ~selection
                 ~epilogue:Identity ~m ~n ~k ~has_bias values output)
        | ( Ir.Op.Q8_linear_silu { m; n; k; bias = has_bias },
            values,
            Some output ) ->
            dispatched
              (dispatch_q8_command batch runtime state ~selection ~epilogue:Silu
                 ~m ~n ~k ~has_bias values output)
        | ( Ir.Op.Q8_linear_add { m; n; k; bias = has_bias },
            values,
            Some output ) ->
            dispatched
              (dispatch_q8_command batch runtime state ~selection ~epilogue:Add
                 ~m ~n ~k ~has_bias values output)
        | ( Ir.Op.Q8_linear_mul_add { m; n; k; bias = has_bias },
            values,
            Some output ) ->
            dispatched
              (dispatch_q8_command batch runtime state ~selection
                 ~epilogue:Mul_add ~m ~n ~k ~has_bias values output)
        | ( Ir.Op.Q8_dual_linear
              { m; n1; n2; k; bias; silu_first; extra_outputs },
            values,
            Some output ) ->
            dispatched
              (dispatch_q8_dual_command batch runtime state ~m ~n1 ~n2 ~k ~bias
                 ~silu_first ~extra_outputs values output)
        | ( Ir.Op.Q8_qkv_linear
              { m; n_q; n_kv; k; bias; extra_outputs },
            values,
            Some output ) ->
            dispatched
              (dispatch_q8_qkv_command batch runtime state ~m ~n_q ~n_kv ~k ~bias
                 ~extra_outputs values output)
        | ( Ir.Op.Q8_lm_head_argmax { m; n; k; epsilon; extra_outputs },
            values,
            Some output ) ->
            dispatched
              (dispatch_q8_lm_head_argmax_command batch runtime state ~m ~n ~k
                 ~epsilon ~extra_outputs values output)
        | ( Ir.Op.Q8_linear_add_norm { m; n; k; epsilon; extra_outputs },
            values,
            Some output ) ->
            dispatched
              (dispatch_q8_linear_add_norm_command batch runtime state ~m ~n ~k
                 ~epsilon ~extra_outputs values output)
        | ( Ir.Op.Q8_fused_swiglu_ffn { m; n; k; epsilon },
            values,
            Some output ) ->
            dispatched
              (dispatch_q8_fused_swiglu_ffn_command batch runtime state ~m ~n ~k
                 ~epsilon values output)
        | ( Ir.Op.Q8_fused_short_conv { m; channels; window; k; epsilon },
            values,
            Some output ) ->
            dispatched
              (dispatch_q8_fused_short_conv_command batch runtime state ~m
                 ~channels ~window ~k ~epsilon values output)
        | Ir.Op.Output { name }, [ input ], None ->
            let* buffer = find_value state input in
            continue
              { state with outputs_rev = (name, buffer) :: state.outputs_rev }
        | Ir.Op.Barrier_create _, [], None
        | Ir.Op.Barrier_arrive _, [], None ->
            continue state
        | Ir.Op.Barrier_wait _, [], None ->
            let* () = Batch.barrier batch in
            continue state
        | _ -> unsupported ())
  in
  let* execution =
    run
      {
        values = Value_map.empty;
        outputs_rev = [];
        kernels_rev = [];
        memory_plan;
        workspace;
      }
      (Serving_schedule.commands schedule)
  in
  execution_batch.schedules <- execution_batch.schedules + 1;
  Ok execution

let execute_schedule runtime ~schedule ~inputs =
  with_execution_batch runtime (fun batch ->
      encode_schedule batch ~schedule ~inputs)

let execute runtime ~inputs =
  execute_schedule runtime
    ~schedule:(Serving_package.schedule runtime.package) ~inputs
