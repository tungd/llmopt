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

external buffer_set_int64_stub : buffer_handle -> int -> int64 -> unit
  = "caml_llmopt_metal_buffer_set_int64"

external buffer_set_u32_array_stub : buffer_handle -> int -> int array -> unit
  = "caml_llmopt_metal_buffer_set_u32_array"

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

external batch_dispatch_all_stub :
  batch_handle -> library_handle ->
  (string * buffer_handle list * bytes * int * int * int * int * int * int) array ->
  unit
  = "caml_llmopt_metal_batch_dispatch_all"

external batch_copy_stub :
  batch_handle -> buffer_handle -> buffer_handle -> unit
  = "caml_llmopt_metal_batch_copy"

external batch_barrier_stub : batch_handle -> unit
  = "caml_llmopt_metal_batch_barrier"

external commit_batch_stub : batch_handle -> unit
  = "caml_llmopt_metal_commit_batch"

external commit_batch_async_stub : batch_handle -> unit
  = "caml_llmopt_metal_commit_batch_async"

external abort_batch_stub : batch_handle -> unit
  = "caml_llmopt_metal_abort_batch"

type ring_handle

external ring_create_stub : unit -> ring_handle = "caml_llmopt_ring_create"

external ring_submit_stub :
  ring_handle -> int -> int -> int -> int -> bool
  = "caml_llmopt_ring_submit"

external ring_wait_completion_stub : ring_handle -> int * int * int
  = "caml_llmopt_ring_wait_completion"

external ring_poll_completion_stub : ring_handle -> (int * int * int) option
  = "caml_llmopt_ring_poll_completion"

external ring_start_worker_stub :
  ring_handle ->
  library_handle ->
  (string * buffer_handle list * bytes * int * int * int * int * int * int)
  array ->
  buffer_handle ->
  buffer_handle ->
  unit = "caml_llmopt_ring_start_worker"

external ring_destroy_stub : ring_handle -> unit = "caml_llmopt_ring_destroy"

type prebaked_handle

external prebaked_create_stub :
  library_handle ->
  (string * buffer_handle list * bytes * int * int * int * int * int * int)
  array ->
  buffer_handle ->
  buffer_handle ->
  prebaked_handle = "caml_llmopt_prebaked_create"

external prebaked_execute_stub :
  prebaked_handle -> int -> int -> int -> int = "caml_llmopt_prebaked_execute"

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
  type dispatch_item = {
    name : string;
    buffers : buffer_handle list;
    parameters : bytes;
    grid_x : int;
    grid_y : int;
    grid_z : int;
    group_x : int;
    group_y : int;
    group_z : int;
  }

  type t = {
    handle : batch_handle;
    context : context_handle;
    mutable pending : (library_handle * dispatch_item) list;
  }

  let create (runtime : runtime) =
    protect (fun () ->
        {
          handle = begin_batch_stub runtime.library;
          context = runtime.context;
          pending = [];
        })

  let flush batch =
    match batch.pending with
    | [] -> Ok ()
    | items ->
        batch.pending <- [];
        let items_rev = List.rev items in
        let rec flush_groups = function
          | [] -> Ok ()
          | (cur_lib, _) :: _ as rest ->
              let group, remainder =
                List.partition (fun (l, _) -> l == cur_lib) rest
              in
              let arr =
                Array.of_list
                  (List.map
                     (fun (_, it) ->
                       ( it.name,
                         it.buffers,
                         it.parameters,
                         it.grid_x,
                         it.grid_y,
                         it.grid_z,
                         it.group_x,
                         it.group_y,
                         it.group_z ))
                     group)
              in
              let* () =
                protect (fun () ->
                    batch_dispatch_all_stub batch.handle cur_lib arr)
              in
              flush_groups remainder
        in
        flush_groups items_rev

  let dispatch batch ~(runtime : runtime) ~name ~buffers ~parameters ~grid
      ~group =
    let grid_x, grid_y, grid_z = grid in
    let group_x, group_y, group_z = group in
    if batch.context != runtime.context then
      Error "Metal batch and dispatch runtime use different device contexts"
    else (
      batch.pending <-
        (runtime.library,
         {
           name;
           buffers;
           parameters;
           grid_x;
           grid_y;
           grid_z;
           group_x;
           group_y;
           group_z;
         })
        :: batch.pending;
      Ok ()
    )

  let copy batch ~source ~destination =
    let* () = flush batch in
    protect (fun () -> batch_copy_stub batch.handle source destination)

  let barrier batch =
    let* () = flush batch in
    protect (fun () -> batch_barrier_stub batch.handle)

  let commit batch =
    let* () = flush batch in
    protect (fun () -> commit_batch_stub batch.handle)

  let commit_async batch =
    let* () = flush batch in
    protect (fun () -> commit_batch_async_stub batch.handle)

  let abort batch =
    batch.pending <- [];
    protect (fun () -> abort_batch_stub batch.handle)
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
            (match Weight_archive.of_file path with
            | Ok archive -> Ok (Some archive)
            | Error _ -> (
                match Gguf.of_file_as_archive path with
                | Ok archive -> Ok (Some archive)
                | Error err -> Error ("cannot load weight store: " ^ err)))
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

  let set_int64 buffer ~offset value =
    protect (fun () -> buffer_set_int64_stub buffer offset value)

  let set_u32_array buffer ~offset values =
    protect (fun () -> buffer_set_u32_array_stub buffer offset values)
end

module Ring_queue = struct
  type t = ring_handle

  let create () = protect ring_create_stub

  let submit ring ~request_id ~token ~past_tokens ~flags =
    protect (fun () ->
        ring_submit_stub ring request_id token past_tokens flags)

  let wait_completion ring =
    protect (fun () -> ring_wait_completion_stub ring)

  let poll_completion ring =
    protect (fun () -> ring_poll_completion_stub ring)

  let start_worker ring ~(runtime : runtime) ~dispatches ~token_buffer
      ~output_buffer =
    protect (fun () ->
        ring_start_worker_stub ring runtime.library dispatches token_buffer
          output_buffer)

  let destroy ring = protect (fun () -> ring_destroy_stub ring)
end

module Prebaked = struct
  type t = prebaked_handle

  let create ~(runtime : runtime) ~dispatches ~token_buffer ~output_buffer =
    protect (fun () ->
        prebaked_create_stub runtime.library dispatches token_buffer output_buffer)

  let execute plan ~token ~past_tokens ~checkpoint =
    protect (fun () -> prebaked_execute_stub plan token past_tokens checkpoint)
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

let kernel_entry ?name runtime ~operation ~input_dtype ~output_dtype =
  let matches =
    Serving_package.kernels runtime.package
    |> List.filter (fun entry ->
           Kernel_abi.Entry.operation entry = operation
           && Kernel_abi.Entry.input_dtype entry = input_dtype
           && Kernel_abi.Entry.output_dtype entry = output_dtype
           &&
           match name with
           | None -> true
           | Some expected -> Kernel_abi.Entry.name entry = expected)
  in
  match matches with
  | [ entry ] -> Ok entry
  | [] ->
      Error
        (Printf.sprintf "serving package has no %s%s kernel for %s -> %s"
           (Kernel_abi.Operation.to_string operation)
           (match name with None -> "" | Some value -> " " ^ value)
           (Ir.Dtype.to_string input_dtype) (Ir.Dtype.to_string output_dtype))
  | entries ->
      Error
        (Printf.sprintf
           "serving package has ambiguous %s kernel selection for %s -> %s: %s"
           (Kernel_abi.Operation.to_string operation)
           (Ir.Dtype.to_string input_dtype) (Ir.Dtype.to_string output_dtype)
           (entries |> List.map Kernel_abi.Entry.name |> String.concat ", "))

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

  let w4a16_lm_head_argmax ~m ~n ~k ~epsilon =
    let bytes = Bytes.make 16 '\000' in
    let* encoded = u32s [ m; n; k ] in
    Bytes.blit encoded 0 bytes 0 12;
    Bytes.set_int32_le bytes 12 (Int32.bits_of_float epsilon);
    Ok bytes

  let w4a16_swiglu_ffn ~m ~n ~k ~epsilon =
    let bytes = Bytes.make 16 '\000' in
    let* encoded = u32s [ m; n; k ] in
    Bytes.blit encoded 0 bytes 0 12;
    Bytes.set_int32_le bytes 12 (Int32.bits_of_float epsilon);
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

  let rms_rope_qk ~batches ~tokens ~q_heads ~k_heads ~width ~half_dimension
      ~trig_batches ~epsilon =
    let bytes = Bytes.make 32 '\000' in
    let* () =
      u32s [ batches; tokens; q_heads; k_heads; width; half_dimension; trig_batches ]
      |> Result.map (fun encoded -> Bytes.blit encoded 0 bytes 0 28)
    in
    Bytes.set_int32_le bytes 28 (Int32.bits_of_float epsilon);
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
      ~mask_batches ~mask_heads ~causal ~scale ?(kv_heads = heads)
      ?(token_first_output = 0) ?(past_length = 0) () =
    let bytes = Bytes.make 48 '\000' in
    let values =
      [ batches; heads; query_length; key_length; head_dimension; mask_batches;
        mask_heads; (if causal then 1 else 0) ]
    in
    let rec write offset = function
      | [] -> Ok ()
      | value :: rest ->
          let* () = set_u32 bytes offset value in
          write (offset + 4) rest
    in
    let* () = write 0 values in
    Bytes.set_int32_le bytes 32 (Int32.bits_of_float scale);
    let* () = set_u32 bytes 36 kv_heads in
    let* () = set_u32 bytes 40 token_first_output in
    let* () = set_u32 bytes 44 past_length in
    Ok bytes

  let paged_attention_q8 ~batches ~query_heads ~kv_heads ~past_length
      ~head_dimension ~mask_batches ~mask_heads ~cache_layer ~attention_layers
      ~group_size ~token_stride ~scale ?(query_length = 1) () =
    let bytes = Bytes.make 52 '\000' in
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
    let* () = set_u32 bytes 48 query_length in
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

  let masked_fill ~output_shape ~input_shape ~mask_shape scalar =
    let* bytes =
      pointwise ~output_shape
        ~left_shape:(Some (Tensor_shape.dimensions input_shape))
        ~right_shape:(Some (Tensor_shape.dimensions mask_shape))
        ~left_scalar:None ~right_scalar:None
    in
    Bytes.set_int32_le bytes 132 (scalar_f32 scalar);
    Ok bytes

  let batched_matmul ~lhs_shape ~rhs_shape ~output_shape =
    let lhs = Tensor_shape.dimensions lhs_shape in
    let rhs = Tensor_shape.dimensions rhs_shape in
    let output = Tensor_shape.dimensions output_shape in
    let split_matrix dimensions =
      match List.rev dimensions with
      | last :: penultimate :: leading ->
          Ok (List.rev leading, penultimate, last)
      | _ -> Error "batched matmul inputs must have rank at least two"
    in
    let* lhs_batch, m, k = split_matrix lhs in
    let* rhs_batch, rhs_k, n = split_matrix rhs in
    let* output_batch, output_m, output_n = split_matrix output in
    let rank = List.length output_batch in
    if rank > 3 then Error "Metal batched matmul supports at most three batch axes"
    else if k <> rhs_k || m <> output_m || n <> output_n then
      Error "batched matmul matrix dimensions are inconsistent"
    else
      let align dimensions =
        List.init (rank - List.length dimensions) (Fun.const 1) @ dimensions
      in
      let lhs_batch = align lhs_batch in
      let rhs_batch = align rhs_batch in
      if
        not
          (List.for_all2
             (fun dimension output -> dimension = 1 || dimension = output)
             lhs_batch output_batch
          && List.for_all2
               (fun dimension output -> dimension = 1 || dimension = output)
               rhs_batch output_batch)
      then Error "batched matmul batch dimensions do not broadcast"
      else
        let pad dimensions = dimensions @ List.init (3 - rank) (Fun.const 1) in
        let values =
          [ Tensor_shape.numel output_shape; m; n; k; rank ]
          @ pad output_batch @ pad lhs_batch @ pad rhs_batch
        in
        u32s values
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
    slots_scratch : Buffer.t;
    pack_slots_scratch : Buffer.t;
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

  let with_batch_async cache operation =
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
          else Batch.commit_async commands
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

  let kernel_names _format =
    ( "llmopt_cache_pack_attention_q8",
      "llmopt_cache_unpack_attention_q8",
      "llmopt_cache_pack_checkpoint_q8",
      "llmopt_cache_unpack_checkpoint_q8" )

  let kernel_dtypes _format =
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

  let select_pack_kernel runtime _format scalar_name dtypes =
    let simd_name = scalar_name ^ "_simd" in
    match cache_entry_opt runtime simd_name dtypes with
    | Some entry -> Ok { entry; dispatch_layout = Pack_layout.Simdgroup }
    | None ->
        let* entry = cache_entry runtime scalar_name dtypes in
        Ok { entry; dispatch_layout = Pack_layout.Scalar }

  let select_unpack_kernel runtime _format scalar_name dtypes =
    let vector_name = scalar_name ^ "_vec4" in
    match cache_entry_opt runtime vector_name dtypes with
    | Some entry -> Ok { entry; unpack_layout = Unpack_layout.Vec4 }
    | None ->
        let* entry = cache_entry runtime scalar_name dtypes in
        Ok { entry; unpack_layout = Unpack_layout.Scalar }

  let create ~runtime ~config =
    if Serving_package.stage runtime.package <> Serving_package.Stage.Serving then
      Error "physical Metal cache requires a serving-stage package"
    else
      let layout = Kv_cache.Config.layout config in
      let format = Kv_cache.Layout.format layout in
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
        let* slots_scratch =
          Buffer.create ~runtime
            ~bytes:(max 32768 (Kv_cache.Config.token_capacity config * 4))
        in
        let* pack_slots_scratch =
          Buffer.create ~runtime
            ~bytes:(max 32768 (Kv_cache.Config.token_capacity config * 4))
        in
      Ok
        {
            runtime;
            config;
            layout;
            token_pool;
            checkpoint_pool;
            slots_scratch;
            pack_slots_scratch;
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

  let slots_buffer ?(pack = false) cache slots =
    let count = Array.length slots in
    if count = 0 then Error "physical attention cache requires at least one slot"
    else
      let capacity = Kv_cache.Config.token_capacity cache.config in
      if count > capacity then
        Error
          (Printf.sprintf "physical cache slots count %d exceeds capacity %d"
             count capacity)
      else
        let scratch =
          if pack then cache.pack_slots_scratch else cache.slots_scratch
        in
        let int_slots = Array.map Kv_cache.Slot.to_int slots in
        let* () = Buffer.set_u32_array scratch ~offset:0 int_slots in
        if pack then Buffer.view ~parent:scratch ~offset:0 ~bytes:(count * 4)
        else Ok scratch

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
    let groups = elements / Kv_cache.Format.q8_group_size in
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
    let* slots_buffer = slots_buffer ~pack cache slots in
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
        if pack then
          pack_grid cache.kernels.pack_checkpoint
            (layer_elements / Kv_cache.Format.q8_group_size)
        else Ok (unpack_grid cache.kernels.unpack_checkpoint layer_elements)
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
    let* slots = slots_buffer cache slots in
    Ok
      [ Serving_schedule.Sequence.q8_attention_pool_input, cache.token_pool;
        Serving_schedule.Sequence.q8_attention_slots_input, slots ]

  let update_pack_slot cache slot =
    Buffer.set_u32_array cache.pack_slots_scratch ~offset:0
      [| Kv_cache.Slot.to_int slot |]
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
    mutable swiglu_normalized : Buffer.t option;
    mutable swiglu_product : Buffer.t option;
    mutable candidates_buffer : Buffer.t option;
  }
end

let with_execution_batch runtime operation =
  let* commands = Batch.create runtime in
  let batch =
    {
      Execution_batch.runtime;
      commands;
      resources = [];
      schedules = 0;
      swiglu_normalized = None;
      swiglu_product = None;
      candidates_buffer = None;
    }
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
  resources : Buffer.t list;
}

let value_byte_length = Serving_memory_plan.value_bytes

let validate_buffer value buffer =
  let* expected = value_byte_length value in
  let actual = Buffer.byte_length buffer in
  if actual >= expected then Ok ()
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

let linear_f16_grid columns =
  let simdgroups_per_threadgroup = 8 in
  let simd_width = 32 in
  let* rounded_columns = round_up columns simdgroups_per_threadgroup in
  if rounded_columns > max_int / simd_width then
    Error "Metal float16 linear grid dimension overflows"
  else Ok (rounded_columns * simd_width)

let short_row_quant_grid column_groups =
  let simdgroups_per_threadgroup = 2 in
  let simd_width = 32 in
  let* rounded_groups = round_up column_groups simdgroups_per_threadgroup in
  if rounded_groups > max_int / simd_width then
    Error "Metal short-row quantized Linear grid dimension overflows"
  else Ok (rounded_groups * simd_width)

let gated_delta_grid state_rows =
  let simdgroups_per_threadgroup = 4 in
  let simd_width = 32 in
  let* rounded_rows = round_up state_rows simdgroups_per_threadgroup in
  if rounded_rows > max_int / simd_width then
    Error "Metal gated-delta grid dimension overflows"
  else Ok (rounded_rows * simd_width)

let simd_rows_grid rows = linear_f16_grid rows

let wide_rows_grid rows =
  let threadgroup_width = 256 in
  if rows > max_int / threadgroup_width then
    Error "Metal wide-row grid dimension overflows"
  else Ok (rows * threadgroup_width)

let use_wide_rows ~rows ~width =
  let simdgroups_per_threadgroup = 8 in
  let simd_width = 32 in
  rows < simdgroups_per_threadgroup
  && width / simd_width >= simdgroups_per_threadgroup

let rotary_trig_dimensions value =
  match Tensor_shape.dimensions (Ir.Value.logical_shape value) with
  | [ batches; 1; tokens; width ]
  | [ batches; tokens; 1; width ] ->
      Some (batches, tokens, width)
  | _ -> None

let rotary_trig_batches ~batches ~tokens ~width cosine sine =
  match rotary_trig_dimensions cosine, rotary_trig_dimensions sine with
  | ( Some (cosine_batches, cosine_tokens, cosine_width),
      Some (sine_batches, sine_tokens, sine_width) )
    when (cosine_batches = 1 || cosine_batches = batches)
         && sine_batches = cosine_batches && cosine_tokens = tokens
         && sine_tokens = tokens && cosine_width = width
         && sine_width = width ->
      Ok cosine_batches
  | _ -> Error "Metal rotary trigonometric tables are inconsistent"

let rms_norm_kernel_name ~wide input_dtype weight_dtype =
  match input_dtype, weight_dtype with
  | Ir.Dtype.Float32, Ir.Dtype.Float16 ->
      Ok "llmopt_rms_norm_f32_f16_simd"
  | Ir.Dtype.Float16, Ir.Dtype.Float16 -> Ok "llmopt_rms_norm_f16_simd"
  | Ir.Dtype.Float32, Ir.Dtype.Float32 ->
      Ok "llmopt_rms_norm_f32_f16_wf32_simd"
  | Ir.Dtype.Float16, Ir.Dtype.Float32 when wide ->
      Ok "llmopt_rms_norm_f16_wf32_wide"
  | Ir.Dtype.Float16, Ir.Dtype.Float32 ->
      Ok "llmopt_rms_norm_f16_wf32_simd"
  | _ ->
      Error
        (Printf.sprintf "unsupported Metal RMSNorm dtypes: input=%s weight=%s"
           (Ir.Dtype.to_string input_dtype) (Ir.Dtype.to_string weight_dtype))

let legacy_rms_norm_kernel_name input_dtype weight_dtype =
  match input_dtype, weight_dtype with
  | Ir.Dtype.Float32, Ir.Dtype.Float16 -> Some "llmopt_rms_norm_f32_f16"
  | Ir.Dtype.Float16, Ir.Dtype.Float16 -> Some "llmopt_rms_norm_f16"
  | _ -> None

let select_rms_norm_kernel runtime ~wide input_dtype weight_dtype output_dtype =
  let* preferred = rms_norm_kernel_name ~wide input_dtype weight_dtype in
  match
    kernel_entry ~name:preferred runtime
      ~operation:Kernel_abi.Operation.Rms_norm ~input_dtype ~output_dtype
  with
  | Ok entry -> Ok (entry, true)
  | Error preferred_error -> (
      match legacy_rms_norm_kernel_name input_dtype weight_dtype with
      | None -> Error preferred_error
      | Some legacy ->
          kernel_entry ~name:legacy runtime
            ~operation:Kernel_abi.Operation.Rms_norm ~input_dtype ~output_dtype
          |> Result.map (fun entry -> entry, false))

let select_attention_kernel runtime input_dtype output_dtype head_dimension =
  let specialized =
    Printf.sprintf "llmopt_attention_f16_simd_h%d" head_dimension
  in
  let candidates =
    if head_dimension mod 32 = 0 then
      [ specialized, true; "llmopt_attention_f16", false ]
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

let quant_linear_kernel_name = function
  | Ir.Dtype.Q8_0 -> "llmopt_q8_0_linear_f16"
  | Ir.Dtype.Q4_K -> "llmopt_q4_k_linear_f16"
  | Ir.Dtype.Q5_K -> "llmopt_q5_k_linear_f16"
  | Ir.Dtype.Q6_K -> "llmopt_q6_k_linear_f16"
  | Ir.Dtype.Q5_0 -> "llmopt_q5_0_linear_f16"
  | Ir.Dtype.Q4_0 -> "llmopt_q4_0_linear_f16"
  | Ir.Dtype.IQ4_XS -> "llmopt_iq4_xs_linear_f16"

let quant_linear_kernel_names ~m quant =
  let generic = quant_linear_kernel_name quant in
  let specialized =
    match quant with
    | Ir.Dtype.Q4_K -> Some (generic ^ "_m2_x2_l32")
    | Ir.Dtype.Q5_K -> Some (generic ^ "_m2_x1_l32")
    | _ -> None
  in
  if m = 2 then
    Option.to_list specialized
    @ [ generic ^ "_m2_x4"; generic ^ "_m2"; generic ]
  else [ generic ]

let select_quant_linear_kernel runtime ~m quant =
  let input_dtype = Ir.Dtype.Quant quant in
  let rec select = function
    | [] ->
        Error
          (Printf.sprintf "serving package has no quantized Linear kernel for %s"
             (Ir.Dtype.to_string input_dtype))
    | name :: rest -> (
        match
          kernel_entry ~name runtime ~operation:Kernel_abi.Operation.Linear
            ~input_dtype ~output_dtype:Ir.Dtype.Float16
        with
        | Ok entry -> Ok entry
        | Error _ -> select rest)
  in
  select (quant_linear_kernel_names ~m quant)

let same_value_metadata left right =
  Ir.Value.dtype left = Ir.Value.dtype right
  && Tensor_shape.equal
       (Ir.Value.logical_shape left)
       (Ir.Value.logical_shape right)

let macro_kernel_name dtype ~base ~has_bias =
  match dtype with
  | Ir.Dtype.Float16 -> Ok (base ^ "_f16" ^ if has_bias then "_bias" else "")
  | Ir.Dtype.Float32 -> Ok (base ^ "_f32" ^ if has_bias then "_bias" else "")
  | dtype ->
      Error
        (Printf.sprintf "Q8 macro dispatch requires f16 or f32 activations, got %s"
           (Ir.Dtype.to_string dtype))

let dispatch_w4a16_swiglu_ffn_command execution_batch batch runtime state ~m ~n ~k ~epsilon
    values output =
  let* activation_value, residual_value, gate_weight_value, gate_scale_value,
       up_weight_value, up_scale_value, down_weight_value, down_scale_value,
       norm_weight_value =
    match values with
    | [ activation; residual; gate_weight; gate_scale; up_weight; up_scale;
        down_weight; down_scale; norm_weight ] ->
        Ok
          ( activation, residual, gate_weight, gate_scale, up_weight, up_scale,
            down_weight, down_scale, norm_weight )
    | _ -> Error "W4A16 SwiGLU command has inconsistent inputs"
  in
  let* buffers = find_values state values in
  let* activation, residual, gate_weight, gate_scale, up_weight, up_scale,
       down_weight, down_scale, norm_weight =
    match buffers with
    | [ activation; residual; gate_weight; gate_scale; up_weight; up_scale;
        down_weight; down_scale; norm_weight ] ->
        Ok
          ( activation, residual, gate_weight, gate_scale, up_weight, up_scale,
            down_weight, down_scale, norm_weight )
    | _ -> Error "W4A16 SwiGLU buffer binding is inconsistent"
  in
  let* output_buffer = workspace_buffer state output in
  let dimensions value =
    Ir.Value.logical_shape value |> Tensor_shape.dimensions
  in
  let* () =
    if
      Float.is_finite epsilon && epsilon > 0.0
      && m > 0 && n > 0 && k > 0 && n mod 64 = 0 && k mod 64 = 0
      && k <= 2048 && n <= 8192
      && (Ir.Value.dtype activation_value = Ir.Dtype.Float16
          || Ir.Value.dtype activation_value = Ir.Dtype.Float32)
      && Ir.Value.dtype residual_value = Ir.Dtype.Float16
      && Ir.Value.dtype gate_weight_value = Ir.Dtype.UInt8
      && Ir.Value.dtype gate_scale_value = Ir.Dtype.Float16
      && Ir.Value.dtype up_weight_value = Ir.Dtype.UInt8
      && Ir.Value.dtype up_scale_value = Ir.Dtype.Float16
      && Ir.Value.dtype down_weight_value = Ir.Dtype.UInt8
      && Ir.Value.dtype down_scale_value = Ir.Dtype.Float16
      && Ir.Value.dtype norm_weight_value = Ir.Dtype.Float16
      && Ir.Value.dtype output = Ir.Dtype.Float16
      && Tensor_shape.numel (Ir.Value.logical_shape activation_value) = m * k
      && Tensor_shape.equal (Ir.Value.logical_shape residual_value)
           (Ir.Value.logical_shape output)
      && Tensor_shape.numel (Ir.Value.logical_shape output) = m * k
      && dimensions gate_weight_value = [ n; k / 2 ]
      && dimensions gate_scale_value = [ n; k / 64 ]
      && dimensions up_weight_value = [ n; k / 2 ]
      && dimensions up_scale_value = [ n; k / 64 ]
      && dimensions down_weight_value = [ k; n / 2 ]
      && dimensions down_scale_value = [ k; n / 64 ]
      && dimensions norm_weight_value = [ k ]
    then Ok ()
    else Error "W4A16 SwiGLU input metadata is inconsistent"
  in
  let* rms_entry =
    let* name =
      match Ir.Value.dtype activation_value with
      | Ir.Dtype.Float16 -> Ok "llmopt_w4a16_swiglu_rms_f16_g64"
      | Ir.Dtype.Float32 -> Ok "llmopt_w4a16_swiglu_rms_f32_g64"
      | _ -> Error "W4A16 SwiGLU activation must be float16 or float32"
    in
    kernel_entry ~name runtime
      ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
      ~input_dtype:(Ir.Value.dtype activation_value)
      ~output_dtype:Ir.Dtype.Float16
  in
  let* dual_entry =
    kernel_entry ~name:"llmopt_w4a16_dual_swiglu_f16_g64" runtime
      ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
  in
  let* down_entry =
    kernel_entry ~name:"llmopt_w4a16_down_add_f16_g64" runtime
      ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
      ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
  in
  let* normalized =
    match execution_batch.Execution_batch.swiglu_normalized with
    | Some buf when Buffer.byte_length buf >= m * k * 2 -> Ok buf
    | _ ->
        let* buf = Buffer.create ~runtime ~bytes:(m * k * 2) in
        execution_batch.Execution_batch.swiglu_normalized <- Some buf;
        execution_batch.Execution_batch.resources <- buf :: execution_batch.Execution_batch.resources;
        Ok buf
  in
  let* product =
    match execution_batch.Execution_batch.swiglu_product with
    | Some buf when Buffer.byte_length buf >= m * n * 2 -> Ok buf
    | _ ->
        let* buf = Buffer.create ~runtime ~bytes:(m * n * 2) in
        execution_batch.Execution_batch.swiglu_product <- Some buf;
        execution_batch.Execution_batch.resources <- buf :: execution_batch.Execution_batch.resources;
        Ok buf
  in
  let* parameters = Parameters.w4a16_swiglu_ffn ~m ~n ~k ~epsilon in
  let* rms_kernel =
    dispatch ~batch runtime rms_entry
      ~buffers:[ activation; norm_weight; normalized ] ~parameters
      ~grid:(m * 256, 1, 1)
  in
  let* dual_kernel =
    if m = 1 then
      let* dual_grid_x = linear_f16_grid (m * n) in
      dispatch ~batch runtime dual_entry
        ~buffers:[ normalized; gate_weight; gate_scale; up_weight; up_scale; product ]
        ~parameters ~grid:(dual_grid_x, 1, 1)
    else
      let m_blocks = (m + 3) / 4 in
      let* dual_entry_m4 =
        kernel_entry ~name:"llmopt_w4a16_dual_swiglu_f16_g64_m4" runtime
          ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
          ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
      in
      let* dual_grid_x = linear_f16_grid (m_blocks * n) in
      dispatch ~batch runtime dual_entry_m4
        ~buffers:[ normalized; gate_weight; gate_scale; up_weight; up_scale; product ]
        ~parameters ~grid:(dual_grid_x, 1, 1)
  in
  let* down_kernel =
    if m = 1 then
      let* down_grid_x = linear_f16_grid (m * k) in
      dispatch ~batch runtime down_entry
        ~buffers:[ product; down_weight; down_scale; residual; output_buffer ]
        ~parameters ~grid:(down_grid_x, 1, 1)
    else
      let m_blocks = (m + 3) / 4 in
      let* down_entry_m4 =
        kernel_entry ~name:"llmopt_w4a16_down_add_f16_g64_m4" runtime
          ~operation:Kernel_abi.Operation.W4a16_swiglu_ffn
          ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
      in
      let* down_grid_x = linear_f16_grid (m_blocks * k) in
      dispatch ~batch runtime down_entry_m4
        ~buffers:[ product; down_weight; down_scale; residual; output_buffer ]
        ~parameters ~grid:(down_grid_x, 1, 1)
  in
  let state = bind_value state output output_buffer in
  Ok (state, [ rms_kernel; dual_kernel; down_kernel ])

let dispatch_w4a16_qkv_linear_command batch runtime state ~m ~k ~n_q ~n_k ~n_v
    ~extra_outputs values output =
  let* input_value, q_weight_value, q_scale_value, k_weight_value,
       k_scale_value, v_weight_value, v_scale_value =
    match values with
    | [ input; qw; qs; kw; ks; vw; vs ] -> Ok (input, qw, qs, kw, ks, vw, vs)
    | _ -> Error "W4A16 QKV linear command has inconsistent inputs"
  in
  let* input = find_value state input_value in
  let* q_weight = find_value state q_weight_value in
  let* q_scale = find_value state q_scale_value in
  let* k_weight = find_value state k_weight_value in
  let* k_scale = find_value state k_scale_value in
  let* v_weight = find_value state v_weight_value in
  let* v_scale = find_value state v_scale_value in
  let* q_output_buffer = workspace_buffer state output in
  let* k_output_value, v_output_value =
    match extra_outputs with
    | [ k; v ] -> Ok (k, v)
    | _ -> Error "W4A16 QKV linear command requires 2 extra outputs (K and V)"
  in
  let* k_output_buffer = workspace_buffer state k_output_value in
  let* v_output_buffer = workspace_buffer state v_output_value in
  let* parameters = Parameters.u32s [ m; k; n_q; n_k; n_v ] in
  let total_cols = n_q + n_k + n_v in
  let* kernel =
    if m = 1 then
      let* entry =
        kernel_entry ~name:"llmopt_w4a16_qkv_linear_f16_g64" runtime
          ~operation:Kernel_abi.Operation.W4a16_qkv_linear
          ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
      in
      let* grid_x = linear_f16_grid (m * total_cols) in
      dispatch ~batch runtime entry
        ~buffers:
          [ input; q_weight; q_scale; k_weight; k_scale; v_weight; v_scale;
            q_output_buffer; k_output_buffer; v_output_buffer ]
        ~parameters ~grid:(grid_x, 1, 1)
    else
      let m_blocks = (m + 3) / 4 in
      let* entry =
        kernel_entry ~name:"llmopt_w4a16_qkv_linear_f16_g64_m4" runtime
          ~operation:Kernel_abi.Operation.W4a16_qkv_linear
          ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
      in
      let* grid_x = linear_f16_grid (m_blocks * total_cols) in
      dispatch ~batch runtime entry
        ~buffers:
          [ input; q_weight; q_scale; k_weight; k_scale; v_weight; v_scale;
            q_output_buffer; k_output_buffer; v_output_buffer ]
        ~parameters ~grid:(grid_x, 1, 1)
  in
  let state = bind_value state output q_output_buffer in
  let state = bind_value state k_output_value k_output_buffer in
  let state = bind_value state v_output_value v_output_buffer in
  Ok (state, [ kernel ])

let dispatch_w4a16_lm_head_argmax_command execution_batch batch runtime state ~m ~n ~k ~epsilon
    ~extra_outputs values output =
  let* input_value, norm_weight_value, weight_value, scale_value =
    match values with
    | [ input; norm_weight; weight; scale ] ->
        Ok (input, norm_weight, weight, scale)
    | _ -> Error "W4A16 LM-head argmax command has inconsistent inputs"
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
    | _ -> Error "W4A16 LM-head argmax command has too many secondary outputs"
  in
  let* () =
    if
      Tensor_shape.numel (Ir.Value.logical_shape input_value) = m * k
      && Tensor_shape.dimensions (Ir.Value.logical_shape norm_weight_value) = [ k ]
      && k mod 64 = 0
      && Tensor_shape.dimensions (Ir.Value.logical_shape weight_value) =
         [ n; k / 2 ]
      && Tensor_shape.dimensions (Ir.Value.logical_shape scale_value) =
         [ n; k / 64 ]
      && Tensor_shape.dimensions (Ir.Value.logical_shape output) = [ m ]
      && (Ir.Value.dtype input_value = Ir.Dtype.Float16
          || Ir.Value.dtype input_value = Ir.Dtype.Float32)
      && Ir.Value.dtype norm_weight_value = Ir.Dtype.Float16
      && Ir.Value.dtype weight_value = Ir.Dtype.UInt8
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
    else Error "W4A16 LM-head argmax input metadata is inconsistent"
  in
  let stage1_kernel_name =
    macro_kernel_name (Ir.Value.dtype input_value)
      ~base:
        (if Option.is_some extra_output then
           "llmopt_w4a16_lm_head_argmax_stage1_extra"
         else "llmopt_w4a16_lm_head_argmax_stage1")
      ~has_bias:false
  in
  let stage1_entry =
    let* name = stage1_kernel_name in
    kernel_entry ~name runtime
      ~operation:Kernel_abi.Operation.W4a16_lm_head_argmax
      ~input_dtype:(Ir.Value.dtype input_value)
      ~output_dtype:Ir.Dtype.Int32
  in
  let reduce_entry =
    kernel_entry ~name:"llmopt_w4a16_lm_head_reduce" runtime
      ~operation:Kernel_abi.Operation.W4a16_lm_head_argmax
      ~input_dtype:Ir.Dtype.Int32
      ~output_dtype:Ir.Dtype.Int32
  in
  match stage1_entry, reduce_entry with
  | Ok entry1, Ok entry2 ->
      let candidate_bytes = m * 256 * 8 in
      let* candidates_buffer =
        match execution_batch.Execution_batch.candidates_buffer with
        | Some buf when Buffer.byte_length buf >= candidate_bytes -> Ok buf
        | _ ->
            let* buf = Buffer.create ~runtime ~bytes:candidate_bytes in
            execution_batch.Execution_batch.candidates_buffer <- Some buf;
            execution_batch.Execution_batch.resources <- buf :: execution_batch.Execution_batch.resources;
            Ok buf
      in
      let* parameters1 = Parameters.w4a16_lm_head_argmax ~m ~n ~k ~epsilon in
      let* parameters2 = Parameters.u32s [ m ] in
      let buffers1 =
        [ input; norm_weight; weight; scale; candidates_buffer ]
        @ Option.to_list extra_output_buffer
      in
      let tile =
        Kernel_cost_model.Megakernel.select_lm_head_tile
          ~device:Kernel_cost_model.Device.default ~vocab_size:n
      in
      let* kernel1 =
        dispatch ~batch runtime entry1
          ~buffers:buffers1 ~parameters:parameters1
          ~grid:(tile.stage1_threadgroups * tile.stage1_threads, m, 1)
      in
      let buffers2 = [ candidates_buffer; output_buffer ] in
      let* kernel2 =
        dispatch ~batch runtime entry2
          ~buffers:buffers2 ~parameters:parameters2
          ~grid:(m * 256, 1, 1)
      in
      let state = bind_value state output output_buffer in
      let state =
        match extra_output, extra_output_buffer with
        | Some value, Some buffer -> bind_value state value buffer
        | _ -> state
      in
      Ok (state, kernel1 ^ "+" ^ kernel2)
  | _ ->
      let* kernel_name =
        macro_kernel_name (Ir.Value.dtype input_value)
          ~base:
            (if Option.is_some extra_output then
               "llmopt_w4a16_lm_head_argmax_extra"
             else "llmopt_w4a16_lm_head_argmax")
          ~has_bias:false
      in
      let* entry =
        kernel_entry ~name:kernel_name runtime
          ~operation:Kernel_abi.Operation.W4a16_lm_head_argmax
          ~input_dtype:(Ir.Value.dtype input_value)
          ~output_dtype:Ir.Dtype.Int32
      in
      let* parameters = Parameters.w4a16_lm_head_argmax ~m ~n ~k ~epsilon in
      let* grid =
        if m > max_int / 256 then Error "LM-head argmax grid dimension overflows"
        else Ok (m * 256, 1, 1)
      in
      let buffers =
        [ input; norm_weight; weight; scale; output_buffer ]
        @ Option.to_list extra_output_buffer
      in
      let* kernel =
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
  match operation, tensor_dtypes, output_dtype with
  | Ir.Pointwise.Binary (Ir.Pointwise.Mul, _, _),
    [ Ir.Dtype.Float16; Ir.Dtype.Float32 ], Ir.Dtype.Float32 ->
      Ok ("llmopt_mul_f16_f32", Ir.Dtype.Float16)
  | _ ->
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
    | Ir.Pointwise.Binary (Ir.Pointwise.Add, _, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_add_f32"
    | Ir.Pointwise.Binary (Ir.Pointwise.Add, _, _), Ir.Dtype.Int64,
      Ir.Dtype.Int64 ->
        Some "llmopt_add_i64"
    | Ir.Pointwise.Binary (Ir.Pointwise.Sub, _, _), Ir.Dtype.Int64,
      Ir.Dtype.Int64 ->
        Some "llmopt_sub_i64"
    | Ir.Pointwise.Binary (Ir.Pointwise.Sub, _, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_sub_f32"
    | Ir.Pointwise.Binary (Ir.Pointwise.Div, _, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_div_f16"
    | Ir.Pointwise.Binary (Ir.Pointwise.Mul, _, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_mul_f16"
    | Ir.Pointwise.Binary (Ir.Pointwise.Mul, _, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_mul_f32"
    | Ir.Pointwise.Binary (Ir.Pointwise.Silu_mul, _, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_silu_mul_f16"
    | Ir.Pointwise.Binary (Ir.Pointwise.Gelu_mul, _, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_gelu_mul_f16"
    | Ir.Pointwise.Binary (Ir.Pointwise.Sigmoid_mul, _, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_sigmoid_mul_f16"
    | Ir.Pointwise.Binary (Ir.Pointwise.Less_equal, _, _), Ir.Dtype.Int64,
      Ir.Dtype.Bool ->
        Some "llmopt_le_i64"
    | Ir.Pointwise.Binary (Ir.Pointwise.Greater, _, _), Ir.Dtype.Int64,
      Ir.Dtype.Bool ->
        Some "llmopt_gt_i64"
    | Ir.Pointwise.Binary (Ir.Pointwise.Equal, _, _), Ir.Dtype.Int64,
      Ir.Dtype.Bool ->
        Some "llmopt_eq_i64"
    | Ir.Pointwise.Binary (Ir.Pointwise.Not_equal, _, _), Ir.Dtype.Int64,
      Ir.Dtype.Bool ->
        Some "llmopt_ne_i64"
    | Ir.Pointwise.Binary (Ir.Pointwise.Logical_and, _, _), Ir.Dtype.Bool,
      Ir.Dtype.Bool ->
        Some "llmopt_and_bool"
    | Ir.Pointwise.Unary (Ir.Pointwise.Neg, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_neg_f16"
    | Ir.Pointwise.Unary (Ir.Pointwise.Neg, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_neg_f32"
    | Ir.Pointwise.Unary (Ir.Pointwise.Silu, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_silu_f16"
    | Ir.Pointwise.Unary (Ir.Pointwise.Silu, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_silu_f32"
    | Ir.Pointwise.Unary (Ir.Pointwise.Cos, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_cos_f32"
    | Ir.Pointwise.Unary (Ir.Pointwise.Sin, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_sin_f32"
    | Ir.Pointwise.Unary (Ir.Pointwise.Tanh, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_tanh_f16"
    | Ir.Pointwise.Unary (Ir.Pointwise.Exp, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_exp_f32"
    | Ir.Pointwise.Unary (Ir.Pointwise.Sigmoid, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_sigmoid_f16"
    | Ir.Pointwise.Unary (Ir.Pointwise.Softplus, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_softplus_f32"
    | Ir.Pointwise.Unary (Ir.Pointwise.Pow _, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_pow_f32"
    | Ir.Pointwise.Unary (Ir.Pointwise.Rsqrt, _), Ir.Dtype.Float16,
      Ir.Dtype.Float16 ->
        Some "llmopt_rsqrt_f16"
    | Ir.Pointwise.Unary (Ir.Pointwise.Rsqrt, _), Ir.Dtype.Float32,
      Ir.Dtype.Float32 ->
        Some "llmopt_rsqrt_f32"
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
      | Ir.Movement.Expand, Ir.Dtype.Int64 -> Some "llmopt_expand_i64"
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
  | Ir.Reduction.Sum, [ axis ], Ir.Dtype.Float32, Ir.Dtype.Float32 ->
      Ok ("llmopt_sum_f32", axis)
  | Ir.Reduction.Mean, [ axis ], Ir.Dtype.Float32, Ir.Dtype.Float32 ->
      Ok ("llmopt_mean_f32", axis)
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
  | Ir.Dtype.Float32, Ir.Dtype.Float32, Ir.Dtype.Float32 ->
      Ok "llmopt_update_slice_f32"
  | destination_dtype, source_dtype, output_dtype ->
      Error
        (Printf.sprintf "unsupported Metal slice-update kernel: %s + %s -> %s"
           (Ir.Dtype.to_string destination_dtype)
           (Ir.Dtype.to_string source_dtype)
           (Ir.Dtype.to_string output_dtype))

let encode_schedule ?workspace ?memory_plan execution_batch ~schedule ~inputs =
  let runtime = execution_batch.Execution_batch.runtime in
  let batch = execution_batch.commands in
  let* runtime_inputs = runtime_input_map inputs in
  let* memory_plan =
    match memory_plan with
    | Some plan -> Ok plan
    | None -> Serving_memory_plan.create schedule
  in
  let workspace_bytes = Serving_memory_plan.workspace_bytes memory_plan in
  let* workspace =
    match workspace with
    | Some buf ->
        if Buffer.byte_length buf < workspace_bytes then
          Error "provided workspace buffer is too small for schedule"
        else Ok (Some buf)
    | None ->
        if workspace_bytes = 0 then Ok None
        else Buffer.create ~runtime ~bytes:workspace_bytes |> Result.map Option.some
  in
  execution_batch.resources <-
    List.map snd inputs @ Option.to_list workspace @ execution_batch.resources;
  let dispatch_output ?name = dispatch_output ?name ~batch in
  let rec run state = function
    | [] ->
        execution_batch.resources <-
          state.resources @ execution_batch.resources;
        Ok
          {
            Execution.outputs = List.rev state.outputs_rev;
            kernels = List.rev state.kernels_rev;
            workspace_bytes;
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
        let dispatched_many result =
          let* state, kernels = result in
          continue
            { state with
              kernels_rev = List.rev_append kernels state.kernels_rev }
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
        | ( Ir.Op.Primitive (Ir.Primitive.Triangular_recurrence config),
            [ input ],
            Some output ) ->
            let dimensions =
              Tensor_shape.dimensions (Ir.Value.logical_shape input)
            in
            let reversed = List.rev dimensions in
            let width = List.hd reversed in
            let rows = List.nth reversed 1 in
            if width > 64 || rows <> width then
              Error "unsupported Metal triangular recurrence shape"
            else
              let outer =
                Tensor_shape.numel (Ir.Value.logical_shape input)
                / (width * width)
              in
              let* input_buffer = find_value state input in
              let* parameters =
                Parameters.u32s
                  [ outer; width;
                    Ir.Triangular_recurrence.start config;
                    Ir.Triangular_recurrence.stop config ]
              in
              dispatched
                (dispatch_output ~name:"llmopt_triangular_recurrence_f32"
                   runtime state output
                   ~operation:Kernel_abi.Operation.Triangular_recurrence
                   ~input_dtype:Ir.Dtype.Float32 ~buffers:[ input_buffer ]
                   ~parameters ~grid:(outer * 64, 1, 1))
        | ( Ir.Op.Primitive Ir.Primitive.Gated_delta,
            [ query; key; value; gate; beta ],
            Some output ) ->
            let shape value =
              Tensor_shape.dimensions (Ir.Value.logical_shape value)
            in
            let* batch_size, heads, tokens, width, name, input_dtype =
              match Ir.Value.dtype query, shape query, shape output with
              | Ir.Dtype.Float32,
                [ batch_size; heads; tokens; width ],
                [ output_batch; output_tokens; output_heads; output_width ]
                when width > 0 && width mod 32 = 0
                     && output_batch = batch_size && output_tokens = tokens
                     && output_heads = heads && output_width = width
                     && Ir.Value.dtype key = Ir.Dtype.Float32
                     && Ir.Value.dtype value = Ir.Dtype.Float32
                     && Ir.Value.dtype gate = Ir.Dtype.Float32
                     && Ir.Value.dtype beta = Ir.Dtype.Float32
                     && shape key = shape query && shape value = shape query
                     && shape gate = [ batch_size; heads; tokens ]
                     && shape beta = [ batch_size; heads; tokens ] ->
                  Ok
                    ( batch_size, heads, tokens, width,
                      Printf.sprintf "llmopt_gated_delta_f32_d%d" width,
                      Ir.Dtype.Float32 )
              | Ir.Dtype.Float16,
                [ batch_size; tokens; heads; width ],
                [ output_batch; output_tokens; output_heads; output_width ]
                when width > 0 && width mod 32 = 0
                     && output_batch = batch_size && output_tokens = tokens
                     && output_heads = heads && output_width = width
                     && Ir.Value.dtype key = Ir.Dtype.Float16
                     && Ir.Value.dtype value = Ir.Dtype.Float16
                     && Ir.Value.dtype gate = Ir.Dtype.Float32
                     && Ir.Value.dtype beta = Ir.Dtype.Float16
                     && shape key = shape query && shape value = shape query
                     && shape gate = [ batch_size; tokens; heads ]
                     && shape beta = [ batch_size; tokens; heads ] ->
                  Ok
                    ( batch_size, heads, tokens, width,
                      Printf.sprintf "llmopt_gated_delta_tm_f16_d%d" width,
                      Ir.Dtype.Float16 )
              | _ ->
                  Error "unsupported Metal gated-delta layout or dtype contract"
            in
            let* buffers = find_values state [ query; key; value; gate; beta ] in
            let* parameters =
              Parameters.u32s [ batch_size; heads; tokens; width ]
            in
            let* grid_x = gated_delta_grid (batch_size * heads * width) in
            dispatched
              (dispatch_output
                 ~name
                 runtime state output
                 ~operation:Kernel_abi.Operation.Gated_delta
                 ~input_dtype ~buffers ~parameters
                 ~grid:(grid_x, 1, 1))
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
        | Ir.Op.Gelu, [ input ], Some output
          when Ir.Value.dtype input = Ir.Dtype.Float16
               && Ir.Value.dtype output = Ir.Dtype.Float16 ->
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* input_buffer = find_value state input in
            let* parameters = Parameters.u32s [ count ] in
            dispatched
              (dispatch_output ~name:"llmopt_gelu_f16" runtime state output
                 ~operation:Kernel_abi.Operation.Pointwise
                 ~input_dtype:Ir.Dtype.Float16 ~buffers:[ input_buffer ]
                 ~parameters ~grid:(count, 1, 1))
        | ( Ir.Op.Primitive (Ir.Primitive.Pointwise operation),
            _,
            Some output ) ->
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* name, input_dtype = pointwise_kernel operation output in
            (match operation with
            | Ir.Pointwise.Unary (Ir.Pointwise.Pow exponent, input) ->
                let* input_buffer = find_value state input in
                let* parameters =
                  Parameters.pointwise
                    ~output_shape:(Ir.Value.logical_shape output)
                    ~left_shape:None ~right_shape:None
                    ~left_scalar:(Some exponent) ~right_scalar:None
                in
                dispatched
                  (dispatch_output ~name runtime state output
                     ~operation:Kernel_abi.Operation.Pointwise ~input_dtype
                     ~buffers:[ input_buffer ] ~parameters ~grid:(count, 1, 1))
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
            | Ir.Dtype.Float16, Ir.Dtype.Bfloat16, Ir.Dtype.Float16, None ->
                let* parameters = Parameters.u32s [ m; n; k ] in
                let* grid_x = linear_f16_grid n in
                dispatched
                  (dispatch_output ~name:"llmopt_linear_f16_bf16" runtime state
                     output ~operation:Kernel_abi.Operation.Linear
                     ~input_dtype:Ir.Dtype.Float16 ~buffers ~parameters
                     ~grid:(grid_x, 1, 1))
            | Ir.Dtype.Float16, Ir.Dtype.Float32, Ir.Dtype.Float16, None ->
                let* parameters = Parameters.u32s [ m; n; k ] in
                let* grid_x = linear_f16_grid n in
                dispatched
                  (dispatch_output ~name:"llmopt_linear_f16_f32" runtime state
                     output ~operation:Kernel_abi.Operation.Linear
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
            | ( Ir.Dtype.Float16,
                Ir.Dtype.Quant quant,
                Ir.Dtype.Float16,
                bias_value ) ->
                let* () =
                  match bias_value with
                  | None -> Ok ()
                  | Some bias
                    when Ir.Value.dtype bias = Ir.Dtype.Float16 ->
                      Ok ()
                  | Some _ -> Error "quantized Metal linear bias must be float16"
                in
                let* entry = select_quant_linear_kernel runtime ~m quant in
                let name = Kernel_abi.Entry.name entry in
                let* input_buffer, weight_buffer, bias_buffer =
                  match buffers with
                  | [ input; weight ] -> Ok (input, weight, weight)
                  | [ input; weight; bias ] -> Ok (input, weight, bias)
                  | _ -> Error "quantized Metal linear buffer binding is inconsistent"
                in
                let* parameters =
                  Parameters.u32s [ m; n; k; if Option.is_some bias_value then 1 else 0 ]
                in
                let columns =
                  if String.ends_with ~suffix:"_m2_x2_l32" name then
                    m * ((n + 1) / 2)
                  else if String.ends_with ~suffix:"_m2_x1_l32" name then
                    m * n
                  else if String.ends_with ~suffix:"_m2_x4" name then (n + 3) / 4
                  else if String.ends_with ~suffix:"_m2" name then n
                  else m * n
                in
                let* grid_x =
                  if String.ends_with ~suffix:"_m2_x2_l32" name
                     || String.ends_with ~suffix:"_m2_x1_l32" name
                     || String.ends_with ~suffix:"_m2_x4" name
                  then
                    short_row_quant_grid columns
                  else linear_f16_grid columns
                in
                dispatched
                  (dispatch_output ~name runtime state output
                     ~operation:Kernel_abi.Operation.Linear
                     ~input_dtype:(Ir.Dtype.Quant quant)
                     ~buffers:[ input_buffer; weight_buffer; bias_buffer ]
                     ~parameters ~grid:(grid_x, 1, 1))
            | input_dtype, weight_dtype, output_dtype, _ ->
                Error
                  (Printf.sprintf "unsupported Metal linear kernel: %s + %s -> %s"
                     (Ir.Dtype.to_string input_dtype)
                     (Ir.Dtype.to_string weight_dtype)
                     (Ir.Dtype.to_string output_dtype)))
        | ( Ir.Op.Gated_linear { m; n; k; activation },
            [ input; gate_weight; up_weight ],
            Some output ) ->
            let* () = validate_linear_shapes ~m ~n ~k input gate_weight output in
            let* () = validate_linear_shapes ~m ~n ~k input up_weight output in
            let* quant =
              match Ir.Value.dtype gate_weight, Ir.Value.dtype up_weight with
              | Ir.Dtype.Quant quant, up_dtype
                when up_dtype = Ir.Dtype.Quant quant
                     && (quant = Ir.Dtype.Q4_K
                        || (m = 2
                           && (quant = Ir.Dtype.Q5_K
                              || quant = Ir.Dtype.IQ4_XS))) ->
                  Ok quant
              | gate_dtype, up_dtype ->
                  Error
                    (Printf.sprintf
                       "unsupported gated Linear weights: gate=%s up=%s"
                       (Ir.Dtype.to_string gate_dtype)
                       (Ir.Dtype.to_string up_dtype))
            in
            let* buffers = find_values state [ input; gate_weight; up_weight ] in
            let activation_tag =
              match activation with
              | Ir.Gated_activation.Silu -> 0
              | Ir.Gated_activation.Gelu -> 1
              | Ir.Gated_activation.Sigmoid -> 2
            in
            let* parameters = Parameters.u32s [ m; n; k; activation_tag ] in
            let name =
              match quant, m with
              | Ir.Dtype.Q4_K, 2 ->
                  "llmopt_q4_k_gated_linear_f16_m2_x1_l32"
              | Ir.Dtype.Q4_K, _ -> "llmopt_q4_k_gated_linear_f16"
              | Ir.Dtype.Q5_K, 2 ->
                  "llmopt_q5_k_gated_linear_f16_m2_x1_l32"
              | Ir.Dtype.IQ4_XS, 2 ->
                  "llmopt_iq4_xs_gated_linear_f16_m2"
              | _ -> assert false
            in
            let* grid_x =
              match quant, m with
              | (Ir.Dtype.Q4_K | Ir.Dtype.Q5_K), 2 ->
                  short_row_quant_grid (m * n)
              | Ir.Dtype.IQ4_XS, 2 -> linear_f16_grid n
              | Ir.Dtype.Q4_K, _ -> linear_f16_grid (m * n)
              | _ -> assert false
            in
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Gated_linear
                 ~input_dtype:(Ir.Dtype.Quant quant) ~buffers ~parameters
                 ~grid:(grid_x, 1, 1))
        | ( Ir.Op.Rms_norm_add { epsilon },
            [ input; weight; residual ],
            Some output ) ->
            let input_shape = Ir.Value.logical_shape input in
            let* width =
              match List.rev (Tensor_shape.dimensions input_shape) with
              | width :: _ when width > 0 -> Ok width
              | _ -> Error "RMSNorm-add requires a non-empty final dimension"
            in
            let rows = Tensor_shape.numel input_shape / width in
            let wide =
              use_wide_rows ~rows ~width
              && Ir.Value.dtype input = Ir.Dtype.Float16
              && Ir.Value.dtype weight = Ir.Dtype.Float32
            in
            let* buffers = find_values state [ input; weight; residual ] in
            let* parameters = Parameters.rms_norm ~rows ~width ~epsilon in
            let name =
              if wide then "llmopt_rms_norm_add_f16_wf32_wide"
              else "llmopt_rms_norm_add_f16_wf32_simd"
            in
            let* entry =
              kernel_entry ~name runtime
                ~operation:Kernel_abi.Operation.Rms_norm
                ~input_dtype:(Ir.Value.dtype input)
                ~output_dtype:(Ir.Value.dtype output)
            in
            let* grid_x =
              if wide then wide_rows_grid rows else simd_rows_grid rows
            in
            let* output_buffer = workspace_buffer state output in
            let* kernel =
              dispatch ~batch runtime entry ~buffers:(buffers @ [ output_buffer ])
                ~parameters ~grid:(grid_x, 1, 1)
            in
            dispatched (Ok (bind_value state output output_buffer, kernel))
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
            let weight_dtype = Ir.Value.dtype weight in
            let wide =
              use_wide_rows ~rows ~width
              && input_dtype = Ir.Dtype.Float16
              && weight_dtype = Ir.Dtype.Float32
            in
            let* entry, simd =
              select_rms_norm_kernel runtime ~wide input_dtype weight_dtype
                (Ir.Value.dtype output)
            in
            let* grid_x =
              if wide then wide_rows_grid rows
              else if simd then simd_rows_grid rows
              else Ok rows
            in
            let* output_buffer = workspace_buffer state output in
            let* kernel =
              dispatch ~batch runtime entry ~buffers:(buffers @ [ output_buffer ])
                ~parameters ~grid:(grid_x, 1, 1)
            in
            dispatched (Ok (bind_value state output output_buffer, kernel))
        | ( Ir.Op.Primitive (Ir.Primitive.L2_norm { epsilon }),
            [ input ],
            Some output ) ->
            let input_shape = Ir.Value.logical_shape input in
            let* width =
              match List.rev (Tensor_shape.dimensions input_shape) with
              | width :: _ when width > 0 -> Ok width
              | _ -> Error "L2 normalization requires a non-empty final dimension"
            in
            let rows = Tensor_shape.numel input_shape / width in
            let* input_buffer = find_value state input in
            let* parameters = Parameters.rms_norm ~rows ~width ~epsilon in
            let* grid_x = simd_rows_grid rows in
            dispatched
              (dispatch_output ~name:"llmopt_l2_norm_f16_simd" runtime state
                 output ~operation:Kernel_abi.Operation.Rms_norm
                 ~input_dtype:(Ir.Value.dtype input) ~buffers:[ input_buffer ]
                 ~parameters ~grid:(grid_x, 1, 1))
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
              rotary_trig_batches ~batches ~tokens ~width cosine sine
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
              || (Ir.Value.dtype weight <> Ir.Dtype.Float16
                 && Ir.Value.dtype weight <> Ir.Dtype.Float32)
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
              let* kernel_name =
                match Ir.Value.dtype weight with
                | Ir.Dtype.Float16 -> Ok "llmopt_rms_rope_f16_simd_h64"
                | Ir.Dtype.Float32 -> Ok "llmopt_rms_rope_f16_wf32_simd"
                | dtype ->
                    Error
                      ("unsupported Metal RMSNorm-RoPE weight dtype: "
                      ^ Ir.Dtype.to_string dtype)
              in
              let* entry =
                kernel_entry ~name:kernel_name runtime
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
        | ( Ir.Op.Rms_rope_qk
              {
                q_heads;
                k_heads;
                width;
                half_dimension;
                epsilon;
                extra_outputs = [ k_output ];
              },
            [ q_input; q_weight; k_input; k_weight; cosine; sine ],
            Some q_output ) ->
            let* batches, tokens, _, _ =
              match Tensor_shape.dimensions (Ir.Value.logical_shape q_input) with
              | [ batches; tokens; heads; w ]
                when batches > 0 && tokens > 0 && heads = q_heads && w = width ->
                  Ok (batches, tokens, heads, w)
              | _ -> Error "Metal RMSNorm-RoPE-QK Q input must have rank four"
            in
            let* trig_batches =
              rotary_trig_batches ~batches ~tokens ~width cosine sine
            in
            if
              Tensor_shape.dimensions (Ir.Value.logical_shape k_input)
              <> [ batches; tokens; k_heads; width ]
              || Tensor_shape.dimensions (Ir.Value.logical_shape q_weight)
                 <> [ width ]
              || Tensor_shape.dimensions (Ir.Value.logical_shape k_weight)
                 <> [ width ]
              || Tensor_shape.dimensions (Ir.Value.logical_shape q_output)
                 <> [ batches; q_heads; tokens; width ]
              || Tensor_shape.dimensions (Ir.Value.logical_shape k_output)
                 <> [ batches; k_heads; tokens; width ]
              || Ir.Value.dtype q_input <> Ir.Dtype.Float16
              || Ir.Value.dtype k_input <> Ir.Dtype.Float16
              || Ir.Value.dtype q_weight <> Ir.Value.dtype k_weight
              || (Ir.Value.dtype q_weight <> Ir.Dtype.Float16
                 && Ir.Value.dtype q_weight <> Ir.Dtype.Float32)
              || Ir.Value.dtype cosine <> Ir.Dtype.Float16
              || Ir.Value.dtype sine <> Ir.Dtype.Float16
              || Ir.Value.dtype q_output <> Ir.Dtype.Float16
              || Ir.Value.dtype k_output <> Ir.Dtype.Float16
            then Error "Metal RMSNorm-RoPE-QK tensor metadata is inconsistent"
            else
            let* buffers =
              find_values state [ q_input; q_weight; k_input; k_weight; cosine; sine ]
            in
            let* parameters =
              Parameters.rms_rope_qk ~batches ~tokens ~q_heads ~k_heads ~width
                ~half_dimension ~trig_batches ~epsilon
            in
            let* kernel_name =
              match Ir.Value.dtype q_weight with
              | Ir.Dtype.Float16 -> Ok "llmopt_rms_rope_qk_f16_simd_h64"
              | Ir.Dtype.Float32 -> Ok "llmopt_rms_rope_qk_f16_wf32_simd"
              | dtype ->
                  Error
                    ("unsupported Metal RMSNorm-RoPE-QK weight dtype: "
                    ^ Ir.Dtype.to_string dtype)
            in
            let* entry =
              kernel_entry ~name:kernel_name runtime
                ~operation:Kernel_abi.Operation.Rms_rope_qk
                ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
            in
            let* grid_x = simd_rows_grid (batches * (q_heads + k_heads) * tokens) in
            let* q_out_buf = workspace_buffer state q_output in
            let* k_out_buf = workspace_buffer state k_output in
            let* kernel =
              dispatch ~batch runtime entry
                ~buffers:(buffers @ [ q_out_buf; k_out_buf ])
                ~parameters
                ~grid:(grid_x, 1, 1)
            in
            let state = bind_value state q_output q_out_buf in
            let state = bind_value state k_output k_out_buf in
            dispatched (Ok (state, kernel))
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
            (in_proj :: conv_weight :: conv_state_out :: rest),
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
            let* (has_initial_state, extra_buffers) =
              match rest with
              | [ conv_state_in ] ->
                  let* state_in_buf = find_value state conv_state_in in
                  Ok (1, [ state_in_buf ])
              | [] -> Ok (0, [])
              | _ -> Error "Metal ShortConv prefill has too many inputs"
            in
            let* parameters = Parameters.u32s [ tokens; channels; has_initial_state ] in
            let* entry =
              kernel_entry
                ~name:
                  (if has_initial_state = 1 then
                     "llmopt_short_conv_prefill_init_f16"
                   else "llmopt_short_conv_prefill_f16")
                runtime
                ~operation:Kernel_abi.Operation.Short_conv_prefill
                ~input_dtype:Ir.Dtype.Float16 ~output_dtype:Ir.Dtype.Float16
            in
            let* kernel =
              dispatch ~batch runtime entry
                ~buffers:([ in_proj_buf; weight_buf; output_buf; state_out_buf ] @ extra_buffers)
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
            let* name =
              match Ir.Value.dtype weight with
              | Ir.Dtype.Float16 -> Ok "llmopt_embedding_f16"
              | Ir.Dtype.Quant Q8_0 -> Ok "llmopt_embedding_q8_0"
              | Ir.Dtype.Quant Q4_K -> Ok "llmopt_embedding_q4_k"
              | Ir.Dtype.Quant Q5_K -> Ok "llmopt_embedding_q5_k"
              | Ir.Dtype.Quant Q6_K -> Ok "llmopt_embedding_q6_k"
              | Ir.Dtype.Quant Q5_0 -> Ok "llmopt_embedding_q5_0"
              | Ir.Dtype.Quant Q4_0 -> Ok "llmopt_embedding_q4_0"
              | Ir.Dtype.Quant IQ4_XS ->
                  Error "IQ4_XS must be transcoded before Metal embedding execution"
              | dtype ->
                  Error
                    ("unsupported Metal embedding weight dtype: "
                    ^ Ir.Dtype.to_string dtype)
            in
            dispatched
              (dispatch_output ~name runtime state output
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
        | ( Ir.Op.Primitive (Ir.Primitive.Pad_right_zero { axis }),
            [ input ], Some output ) ->
            let input_dimensions =
              Tensor_shape.dimensions (Ir.Value.logical_shape input)
            in
            let output_dimensions =
              Tensor_shape.dimensions (Ir.Value.logical_shape output)
            in
            let input_axis = List.nth input_dimensions axis in
            let output_axis = List.nth output_dimensions axis in
            let inner =
              input_dimensions |> List.filteri (fun index _ -> index > axis)
              |> List.fold_left ( * ) 1
            in
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* buffers = find_values state [ input ] in
            let* parameters =
              Parameters.u32s [ count; input_axis; output_axis; inner ]
            in
            dispatched
              (dispatch_output ~name:"llmopt_pad_right_zero_f32" runtime state
                 output ~operation:Kernel_abi.Operation.Movement
                 ~input_dtype:(Ir.Value.dtype input) ~buffers ~parameters
                 ~grid:(count, 1, 1))
        | ( Ir.Op.Primitive
              (Ir.Primitive.Triangular { upper; diagonal }),
            [ input ], Some output ) ->
            let dimensions =
              Tensor_shape.dimensions (Ir.Value.logical_shape output)
            in
            let rows, cols =
              match List.rev dimensions with
              | cols :: rows :: _ -> rows, cols
              | _ -> 0, 0
            in
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* buffers = find_values state [ input ] in
            let* parameters =
              Parameters.u32s
                [ count; rows; cols; if upper then 1 else 0; diagonal ]
            in
            let name =
              match Ir.Value.dtype input with
              | Ir.Dtype.Bool -> "llmopt_triangular_bool"
              | Ir.Dtype.Float32 -> "llmopt_triangular_f32"
              | _ -> ""
            in
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Pointwise
                 ~input_dtype:(Ir.Value.dtype input) ~buffers ~parameters
                 ~grid:(count, 1, 1))
        | ( Ir.Op.Primitive (Ir.Primitive.Masked_fill scalar),
            [ input; mask ], Some output ) ->
            let count = Tensor_shape.numel (Ir.Value.logical_shape output) in
            let* buffers = find_values state [ input; mask ] in
            let* parameters =
              Parameters.masked_fill
                ~output_shape:(Ir.Value.logical_shape output)
                ~input_shape:(Ir.Value.logical_shape input)
                ~mask_shape:(Ir.Value.logical_shape mask) scalar
            in
            dispatched
              (dispatch_output ~name:"llmopt_masked_fill_f32" runtime state
                 output ~operation:Kernel_abi.Operation.Pointwise
                 ~input_dtype:(Ir.Value.dtype input) ~buffers ~parameters
                 ~grid:(count, 1, 1))
        | Ir.Op.Primitive Ir.Primitive.Eye, [], Some output ->
            let rows, cols =
              match Tensor_shape.dimensions (Ir.Value.logical_shape output) with
              | [ rows; cols ] -> rows, cols
              | _ -> 0, 0
            in
            let count = rows * cols in
            let* parameters = Parameters.u32s [ count; rows; cols ] in
            dispatched
              (dispatch_output runtime state output
                 ~operation:Kernel_abi.Operation.Eye
                 ~input_dtype:Ir.Dtype.Float32 ~buffers:[] ~parameters
                 ~grid:(count, 1, 1))
        | ( Ir.Op.Primitive Ir.Primitive.Batched_matmul,
            [ lhs; rhs ], Some output ) ->
            let output_dimensions =
              Tensor_shape.dimensions (Ir.Value.logical_shape output)
            in
            let* batch_count, m, n =
              match List.rev output_dimensions with
              | n :: m :: reversed_batch ->
                  Ok
                    ( List.fold_left ( * ) 1 reversed_batch,
                      m,
                      n )
              | _ -> Error "batched matmul output must have rank at least two"
            in
            let* buffers = find_values state [ lhs; rhs ] in
            let* parameters =
              Parameters.batched_matmul
                ~lhs_shape:(Ir.Value.logical_shape lhs)
                ~rhs_shape:(Ir.Value.logical_shape rhs)
                ~output_shape:(Ir.Value.logical_shape output)
            in
            let* grid_x = round_up n 16 in
            let* grid_y = round_up m 16 in
            let lhs_dimensions =
              Tensor_shape.dimensions (Ir.Value.logical_shape lhs)
            in
            let rhs_dimensions =
              Tensor_shape.dimensions (Ir.Value.logical_shape rhs)
            in
            let* k =
              match List.rev lhs_dimensions, List.rev rhs_dimensions with
              | k :: _ :: _, _n :: rhs_k :: _ when k = rhs_k -> Ok k
              | _ -> Error "batched matmul input dimensions are inconsistent"
            in
            let simd8 = m mod 8 = 0 && n mod 8 = 0 && k mod 8 = 0 in
            let name =
              if simd8 then "llmopt_batched_matmul_f32_simd8"
              else "llmopt_batched_matmul_f32_tiled"
            in
            let* grid =
              if simd8 then
                let* columns = round_up n 8 in
                let* rows = round_up m 8 in
                Ok ((columns / 8) * 32, rows / 8, batch_count)
              else Ok (grid_x, grid_y, batch_count)
            in
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Batched_matmul
                 ~input_dtype:Ir.Dtype.Float32 ~buffers ~parameters
                 ~grid)
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
            let name =
              match Ir.Value.dtype weight with
              | Ir.Dtype.Float16 -> "llmopt_short_conv_f16"
              | Ir.Dtype.Float32 -> "llmopt_short_conv_f16_f32"
              | _ -> ""
            in
            dispatched
              (dispatch_output ~name runtime state output
                 ~operation:Kernel_abi.Operation.Short_conv
                 ~input_dtype:(Ir.Value.dtype weight) ~buffers ~parameters
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
            let* kv_heads, key_length =
              match Tensor_shape.dimensions (Ir.Value.logical_shape key) with
              | [ _batches; kv_heads; key_length; _dimension ] ->
                  Ok (kv_heads, key_length)
              | _ -> Error "attention key must have rank four"
            in
            let* mask_batches, mask_heads =
              match Tensor_shape.dimensions (Ir.Value.logical_shape mask) with
              | [ mask_batches; mask_heads; _query_length; _key_length ] ->
                  Ok (mask_batches, mask_heads)
              | _ -> Error "attention mask must have rank four"
            in
            let token_first_output =
              match Tensor_shape.dimensions (Ir.Value.logical_shape output) with
              | [ b; q; hidden ]
                when b = batches && q = query_length && hidden = heads * head_dimension ->
                  1
              | [ b; q; h; d ]
                when b = batches && q = query_length && h = heads && d = head_dimension ->
                  1
              | _ -> 0
            in
            let* buffers = find_values state [ query; key; value; mask ] in
            let past_length = max 0 (key_length - query_length) in
            let* parameters =
              Parameters.attention ~batches ~heads ~query_length ~key_length
                ~head_dimension ~mask_batches ~mask_heads
                ~causal:(Ir.Attention.causal config)
                ~scale:(Ir.Attention.scale config)
                ~kv_heads ~token_first_output ~past_length ()
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
            let* batches, query_heads, query_length, head_dimension =
              match Tensor_shape.dimensions (Ir.Value.logical_shape query) with
              | [ batches; query_heads; query_length; head_dimension ] ->
                  Ok (batches, query_heads, query_length, head_dimension)
              | _ -> Error "paged Q8 attention query must have shape [b,h,q,d]"
            in
            let* kv_heads =
              match Tensor_shape.dimensions (Ir.Value.logical_shape current_key) with
              | [ _batches; kv_heads; key_tokens; _head_dimension ]
                when key_tokens = query_length -> Ok kv_heads
              | _ ->
                  Error "paged Q8 attention current key must match query tokens"
            in
            let* past_length =
              match Tensor_shape.dimensions (Ir.Value.logical_shape slots) with
              | [ past_length ] -> Ok past_length
              | _ -> Error "paged Q8 attention slots must have rank one"
            in
            let* mask_batches, mask_heads =
              match Tensor_shape.dimensions (Ir.Value.logical_shape mask) with
              | [ mask_batches; mask_heads; mask_query; _key_length ]
                when mask_query = query_length || mask_query = 1 ->
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
                ~query_length ()
            in
            let* entry =
              kernel_entry ~name:"llmopt_attention_q8_paged_simd_h64" runtime
                ~operation:Kernel_abi.Operation.Attention
                ~input_dtype:Ir.Dtype.Float16
                ~output_dtype:(Ir.Value.dtype output)
            in
            let* grid_x = simd_rows_grid (batches * query_heads * query_length) in
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
            if m = 1 then
              let* grid_x = linear_f16_grid (m * n) in
              dispatched
                (dispatch_output ~name:"llmopt_w4a16_linear_f16_g64" runtime
                   state output ~operation:Kernel_abi.Operation.W4a16_linear
                   ~input_dtype:Ir.Dtype.Float16
                   ~buffers:[ input; weight; scale; bias_buffer ] ~parameters
                   ~grid:(grid_x, 1, 1))
            else
              let m_blocks = (m + 3) / 4 in
              let* grid_x = linear_f16_grid (m_blocks * n) in
              dispatched
                (dispatch_output ~name:"llmopt_w4a16_linear_f16_g64_m4" runtime
                   state output ~operation:Kernel_abi.Operation.W4a16_linear
                   ~input_dtype:Ir.Dtype.Float16
                   ~buffers:[ input; weight; scale; bias_buffer ] ~parameters
                   ~grid:(grid_x, 1, 1))
        | ( Ir.Op.W4a16_swiglu_ffn { m; n; k; epsilon },
            values,
            Some output ) ->
            dispatched_many
              (dispatch_w4a16_swiglu_ffn_command execution_batch batch runtime state ~m ~n ~k
                 ~epsilon values output)
        | ( Ir.Op.W4a16_qkv_linear { m; k; n_q; n_k; n_v; extra_outputs },
            values,
            Some output ) ->
            dispatched_many
              (dispatch_w4a16_qkv_linear_command batch runtime state ~m ~k ~n_q
                 ~n_k ~n_v ~extra_outputs values output)
        | ( Ir.Op.W4a16_lm_head_argmax { m; n; k; epsilon; extra_outputs },
            values,
            Some output ) ->
            dispatched
              (dispatch_w4a16_lm_head_argmax_command execution_batch batch runtime state ~m ~n ~k
                 ~epsilon ~extra_outputs values output)
        | Ir.Op.Output { name }, [ input ], None ->
            let* buffer = find_value state input in
            continue
              { state with outputs_rev = (name, buffer) :: state.outputs_rev }
        | Ir.Op.Barrier_create _, [], None
        | Ir.Op.Barrier_arrive _, [], None
        | Ir.Op.Barrier_wait _, [], None ->
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
        resources = [];
      }
      (Serving_schedule.commands schedule)
  in
  execution_batch.schedules <- execution_batch.schedules + 1;
  Ok execution

let execute_schedule ?workspace ?memory_plan runtime ~schedule ~inputs =
  with_execution_batch runtime (fun batch ->
      encode_schedule ?workspace ?memory_plan batch ~schedule ~inputs)

let execute_decode_step ?workspace ?memory_plan runtime ~cache ~schedule ~inputs
    ~cache_pack =
  with_execution_batch runtime (fun batch ->
      let* execution =
        encode_schedule ?workspace ?memory_plan batch ~schedule ~inputs
      in
      let cache_batch =
        { Cache.cache; commands = batch.commands; resources = []; dispatches = 0 }
      in
      let* () = cache_pack execution cache_batch in
      batch.resources <- cache_batch.resources @ batch.resources;
      Ok execution)

let precompile_decode_batch ?workspace ?memory_plan ?cache ?cache_pack runtime
    ~schedule ~inputs =
  let* commands = Batch.create runtime in
  let batch =
    {
      Execution_batch.runtime;
      commands;
      resources = [];
      schedules = 0;
      swiglu_normalized = None;
      swiglu_product = None;
      candidates_buffer = None;
    }
  in
  let* execution =
    encode_schedule ?workspace ?memory_plan batch ~schedule ~inputs
  in
  let* () =
    match cache, cache_pack with
    | Some cache, Some pack_fn ->
        let cache_batch =
          { Cache.cache; commands = batch.commands; resources = []; dispatches = 0 }
        in
        let* () = pack_fn execution cache_batch in
        batch.resources <- cache_batch.resources @ batch.resources;
        Ok ()
    | _ -> Ok ()
  in
  let items_rev = List.rev batch.commands.pending in
  let dispatches =
    Array.of_list
      (List.map
         (fun (_, it) ->
            (it.Batch.name, it.Batch.buffers, it.Batch.parameters,
             it.Batch.grid_x, it.Batch.grid_y, it.Batch.grid_z,
             it.Batch.group_x, it.Batch.group_y, it.Batch.group_z))
         items_rev)
  in
  ignore (Batch.abort commands);
  Ok (dispatches, execution)

let execute runtime ~inputs =
  execute_schedule runtime
    ~schedule:(Serving_package.schedule runtime.package) ~inputs
