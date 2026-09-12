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

external preload_pipeline_stub : library_handle -> string -> unit
  = "caml_llmopt_metal_preload_pipeline"

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

external buffer_argmax_f16_stub : buffer_handle -> int -> int
  = "caml_llmopt_metal_buffer_argmax_f16"

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
  (string * buffer_handle list * bytes * int * int * int * int * int * int * bool)
  array ->
  buffer_handle ->
  buffer_handle ->
  unit = "caml_llmopt_ring_start_worker"

external ring_destroy_stub : ring_handle -> unit = "caml_llmopt_ring_destroy"

type prebaked_handle

external prebaked_create_stub :
  library_handle ->
  (string * buffer_handle list * bytes * int * int * int * int * int * int * bool)
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

module Decode_uniforms = struct
  type t = { past_tokens : int }

  let create ~past_tokens =
    if past_tokens <= 0 then Error "decode past_tokens must be positive"
    else Ok { past_tokens }

  let past_tokens uniforms = uniforms.past_tokens
end

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
    barrier_before : bool;
  }

  type t = {
    handle : batch_handle;
    context : context_handle;
    mutable pending : (library_handle * dispatch_item) list;
    mutable next_dispatch_barrier : bool;
  }

  let create (runtime : runtime) =
    protect (fun () ->
        {
          handle = begin_batch_stub runtime.library;
          context = runtime.context;
          pending = [];
          next_dispatch_barrier = false;
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
           barrier_before = batch.next_dispatch_barrier;
         })
        :: batch.pending;
      batch.next_dispatch_barrier <- false;
      Ok ()
    )

  let mark_barrier batch = batch.next_dispatch_barrier <- true

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
    batch.next_dispatch_barrier <- false;
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

let rec preload_declared_pipelines library = function
  | [] -> Ok ()
  | entry :: rest ->
      let name = Kernel_abi.Entry.name entry in
      let* () = protect (fun () -> preload_pipeline_stub library name) in
      preload_declared_pipelines library rest

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
          let* () =
            preload_declared_pipelines library
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
  let argmax_f16 buffer ~vocabulary =
    protect (fun () -> buffer_argmax_f16_stub buffer vocabulary)

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

  let default_pipeline_depth = 1

  let pipeline_depth () =
    match Sys.getenv_opt "LLMOPT_QUEUE_DEPTH" with
    | Some s -> (
        match int_of_string_opt s with
        | Some d when d >= 1 -> d
        | _ -> default_pipeline_depth)
    | None -> default_pipeline_depth

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
