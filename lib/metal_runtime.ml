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
