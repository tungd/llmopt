let ( let* ) = Result.bind

module Stage = struct
  type t = Compiled_graph | Serving

  let to_string = function
    | Compiled_graph -> "compiled-graph"
    | Serving -> "serving"
end

module Artifact = struct
  type t = string

  let create path =
    let path = String.trim path in
    let components = String.split_on_char '/' path in
    if path = "" then Error "serving-package artifact path cannot be empty"
    else if not (Filename.is_relative path) then
      Error ("serving-package artifact path must be relative: " ^ path)
    else if String.contains path '\000' then
      Error "serving-package artifact path contains a NUL byte"
    else if
      List.exists
        (fun component -> component = "" || component = "." || component = "..")
        components
    then Error ("serving-package artifact path is not canonical: " ^ path)
    else Ok path

  let path artifact = artifact
end

module Files = struct
  type t = { metal_library : Artifact.t }

  let create ~metal_library = { metal_library }
  let metal_library files = files.metal_library
end

module Tensor_store = struct
  type t = Weights of { file : Artifact.t }

  let weights ~file = Weights { file }
  let file (Weights { file }) = file
end

module Cache = struct
  type t = {
    page_size : int;
    default_kv : Kv_cache.Format.t;
    supported_kv : Kv_cache.Format.t list;
  }

  let create ~page_size ~default_kv ~supported_kv =
    let unique = List.sort_uniq Stdlib.compare supported_kv in
    if page_size <= 0 then Error "serving-package radix page_size must be positive"
    else if supported_kv = [] then
      Error "serving-package must support at least one KV format"
    else if List.length unique <> List.length supported_kv then
      Error "serving-package KV formats must be unique"
    else if not (List.mem default_kv supported_kv) then
      Error "serving-package default KV format must be supported"
    else Ok { page_size; default_kv; supported_kv }

  let default =
    match
      create ~page_size:1 ~default_kv:Kv_cache.Format.default
        ~supported_kv:[ Kv_cache.Format.default; Kv_cache.Format.f16 ]
    with
    | Ok cache -> cache
    | Error message -> invalid_arg message

  let page_size cache = cache.page_size
  let default_kv cache = cache.default_kv
  let supported_kv cache = cache.supported_kv
end

type t = {
  stage : Stage.t;
  model : string option;
  files : Files.t;
  kernels : Kernel_abi.Entry.t list;
  schedule : Serving_schedule.t;
  tensor_store : Tensor_store.t option;
  cache : Cache.t;
}

let create ~stage ?model ~files ~kernels ~schedule ~tensor_store ~cache () =
  let kernel_names = List.map Kernel_abi.Entry.name kernels in
  let tensor_inputs = Serving_schedule.tensor_inputs schedule in
  if Option.exists (fun value -> String.trim value = "") model then
    Error "serving-package model identifier cannot be empty"
  else if kernels = [] then Error "serving-package must declare at least one kernel"
  else if
    List.length kernel_names
    <> List.length (List.sort_uniq String.compare kernel_names)
  then Error "serving-package kernel entry-point names must be unique"
  else if stage = Stage.Compiled_graph && Option.is_some tensor_store then
    Error "compiled-graph package cannot declare a tensor store"
  else if stage = Stage.Serving && Option.is_none tensor_store then
    Error "serving-stage package must declare a tensor store"
  else if stage = Stage.Serving && tensor_inputs = [] then
    Error "serving-stage package schedule has no tensor-store inputs"
  else Ok { stage; model; files; kernels; schedule; tensor_store; cache }

let compiled_graph ?model ~files ~kernels ~schedule ~cache () =
  create ~stage:Stage.Compiled_graph ?model ~files ~kernels ~schedule
    ~tensor_store:None ~cache ()

let serving ?model ~files ~kernels ~schedule ~tensor_store ~cache () =
  create ~stage:Stage.Serving ?model ~files ~kernels ~schedule
    ~tensor_store:(Some tensor_store) ~cache ()

let stage package = package.stage
let model package = package.model
let files package = package.files
let kernels package = package.kernels
let schedule package = package.schedule
let tensor_store package = package.tensor_store
let cache package = package.cache

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
  | _ -> Error (Printf.sprintf "unknown serving-package option tag: %d" tag)

let write_artifact writer artifact =
  Binary.Writer.string writer (Artifact.path artifact)

let read_artifact reader =
  let* path = Binary.Reader.string reader in
  Artifact.create path

let operation_tag = function
  | Kernel_abi.Operation.Matmul -> 0
  | Kernel_abi.Operation.Fused_linear -> 1
  | Kernel_abi.Operation.Linear -> 2
  | Kernel_abi.Operation.Q8_linear -> 3
  | Kernel_abi.Operation.Q8_dequantize -> 4
  | Kernel_abi.Operation.Rms_norm -> 5
  | Kernel_abi.Operation.Short_conv -> 6
  | Kernel_abi.Operation.Attention -> 7
  | Kernel_abi.Operation.Embedding -> 8
  | Kernel_abi.Operation.Arange -> 9
  | Kernel_abi.Operation.Diff -> 10
  | Kernel_abi.Operation.Cumsum -> 11
  | Kernel_abi.Operation.Fill -> 12
  | Kernel_abi.Operation.Gather2 -> 13
  | Kernel_abi.Operation.Cast -> 14
  | Kernel_abi.Operation.Pointwise -> 15
  | Kernel_abi.Operation.Movement -> 16
  | Kernel_abi.Operation.Reduction -> 17
  | Kernel_abi.Operation.Update_slice -> 18

let operation_of_tag = function
  | 0 -> Ok Kernel_abi.Operation.Matmul
  | 1 -> Ok Kernel_abi.Operation.Fused_linear
  | 2 -> Ok Kernel_abi.Operation.Linear
  | 3 -> Ok Kernel_abi.Operation.Q8_linear
  | 4 -> Ok Kernel_abi.Operation.Q8_dequantize
  | 5 -> Ok Kernel_abi.Operation.Rms_norm
  | 6 -> Ok Kernel_abi.Operation.Short_conv
  | 7 -> Ok Kernel_abi.Operation.Attention
  | 8 -> Ok Kernel_abi.Operation.Embedding
  | 9 -> Ok Kernel_abi.Operation.Arange
  | 10 -> Ok Kernel_abi.Operation.Diff
  | 11 -> Ok Kernel_abi.Operation.Cumsum
  | 12 -> Ok Kernel_abi.Operation.Fill
  | 13 -> Ok Kernel_abi.Operation.Gather2
  | 14 -> Ok Kernel_abi.Operation.Cast
  | 15 -> Ok Kernel_abi.Operation.Pointwise
  | 16 -> Ok Kernel_abi.Operation.Movement
  | 17 -> Ok Kernel_abi.Operation.Reduction
  | 18 -> Ok Kernel_abi.Operation.Update_slice
  | tag -> Error (Printf.sprintf "unknown kernel operation tag: %d" tag)

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
  | tag -> Error (Printf.sprintf "unknown kernel dtype tag: %d" tag)

let write_kernel writer entry =
  Binary.Writer.string writer (Kernel_abi.Entry.name entry);
  Binary.Writer.u8 writer (operation_tag (Kernel_abi.Entry.operation entry));
  Binary.Writer.u8 writer (dtype_tag (Kernel_abi.Entry.input_dtype entry));
  Binary.Writer.u8 writer (dtype_tag (Kernel_abi.Entry.output_dtype entry));
  let x, y, z = Kernel_abi.Entry.threadgroup entry in
  List.iter (Binary.Writer.u32 writer) [ x; y; z ]

let read_kernel reader =
  let* name = Binary.Reader.string reader in
  let* operation_tag = Binary.Reader.u8 reader in
  let* operation = operation_of_tag operation_tag in
  let* input_tag = Binary.Reader.u8 reader in
  let* input_dtype = dtype_of_tag input_tag in
  let* output_tag = Binary.Reader.u8 reader in
  let* output_dtype = dtype_of_tag output_tag in
  let* x = Binary.Reader.u32 reader in
  let* y = Binary.Reader.u32 reader in
  let* z = Binary.Reader.u32 reader in
  Kernel_abi.Entry.create ~name ~operation ~input_dtype ~output_dtype
    ~threadgroup:(x, y, z)

let write_kv_format writer = function
  | Kv_cache.Format.F16 -> Binary.Writer.u8 writer 0
  | Kv_cache.Format.Q8 { group_size } ->
      Binary.Writer.u8 writer 1;
      Binary.Writer.u32 writer group_size

let read_kv_format reader =
  let* tag = Binary.Reader.u8 reader in
  match tag with
  | 0 -> Ok Kv_cache.Format.f16
  | 1 ->
      let* group_size = Binary.Reader.u32 reader in
      Kv_cache.Format.q8 ~group_size
  | _ -> Error (Printf.sprintf "unknown KV format tag: %d" tag)

let write_cache writer cache =
  Binary.Writer.u32 writer cache.Cache.page_size;
  write_kv_format writer cache.default_kv;
  Binary.Writer.u16 writer (List.length cache.supported_kv);
  List.iter (write_kv_format writer) cache.supported_kv

let read_cache reader =
  let* page_size = Binary.Reader.u32 reader in
  let* default_kv = read_kv_format reader in
  let* count = Binary.Reader.u16 reader in
  let rec formats acc remaining =
    if remaining = 0 then Ok (List.rev acc)
    else
      let* format = read_kv_format reader in
      formats (format :: acc) (remaining - 1)
  in
  let* supported_kv = formats [] count in
  Cache.create ~page_size ~default_kv ~supported_kv

let write_tensor_store writer (Tensor_store.Weights { file }) =
  Binary.Writer.u8 writer 0;
  write_artifact writer file

let read_tensor_store reader =
  let* tag = Binary.Reader.u8 reader in
  if tag <> 0 then Error (Printf.sprintf "unknown tensor-store tag: %d" tag)
  else
    let* file = read_artifact reader in
    Ok (Tensor_store.weights ~file)

let magic = "LLMOPTPK"

let to_bytes package =
  let writer = Binary.Writer.create () in
  Binary.Writer.raw_string writer magic;
  Binary.Writer.u16 writer 6;
  Binary.Writer.u8 writer
    (match package.stage with Stage.Compiled_graph -> 0 | Stage.Serving -> 1);
  Binary.Writer.u8 writer 0;
  write_option writer Binary.Writer.string package.model;
  write_artifact writer (Files.metal_library package.files);
  Binary.Writer.u16 writer (List.length package.kernels);
  List.iter (write_kernel writer) package.kernels;
  write_option writer write_tensor_store package.tensor_store;
  write_cache writer package.cache;
  package.schedule |> Serving_schedule.to_bytes |> Binary.Writer.bytes writer;
  Binary.Writer.contents writer

let of_bytes bytes =
  let reader = Binary.Reader.create bytes in
  let* actual_magic =
    Binary.Reader.raw_string reader ~length:(String.length magic)
  in
  if actual_magic <> magic then Error "invalid serving-package magic"
  else
    let* version = Binary.Reader.u16 reader in
    if
      version <> 2 && version <> 3 && version <> 4 && version <> 5
      && version <> 6
    then
      Error (Printf.sprintf "unsupported serving-package version: %d" version)
    else
      let* stage_tag = Binary.Reader.u8 reader in
      let* stage =
        match stage_tag with
        | 0 -> Ok Stage.Compiled_graph
        | 1 -> Ok Stage.Serving
        | _ ->
            Error
              (Printf.sprintf "unknown serving-package stage tag: %d" stage_tag)
      in
      let* target = Binary.Reader.u8 reader in
      if target <> 0 then
        Error
          (Printf.sprintf "unsupported serving-package target tag: %d" target)
      else
        let* model = read_option reader Binary.Reader.string in
        let* metal_library = read_artifact reader in
        let files = Files.create ~metal_library in
        let* kernel_count = Binary.Reader.u16 reader in
        let rec kernels acc remaining =
          if remaining = 0 then Ok (List.rev acc)
          else
            let* kernel = read_kernel reader in
            kernels (kernel :: acc) (remaining - 1)
        in
        let* kernels = kernels [] kernel_count in
        let* tensor_store = read_option reader read_tensor_store in
        let* cache = read_cache reader in
        let* schedule_bytes = Binary.Reader.bytes reader in
        let* schedule = Serving_schedule.of_bytes schedule_bytes in
        let* () = Binary.Reader.finish reader in
        create ~stage ?model ~files ~kernels ~schedule ~tensor_store ~cache ()

let write_file path package =
  try
    let channel = open_out_bin path in
    Fun.protect
      ~finally:(fun () -> close_out_noerr channel)
      (fun () ->
        output_bytes channel (to_bytes package);
        flush channel;
        Ok ())
  with Sys_error message -> Error ("cannot write serving package: " ^ message)

let of_file path =
  try
    let channel = open_in_bin path in
    Fun.protect
      ~finally:(fun () -> close_in_noerr channel)
      (fun () ->
        let bytes = really_input_string channel (in_channel_length channel) in
        of_bytes (Bytes.of_string bytes))
  with
  | Sys_error message -> Error ("cannot read serving package: " ^ message)
  | End_of_file -> Error "serving package ended unexpectedly"

let validate_files ~root package =
  let artifacts =
    Files.metal_library package.files
    :: Option.fold ~none:[]
         ~some:(fun tensor_store -> [ Tensor_store.file tensor_store ])
         package.tensor_store
  in
  let rec check = function
    | [] -> Ok ()
    | artifact :: rest ->
        let relative = Artifact.path artifact in
        let path = Filename.concat root relative in
        (try
           let stats = Unix.stat path in
           if stats.st_kind <> Unix.S_REG then
             Error ("serving-package artifact is not a regular file: " ^ relative)
           else check rest
         with Unix.Unix_error (error, _, _) ->
           Error
             (Printf.sprintf "cannot access serving-package artifact %s: %s"
                relative (Unix.error_message error)))
  in
  check artifacts
