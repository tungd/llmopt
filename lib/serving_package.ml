open Yojson.Basic.Util

let ( let* ) = Result.bind

module Stage = struct
  type t = Compiled_graph | Serving

  let to_string = function
    | Compiled_graph -> "compiled-graph"
    | Serving -> "serving"

  let of_string = function
    | "compiled-graph" -> Ok Compiled_graph
    | "serving" -> Ok Serving
    | value -> Error ("unsupported serving-package stage: " ^ value)
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

let artifact_of_json json =
  try json |> to_string |> Artifact.create
  with Type_error (message, _) -> Error ("invalid artifact path: " ^ message)

module Files = struct
  type t = {
    fx : Artifact.t;
    plan : Artifact.t;
    metal_source : Artifact.t;
    metal_library : Artifact.t;
    llvm_ir : Artifact.t;
  }

  let create ~fx ~plan ~metal_source ~metal_library ~llvm_ir =
    { fx; plan; metal_source; metal_library; llvm_ir }

  let fx files = files.fx
  let plan files = files.plan
  let metal_source files = files.metal_source
  let metal_library files = files.metal_library
  let llvm_ir files = files.llvm_ir
  let all files =
    [ files.fx; files.plan; files.metal_source; files.metal_library; files.llvm_ir ]

  let to_yojson files =
    `Assoc
      [ ("fx", `String (Artifact.path files.fx));
        ("plan", `String (Artifact.path files.plan));
        ("metal_source", `String (Artifact.path files.metal_source));
        ("metal_library", `String (Artifact.path files.metal_library));
        ("llvm_ir", `String (Artifact.path files.llvm_ir)) ]

  let of_yojson json =
    let* fx = artifact_of_json (member "fx" json) in
    let* plan = artifact_of_json (member "plan" json) in
    let* metal_source = artifact_of_json (member "metal_source" json) in
    let* metal_library = artifact_of_json (member "metal_library" json) in
    let* llvm_ir = artifact_of_json (member "llvm_ir" json) in
    Ok (create ~fx ~plan ~metal_source ~metal_library ~llvm_ir)
end

module Tensor_store = struct
  type t = Safetensors of { file : Artifact.t }

  let safetensors ~file = Safetensors { file }
  let file (Safetensors { file }) = file

  let to_yojson (Safetensors { file }) =
    `Assoc
      [ ("format", `String "safetensors");
        ("file", `String (Artifact.path file)) ]

  let of_yojson json =
    try
      match json |> member "format" |> to_string with
      | "safetensors" ->
          let* file = artifact_of_json (member "file" json) in
          Ok (safetensors ~file)
      | value -> Error ("unsupported tensor-store format: " ^ value)
    with Type_error (message, _) ->
      Error ("invalid serving-package tensor store: " ^ message)
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

  let format_to_yojson = function
    | Kv_cache.Format.F16 -> `Assoc [ ("format", `String "f16") ]
    | Kv_cache.Format.Q8 { group_size } ->
        `Assoc
          [ ("format", `String "q8"); ("group_size", `Int group_size) ]

  let format_of_yojson json =
    try
      match json |> member "format" |> to_string with
      | "f16" -> Ok Kv_cache.Format.f16
      | "q8" ->
          json |> member "group_size" |> to_int |> fun group_size ->
          Kv_cache.Format.q8 ~group_size
      | value -> Error ("unsupported serving-package KV format: " ^ value)
    with Type_error (message, _) ->
      Error ("invalid serving-package KV format: " ^ message)

  let to_yojson cache =
    `Assoc
      [ ( "radix",
          `Assoc
            [ ("mode", `String "required");
              ("page_size", `Int cache.page_size) ] );
        ( "kv",
          `Assoc
            [ ("default", format_to_yojson cache.default_kv);
              ( "supported",
                `List (List.map format_to_yojson cache.supported_kv) ) ] ) ]

  let of_yojson json =
    try
      let radix = member "radix" json in
      let mode = radix |> member "mode" |> to_string in
      if mode <> "required" then
        Error "serving-package radix cache must be required"
      else
        let page_size = radix |> member "page_size" |> to_int in
        let kv = member "kv" json in
        let* default_kv = format_of_yojson (member "default" kv) in
        let rec parse_formats acc = function
          | [] -> Ok (List.rev acc)
          | value :: rest ->
              let* format = format_of_yojson value in
              parse_formats (format :: acc) rest
        in
        let* supported_kv = parse_formats [] (kv |> member "supported" |> to_list) in
        create ~page_size ~default_kv ~supported_kv
    with Type_error (message, _) ->
      Error ("invalid serving-package cache policy: " ^ message)
end

type t = {
  stage : Stage.t;
  model : string option;
  files : Files.t;
  kernels : Kernel_abi.Entry.t list;
  tensor_store : Tensor_store.t option;
  cache : Cache.t;
}

let create ~stage ?model ~files ~kernels ~tensor_store ~cache () =
  let kernel_names = List.map Kernel_abi.Entry.name kernels in
  if Option.exists (fun value -> String.trim value = "") model then
    Error "serving-package model identifier cannot be empty"
  else if kernels = [] then Error "serving-package must declare at least one kernel"
  else if List.length kernel_names <> List.length (List.sort_uniq String.compare kernel_names)
  then Error "serving-package kernel entry-point names must be unique"
  else if stage = Stage.Compiled_graph && Option.is_some tensor_store then
    Error "compiled-graph package cannot declare a tensor store"
  else if stage = Stage.Serving && Option.is_none tensor_store then
    Error "serving-stage package must declare a tensor store"
  else Ok { stage; model; files; kernels; tensor_store; cache }

let compiled_graph ?model ~files ~kernels ~cache () =
  create ~stage:Stage.Compiled_graph ?model ~files ~kernels ~tensor_store:None
    ~cache ()

let serving ?model ~files ~kernels ~tensor_store ~cache () =
  create ~stage:Stage.Serving ?model ~files ~kernels
    ~tensor_store:(Some tensor_store) ~cache ()

let stage package = package.stage
let model package = package.model
let files package = package.files
let kernels package = package.kernels
let tensor_store package = package.tensor_store
let cache package = package.cache

let to_yojson package =
  `Assoc
    [ ("schema", `String "llmopt.serving-package");
      ("version", `Int 1);
      ("stage", `String (Stage.to_string package.stage));
      ("target", `Assoc [ ("platform", `String "metal") ]);
      ("model", Option.fold ~none:`Null ~some:(fun value -> `String value) package.model);
      ("files", Files.to_yojson package.files);
      ("kernels", `List (List.map Kernel_abi.Entry.to_yojson package.kernels));
      ( "tensor_store",
        Option.fold ~none:`Null ~some:Tensor_store.to_yojson
          package.tensor_store );
      ("cache", Cache.to_yojson package.cache) ]

let parse_list parse values =
  let rec loop acc = function
    | [] -> Ok (List.rev acc)
    | value :: rest ->
        let* parsed = parse value in
        loop (parsed :: acc) rest
  in
  loop [] values

let of_yojson json =
  try
    let schema = json |> member "schema" |> to_string in
    let version = json |> member "version" |> to_int in
    let platform = json |> member "target" |> member "platform" |> to_string in
    if schema <> "llmopt.serving-package" then
      Error ("unsupported serving-package schema: " ^ schema)
    else if version <> 1 then
      Error (Printf.sprintf "unsupported serving-package version: %d" version)
    else if platform <> "metal" then
      Error ("unsupported serving-package target: " ^ platform)
    else
      let* stage = json |> member "stage" |> to_string |> Stage.of_string in
      let model = json |> member "model" |> to_string_option in
      let* files = Files.of_yojson (member "files" json) in
      let* kernels =
        json |> member "kernels" |> to_list
        |> parse_list Kernel_abi.Entry.of_yojson
      in
      let* tensor_store =
        match member "tensor_store" json with
        | `Null -> Ok None
        | value -> Tensor_store.of_yojson value |> Result.map Option.some
      in
      let* cache = Cache.of_yojson (member "cache" json) in
      create ~stage ?model ~files ~kernels ~tensor_store ~cache ()
  with
  | Type_error (message, _) -> Error ("invalid serving package: " ^ message)
  | Yojson.Json_error message -> Error ("invalid serving package: " ^ message)

let write_file path package =
  try
    let channel = open_out path in
    Fun.protect
      ~finally:(fun () -> close_out_noerr channel)
      (fun () ->
        Yojson.Basic.pretty_to_channel channel (to_yojson package);
        output_char channel '\n';
        Ok ())
  with Sys_error message -> Error ("cannot write serving package: " ^ message)

let of_file path =
  try of_yojson (Yojson.Basic.from_file path)
  with
  | Sys_error message -> Error ("cannot read serving package: " ^ message)
  | Yojson.Json_error message -> Error ("invalid serving package: " ^ message)

let validate_files ~root package =
  let tensor_store_artifacts =
    Option.fold ~none:[]
      ~some:(fun tensor_store -> [ Tensor_store.file tensor_store ])
      package.tensor_store
  in
  let artifacts = Files.all package.files @ tensor_store_artifacts in
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
