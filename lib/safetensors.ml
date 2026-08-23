open Yojson.Basic.Util

let ( let* ) = Result.bind

module Dtype = struct
  type t = F32 | F16 | BF16 | I64 | I32 | I8 | Bool

  let to_string = function
    | F32 -> "F32"
    | F16 -> "F16"
    | BF16 -> "BF16"
    | I64 -> "I64"
    | I32 -> "I32"
    | I8 -> "I8"
    | Bool -> "BOOL"

  let of_string = function
    | "F32" -> Ok F32
    | "F16" -> Ok F16
    | "BF16" -> Ok BF16
    | "I64" -> Ok I64
    | "I32" -> Ok I32
    | "I8" -> Ok I8
    | "BOOL" -> Ok Bool
    | value -> Error ("unsupported safetensors dtype: " ^ value)

  let byte_width = function
    | F32 | I32 -> 4
    | F16 | BF16 -> 2
    | I64 -> 8
    | I8 | Bool -> 1
end

module Tensor = struct
  type t = {
    name : string;
    dtype : Dtype.t;
    shape : int list;
    relative_begin : int;
    relative_end : int;
    offset : int;
  }

  let name tensor = tensor.name
  let dtype tensor = tensor.dtype
  let shape tensor = tensor.shape
  let offset tensor = tensor.offset
  let byte_length tensor = tensor.relative_end - tensor.relative_begin
end

module Tensor_map = Map.Make (String)

type t = {
  path : string;
  file_size : int;
  tensors : Tensor.t list;
  index : Tensor.t Tensor_map.t;
}

let path archive = archive.path
let file_size archive = archive.file_size
let tensors archive = archive.tensors
let find archive name = Tensor_map.find_opt name archive.index

let int_of_int64 label value =
  if value < 0L || value > Int64.of_int max_int then
    Error (Printf.sprintf "safetensors %s exceeds the host integer range" label)
  else Ok (Int64.to_int value)

let little_endian_u64 bytes =
  let result = ref 0L in
  for index = 0 to 7 do
    let byte = Int64.of_int (Char.code (Bytes.get bytes index)) in
    result := Int64.logor !result (Int64.shift_left byte (8 * index))
  done;
  !result

let checked_product values =
  let rec loop product = function
    | [] -> Ok product
    | value :: rest ->
        if value < 0 then
          Error "safetensors tensor dimensions cannot be negative"
        else if value <> 0 && product > max_int / value then
          Error "safetensors tensor size exceeds the host integer range"
        else loop (product * value) rest
  in
  loop 1 values

let require_unique_fields label fields =
  let names = List.map fst fields in
  if List.length names = List.length (List.sort_uniq String.compare names) then
    Ok ()
  else Error ("safetensors " ^ label ^ " contains duplicate keys")

let parse_offsets json =
  match json |> to_list |> List.map to_int with
  | [ begin_offset; end_offset ]
    when begin_offset >= 0 && end_offset >= begin_offset ->
      Ok (begin_offset, end_offset)
  | [ _; _ ] -> Error "safetensors data offsets are invalid"
  | _ -> Error "safetensors data_offsets must contain exactly two integers"

let parse_tensor ~data_start (tensor_name, json) =
  match json with
  | `Assoc fields ->
      let* () = require_unique_fields ("tensor " ^ tensor_name) fields in
      let* parsed_dtype =
        json |> member "dtype" |> to_string |> Dtype.of_string
      in
      let parsed_shape = json |> member "shape" |> to_list |> List.map to_int in
      let* elements = checked_product parsed_shape in
      let* relative_begin, relative_end =
        parse_offsets (member "data_offsets" json)
      in
      let byte_length = relative_end - relative_begin in
      if elements <> 0 && Dtype.byte_width parsed_dtype > max_int / elements
      then Error ("safetensors tensor " ^ tensor_name ^ " byte size overflows")
      else if byte_length <> elements * Dtype.byte_width parsed_dtype then
        Error
          ("safetensors tensor " ^ tensor_name
         ^ " byte length disagrees with shape")
      else if data_start > max_int - relative_begin then
        Error ("safetensors tensor " ^ tensor_name ^ " offset overflows")
      else
        Ok
          {
            Tensor.name = tensor_name;
            dtype = parsed_dtype;
            shape = parsed_shape;
            relative_begin;
            relative_end;
            offset = data_start + relative_begin;
          }
  | _ ->
      Error ("safetensors tensor " ^ tensor_name ^ " metadata must be an object")

let validate_layout ~payload_size tensors =
  let sorted =
    List.sort
      (fun left right ->
        Int.compare left.Tensor.relative_begin right.Tensor.relative_begin)
      tensors
  in
  let rec loop cursor = function
    | [] ->
        if cursor = payload_size then Ok sorted
        else Error "safetensors data buffer is not fully indexed"
    | tensor :: rest ->
        if tensor.Tensor.relative_begin <> cursor then
          Error "safetensors tensor data contains a hole or overlap"
        else if tensor.Tensor.relative_end > payload_size then
          Error
            ("safetensors tensor " ^ tensor.Tensor.name ^ " exceeds the file")
        else loop tensor.Tensor.relative_end rest
  in
  loop 0 sorted

let of_file path =
  try
    let stat = Unix.LargeFile.stat path in
    if stat.st_kind <> Unix.S_REG then
      Error ("safetensors path is not a regular file: " ^ path)
    else
      let* file_size = int_of_int64 "file size" stat.st_size in
      if file_size < 10 then Error "safetensors file is too short"
      else
        let channel = open_in_bin path in
        Fun.protect
          ~finally:(fun () -> close_in_noerr channel)
          (fun () ->
            let prefix = really_input_string channel 8 |> Bytes.of_string in
            let* header_size =
              little_endian_u64 prefix |> int_of_int64 "header size"
            in
            if header_size > file_size - 8 then
              Error "safetensors header extends beyond the file"
            else
              let header = really_input_string channel header_size in
              if header_size = 0 || header.[0] <> '{' then
                Error "safetensors header must begin with an object"
              else
                let json = Yojson.Basic.from_string header in
                match json with
                | `Assoc fields ->
                    let* () = require_unique_fields "header" fields in
                    let tensor_fields =
                      List.filter
                        (fun (name, _) -> name <> "__metadata__")
                        fields
                    in
                    let data_start = 8 + header_size in
                    let rec parse acc = function
                      | [] -> Ok (List.rev acc)
                      | field :: rest ->
                          let* tensor = parse_tensor ~data_start field in
                          parse (tensor :: acc) rest
                    in
                    let* parsed = parse [] tensor_fields in
                    let* tensors =
                      validate_layout ~payload_size:(file_size - data_start)
                        parsed
                    in
                    let index =
                      List.fold_left
                        (fun index tensor ->
                          Tensor_map.add (Tensor.name tensor) tensor index)
                        Tensor_map.empty tensors
                    in
                    Ok { path; file_size; tensors; index }
                | _ -> Error "safetensors header must be a JSON object")
  with
  | Sys_error message -> Error ("cannot read safetensors file: " ^ message)
  | Unix.Unix_error (error, _, _) ->
      Error ("cannot stat safetensors file: " ^ Unix.error_message error)
  | End_of_file -> Error "safetensors file ended unexpectedly"
  | Yojson.Json_error message -> Error ("invalid safetensors header: " ^ message)
  | Type_error (message, _) -> Error ("invalid safetensors header: " ^ message)
