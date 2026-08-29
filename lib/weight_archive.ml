let ( let* ) = Result.bind

let magic = "LLMOPTWT"
let version = 1
let alignment = 256
let prefix_bytes = 24

module Dtype = struct
  type quant_type =
    | Q8_0
    | Q4_K
    | Q5_K
    | Q6_K
    | Q5_0
    | Q4_0
    | IQ4_XS

  type t =
    | F32
    | F16
    | BF16
    | I64
    | I32
    | I8
    | Bool
    | U8
    | Quant of quant_type

  let quant_to_string = function
    | Q8_0 -> "Q8_0"
    | Q4_K -> "Q4_K"
    | Q5_K -> "Q5_K"
    | Q6_K -> "Q6_K"
    | Q5_0 -> "Q5_0"
    | Q4_0 -> "Q4_0"
    | IQ4_XS -> "IQ4_XS"

  let quant_of_string = function
    | "Q8_0" | "q8_0" -> Some Q8_0
    | "Q4_K" | "q4_k" -> Some Q4_K
    | "Q5_K" | "q5_k" -> Some Q5_K
    | "Q6_K" | "q6_k" -> Some Q6_K
    | "Q5_0" | "q5_0" -> Some Q5_0
    | "Q4_0" | "q4_0" -> Some Q4_0
    | "IQ4_XS" | "iq4_xs" -> Some IQ4_XS
    | _ -> None

  let block_size = function
    | Q8_0 -> 32
    | Q4_K -> 256
    | Q5_K -> 256
    | Q6_K -> 256
    | Q5_0 -> 32
    | Q4_0 -> 32
    | IQ4_XS -> 256

  let bytes_per_block = function
    | Q8_0 -> 34
    | Q4_K -> 144
    | Q5_K -> 176
    | Q6_K -> 210
    | Q5_0 -> 22
    | Q4_0 -> 18
    | IQ4_XS -> 136

  let to_string = function
    | F32 -> "F32"
    | F16 -> "F16"
    | BF16 -> "BF16"
    | I64 -> "I64"
    | I32 -> "I32"
    | I8 -> "I8"
    | Bool -> "BOOL"
    | U8 -> "U8"
    | Quant q -> quant_to_string q

  let of_tag = function
    | 0 -> Ok F32
    | 1 -> Ok F16
    | 2 -> Ok BF16
    | 3 -> Ok I64
    | 4 -> Ok I32
    | 5 -> Ok I8
    | 6 -> Ok Bool
    | 7 -> Ok U8
    | 8 -> Ok (Quant Q8_0)
    | 9 -> Ok (Quant Q4_K)
    | 10 -> Ok (Quant Q5_K)
    | 11 -> Ok (Quant Q6_K)
    | 12 -> Ok (Quant Q5_0)
    | tag -> Error (Printf.sprintf "unknown weight-archive dtype tag: %d" tag)

  let byte_width = function
    | F32 | I32 -> 4
    | F16 | BF16 -> 2
    | I64 -> 8
    | I8 | Bool | U8 -> 1
    | Quant q -> bytes_per_block q
end

module Tensor = struct
  type t = {
    name : string;
    dtype : Dtype.t;
    shape : int list;
    offset : int;
    byte_length : int;
  }

  let name tensor = tensor.name
  let dtype tensor = tensor.dtype
  let shape tensor = tensor.shape
  let offset tensor = tensor.offset
  let byte_length tensor = tensor.byte_length
end

module Tensor_map = Map.Make (String)

type t = {
  path : string;
  file_size : int;
  index_bytes : int;
  tensors : Tensor.t list;
  index : Tensor.t Tensor_map.t;
}

let path archive = archive.path
let file_size archive = archive.file_size
let index_bytes archive = archive.index_bytes
let tensors archive = archive.tensors
let find archive name = Tensor_map.find_opt name archive.index

let align value =
  if value > max_int - (alignment - 1) then
    Error "weight archive alignment overflows the host integer range"
  else Ok (((value + alignment - 1) / alignment) * alignment)

let checked_product values =
  let rec loop product = function
    | [] -> Ok product
    | value :: rest ->
        if value < 0 then Error "weight archive dimensions cannot be negative"
        else if value <> 0 && product > max_int / value then
          Error "weight archive tensor size exceeds the host integer range"
        else loop (product * value) rest
  in
  loop 1 values

let read_dimensions reader rank =
  let rec loop acc remaining =
    if remaining = 0 then Ok (List.rev acc)
    else
      let* dimension = Binary.Reader.u64 reader in
      loop (dimension :: acc) (remaining - 1)
  in
  loop [] rank

let parse_tensor reader ~previous_name =
  let* name_length = Binary.Reader.u32 reader in
  let* dtype_tag = Binary.Reader.u8 reader in
  let* dtype = Dtype.of_tag dtype_tag in
  let* rank = Binary.Reader.u8 reader in
  let* reserved = Binary.Reader.u16 reader in
  let* name = Binary.Reader.raw_string reader ~length:name_length in
  if reserved <> 0 then Error "weight archive tensor reserved field is non-zero"
  else if name = "" then Error "weight archive tensor name cannot be empty"
  else if String.contains name '\000' then
    Error "weight archive tensor name contains a NUL byte"
  else if
    Option.exists (fun previous -> String.compare previous name >= 0)
      previous_name
  then Error "weight archive tensor names are not strictly sorted"
  else
    let* shape = read_dimensions reader rank in
    let* offset = Binary.Reader.u64 reader in
    let* byte_length = Binary.Reader.u64 reader in
    let* elements = checked_product shape in
    if elements = 0 || byte_length = 0 then
      Error ("weight archive tensor " ^ name ^ " is empty")
    else
      let* expected_byte_length =
        match dtype with
        | Dtype.F32 | Dtype.I32 ->
            if elements > max_int / 4 then
              Error ("weight archive tensor " ^ name ^ " byte size overflows")
            else Ok (elements * 4)
        | Dtype.F16 | Dtype.BF16 ->
            if elements > max_int / 2 then
              Error ("weight archive tensor " ^ name ^ " byte size overflows")
            else Ok (elements * 2)
        | Dtype.I64 ->
            if elements > max_int / 8 then
              Error ("weight archive tensor " ^ name ^ " byte size overflows")
            else Ok (elements * 8)
        | Dtype.I8 | Dtype.Bool | Dtype.U8 -> Ok elements
        | Dtype.Quant q ->
            let blk = Dtype.block_size q in
            let bpb = Dtype.bytes_per_block q in
            if elements mod blk <> 0 then
              Error
                (Printf.sprintf
                   "weight archive tensor %s element count %d is not a \
                    multiple of quant block size %d"
                   name elements blk)
            else if elements / blk > max_int / bpb then
              Error ("weight archive tensor " ^ name ^ " byte size overflows")
            else Ok ((elements / blk) * bpb)
      in
      if byte_length <> expected_byte_length then
        Error
          ("weight archive tensor " ^ name
         ^ " byte length disagrees with its dtype and shape")
      else if offset mod alignment <> 0 then
        Error ("weight archive tensor " ^ name ^ " offset is not aligned")
      else Ok { Tensor.name = name; dtype; shape; offset; byte_length }

let all_zero bytes =
  let rec loop index =
    index = Bytes.length bytes
    || (Bytes.get bytes index = '\000' && loop (index + 1))
  in
  loop 0

let parse_header bytes ~file_size =
  let reader = Binary.Reader.create bytes in
  let* actual_magic =
    Binary.Reader.raw_string reader ~length:(String.length magic)
  in
  if actual_magic <> magic then Error "invalid weight-archive magic"
  else
    let* actual_version = Binary.Reader.u16 reader in
    if actual_version <> version then
      Error
        (Printf.sprintf "unsupported weight-archive version: %d" actual_version)
    else
      let* flags = Binary.Reader.u16 reader in
      if flags <> 0 then
        Error (Printf.sprintf "unsupported weight-archive flags: %d" flags)
      else
        let* tensor_count = Binary.Reader.u32 reader in
        let* index_bytes = Binary.Reader.u64 reader in
        if tensor_count = 0 then Error "weight archive contains no tensors"
        else if index_bytes <> Bytes.length bytes then
          Error "weight archive index length changed while reading"
        else
          let rec entries acc previous_name remaining =
            if remaining = 0 then Ok (List.rev acc)
            else
              let* tensor = parse_tensor reader ~previous_name in
              entries (tensor :: acc) (Some (Tensor.name tensor)) (remaining - 1)
          in
          let* tensors = entries [] None tensor_count in
          let padding_length = Binary.Reader.remaining reader in
          let* padding = Binary.Reader.raw_bytes reader ~length:padding_length in
          let* () = Binary.Reader.finish reader in
          if not (all_zero padding) then
            Error "weight archive index padding contains non-zero bytes"
          else
            let rec validate_payload cursor = function
              | [] ->
                  if cursor = file_size then Ok ()
                  else
                    Error
                      "weight archive file has trailing or missing payload bytes"
              | tensor :: rest ->
                  let* expected_offset = align cursor in
                  if Tensor.offset tensor <> expected_offset then
                    Error
                      ("weight archive tensor " ^ Tensor.name tensor
                     ^ " has an unexpected payload offset")
                  else if
                    expected_offset > file_size
                    || Tensor.byte_length tensor > file_size - expected_offset
                  then
                    Error
                      ("weight archive tensor " ^ Tensor.name tensor
                     ^ " exceeds the file")
                  else
                    validate_payload
                      (expected_offset + Tensor.byte_length tensor)
                      rest
            in
            let* () = validate_payload index_bytes tensors in
            let index =
              List.fold_left
                (fun index tensor ->
                  Tensor_map.add (Tensor.name tensor) tensor index)
                Tensor_map.empty tensors
            in
            Ok (index_bytes, tensors, index)

let int_of_int64 label value =
  if value < 0L || value > Int64.of_int max_int then
    Error (Printf.sprintf "weight archive %s exceeds the host integer range" label)
  else Ok (Int64.to_int value)

let parse_prefix bytes =
  let reader = Binary.Reader.create bytes in
  let* actual_magic = Binary.Reader.raw_string reader ~length:8 in
  if actual_magic <> magic then Error "invalid weight-archive magic"
  else
    let* actual_version = Binary.Reader.u16 reader in
    if actual_version <> version then
      Error
        (Printf.sprintf "unsupported weight-archive version: %d" actual_version)
    else
      let* flags = Binary.Reader.u16 reader in
      if flags <> 0 then
        Error (Printf.sprintf "unsupported weight-archive flags: %d" flags)
      else
        let* tensor_count = Binary.Reader.u32 reader in
        let* index_bytes = Binary.Reader.u64 reader in
        let* () = Binary.Reader.finish reader in
        if tensor_count = 0 then Error "weight archive contains no tensors"
        else Ok index_bytes

let of_file path =
  try
    let stat = Unix.LargeFile.stat path in
    if stat.st_kind <> Unix.S_REG then
      Error ("weight archive path is not a regular file: " ^ path)
    else
      let* file_size = int_of_int64 "file size" stat.st_size in
      if file_size < prefix_bytes then Error "weight archive file is too short"
      else
        let channel = open_in_bin path in
        Fun.protect
          ~finally:(fun () -> close_in_noerr channel)
          (fun () ->
            let prefix = really_input_string channel prefix_bytes |> Bytes.of_string in
            let* index_bytes = parse_prefix prefix in
            if index_bytes < prefix_bytes then
              Error "weight archive index is shorter than its prefix"
            else if index_bytes mod alignment <> 0 then
              Error "weight archive index is not aligned"
            else if index_bytes > file_size then
              Error "weight archive index extends beyond the file"
            else
              let suffix =
                really_input_string channel (index_bytes - prefix_bytes)
                |> Bytes.of_string
              in
              let header = Bytes.cat prefix suffix in
              let* index_bytes, tensors, index =
                parse_header header ~file_size
              in
              Ok { path; file_size; index_bytes; tensors; index })
  with
  | Sys_error message -> Error ("cannot read weight archive: " ^ message)
  | Unix.Unix_error (error, _, _) ->
      Error ("cannot stat weight archive: " ^ Unix.error_message error)
  | End_of_file -> Error "weight archive ended unexpectedly"
