let invalid_unsigned name maximum value =
  invalid_arg
    (Printf.sprintf "binary %s value must be in [0,%d], got %d" name maximum
       value)

let ( let* ) = Result.bind

module Writer = struct
  type t = Buffer.t

  let create () = Buffer.create 256
  let raw_string = Buffer.add_string
  let raw_bytes writer bytes = Buffer.add_string writer (Bytes.unsafe_to_string bytes)

  let encoded length encode =
    let bytes = Bytes.create length in
    encode bytes;
    bytes

  let u8 writer value =
    if value < 0 || value > 0xff then invalid_unsigned "u8" 0xff value;
    Buffer.add_char writer (Char.chr value)

  let u16 writer value =
    if value < 0 || value > 0xffff then invalid_unsigned "u16" 0xffff value;
    encoded 2 (fun bytes -> Bytes.set_uint16_le bytes 0 value)
    |> raw_bytes writer

  let u32 writer value =
    if value < 0 || Int64.of_int value > 0xffff_ffffL then
      invalid_arg (Printf.sprintf "binary u32 value is out of range: %d" value);
    encoded 4 (fun bytes -> Bytes.set_int32_le bytes 0 (Int32.of_int value))
    |> raw_bytes writer

  let u64 writer value =
    if value < 0 then invalid_arg "binary u64 value cannot be negative";
    encoded 8 (fun bytes -> Bytes.set_int64_le bytes 0 (Int64.of_int value))
    |> raw_bytes writer

  let u64_int64 writer value =
    encoded 8 (fun bytes -> Bytes.set_int64_le bytes 0 value)
    |> raw_bytes writer

  let i64 writer value =
    encoded 8 (fun bytes -> Bytes.set_int64_le bytes 0 (Int64.of_int value))
    |> raw_bytes writer

  let i64_int64 = u64_int64

  let float64 writer value =
    encoded 8 (fun bytes ->
        Bytes.set_int64_le bytes 0 (Int64.bits_of_float value))
    |> raw_bytes writer

  let float32 writer value =
    encoded 4 (fun bytes ->
        Bytes.set_int32_le bytes 0 (Int32.bits_of_float value))
    |> raw_bytes writer

  let bool writer value = u8 writer (if value then 1 else 0)

  let string writer value =
    u32 writer (String.length value);
    raw_string writer value

  let bytes writer value =
    u64 writer (Bytes.length value);
    raw_bytes writer value

  let contents writer = Buffer.contents writer |> Bytes.of_string
end

module Reader = struct
  type t = {
    bytes : bytes;
    mutable offset : int;
  }

  let create bytes = { bytes; offset = 0 }
  let offset reader = reader.offset
  let remaining reader = Bytes.length reader.bytes - reader.offset

  let require reader length =
    if length < 0 then Error "binary read length cannot be negative"
    else if length > remaining reader then
      Error
        (Printf.sprintf
           "truncated binary input at byte %d: need %d bytes, have %d"
           reader.offset length (remaining reader))
    else Ok ()

  let raw_bytes reader ~length =
    match require reader length with
    | Error _ as error -> error
    | Ok () ->
        let value = Bytes.sub reader.bytes reader.offset length in
        reader.offset <- reader.offset + length;
        Ok value

  let raw_string reader ~length =
    raw_bytes reader ~length |> Result.map Bytes.to_string

  let u8 reader =
    match require reader 1 with
    | Error _ as error -> error
    | Ok () ->
        let value = Bytes.get_uint8 reader.bytes reader.offset in
        reader.offset <- reader.offset + 1;
        Ok value

  let i8 reader =
    match require reader 1 with
    | Error _ as error -> error
    | Ok () ->
        let value = Bytes.get_int8 reader.bytes reader.offset in
        reader.offset <- reader.offset + 1;
        Ok value

  let fixed reader length decode =
    match require reader length with
    | Error _ as error -> error
    | Ok () ->
        let value = decode reader.bytes reader.offset in
        reader.offset <- reader.offset + length;
        Ok value

  let u16 reader = fixed reader 2 Bytes.get_uint16_le
  let i16 reader = fixed reader 2 Bytes.get_int16_le

  let u32 reader =
    let* value = fixed reader 4 Bytes.get_int32_le in
    let value = Int64.logand (Int64.of_int32 value) 0xffff_ffffL in
    if value > Int64.of_int max_int then
      Error "binary u32 value exceeds the host integer range"
    else Ok (Int64.to_int value)

  let i32_raw reader = fixed reader 4 Bytes.get_int32_le

  let int64_to_int label value =
    if value < Int64.of_int min_int || value > Int64.of_int max_int then
      Error ("binary " ^ label ^ " value exceeds the host integer range")
    else Ok (Int64.to_int value)

  let u64 reader =
    let* value = fixed reader 8 Bytes.get_int64_le in
    if value < 0L then Error "binary u64 value exceeds signed host range"
    else int64_to_int "u64" value

  let u64_int64 reader = fixed reader 8 Bytes.get_int64_le

  let i64 reader =
    let* value = fixed reader 8 Bytes.get_int64_le in
    int64_to_int "i64" value

  let i64_int64 reader = fixed reader 8 Bytes.get_int64_le

  let float32 reader =
    fixed reader 4 Bytes.get_int32_le |> Result.map Int32.float_of_bits

  let float64 reader =
    fixed reader 8 Bytes.get_int64_le |> Result.map Int64.float_of_bits

  let bool reader =
    let* value = u8 reader in
    match value with
    | 0 -> Ok false
    | 1 -> Ok true
    | value -> Error (Printf.sprintf "invalid binary bool tag: %d" value)

  let string reader =
    let* length = u32 reader in
    raw_string reader ~length

  let bytes reader =
    let* length = u64 reader in
    raw_bytes reader ~length

  let finish reader =
    if remaining reader = 0 then Ok ()
    else
      Error
        (Printf.sprintf "binary input has %d trailing bytes" (remaining reader))
end
