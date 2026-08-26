let float16 bits =
  let sign = if bits land 0x8000 = 0 then 1.0 else -1.0 in
  let exponent = bits lsr 10 land 0x1f in
  let fraction = bits land 0x03ff in
  match exponent with
  | 0 when fraction = 0 -> sign *. 0.0
  | 0 -> sign *. Float.ldexp (float_of_int fraction) (-24)
  | 31 when fraction = 0 -> sign *. Float.infinity
  | 31 -> Float.nan
  | exponent ->
      sign
      *. Float.ldexp (1.0 +. (float_of_int fraction /. 1024.0))
           (exponent - 15)

module Float16_logits = struct
  let row_offset ~vocabulary bytes =
    if vocabulary <= 0 then Error "float16 logits vocabulary must be positive"
    else if vocabulary > max_int / 2 then
      Error "float16 logits vocabulary byte length overflows"
    else
      let row_bytes = 2 * vocabulary in
      let length = Bytes.length bytes in
      if length = 0 || length mod row_bytes <> 0 then
        Error
          (Printf.sprintf
             "float16 logits contain %d bytes; expected complete %d-byte rows"
             length row_bytes)
      else Ok (length - row_bytes, row_bytes)

  let last_row ~vocabulary bytes =
    let ( let* ) = Result.bind in
    let* row_offset, row_bytes = row_offset ~vocabulary bytes in
    Ok (Bytes.sub bytes row_offset row_bytes)
end

module Greedy = struct
  let token_at bytes offset =
    let raw = Bytes.get_int32_le bytes offset |> Int64.of_int32 in
    Int64.logand raw 0xffff_ffffL |> Int64.to_int

  let on_device bytes =
    if Bytes.length bytes <> 4 then
      Error
        (Printf.sprintf "on-device greedy token must contain 4 bytes; got %d"
           (Bytes.length bytes))
    else Ok (token_at bytes 0)

  let on_device_last bytes =
    if Bytes.length bytes = 0 || Bytes.length bytes mod 4 <> 0 then
      Error
        (Printf.sprintf
           "on-device greedy token rows contain %d bytes; expected a positive multiple of 4"
           (Bytes.length bytes))
    else Ok (token_at bytes (Bytes.length bytes - 4))

  let f16_last_row ~vocabulary bytes =
    let ( let* ) = Result.bind in
    let* row = Float16_logits.last_row ~vocabulary bytes in
    let value token = Bytes.get_uint16_le row (2 * token) |> float16 in
    let first = value 0 in
    if Float.is_nan first then Error "float16 logits contain NaN at token 0"
    else
      let rec select best_token best_value token =
        if token = vocabulary then Ok best_token
        else
          let candidate = value token in
          if Float.is_nan candidate then
            Error
              (Printf.sprintf "float16 logits contain NaN at token %d" token)
          else if candidate > best_value then
            select token candidate (token + 1)
          else select best_token best_value (token + 1)
      in
      select 0 first 1
end
