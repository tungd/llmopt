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

  let suppress ~vocabulary ~suppress_token_ids row =
    let seen = Array.make vocabulary false in
    let rec mark count = function
      | [] ->
          if count >= vocabulary then
            Error "cannot suppress the complete sampling vocabulary"
          else Ok ()
      | token_id :: _ when token_id < 0 || token_id >= vocabulary ->
          Error
            (Printf.sprintf "suppressed token id %d is outside vocabulary [0,%d)"
               token_id vocabulary)
      | token_id :: rest ->
          let count = if seen.(token_id) then count else count + 1 in
          seen.(token_id) <- true;
          mark count rest
    in
    let ( let* ) = Result.bind in
    let* () = mark 0 suppress_token_ids in
    if suppress_token_ids = [] then Ok row
    else
      let masked = Bytes.copy row in
      Array.iteri
        (fun token_id suppressed ->
          if suppressed then Bytes.set_uint16_le masked (2 * token_id) 0xfc00)
        seen;
      Ok masked

  let top ?(suppress_token_ids = []) ~count ~vocabulary bytes =
    let ( let* ) = Result.bind in
    if count <= 0 then Error "float16 logits top count must be positive"
    else
      let* row = last_row ~vocabulary bytes in
      let* row = suppress ~vocabulary ~suppress_token_ids row in
      let ranked =
        Array.init vocabulary (fun token_id ->
            token_id, (Bytes.get_uint16_le row (2 * token_id) |> float16))
      in
      (match Array.find_opt (fun (_, value) -> Float.is_nan value) ranked with
      | Some (token_id, _) ->
          Error
            (Printf.sprintf "float16 logits contain NaN at token %d" token_id)
      | None ->
          Array.sort
            (fun (left_token, left_value) (right_token, right_value) ->
              let by_value = Float.compare right_value left_value in
              if by_value <> 0 then by_value
              else Int.compare left_token right_token)
            ranked;
          Ok
            (Array.sub ranked 0 (min count vocabulary) |> Array.to_list))
end

module Params = struct
  type t = {
    temperature : float;
    top_k : int;
    top_p : float;
    min_p : float;
    seed : int;
  }

  let greedy =
    {
      temperature = 0.0;
      top_k = 1;
      top_p = 1.0;
      min_p = 0.0;
      seed = 0;
    }

  let is_greedy params =
    params.temperature <= 0.0001 || params.top_k = 1

  let create ?(temperature = 0.0) ?(top_k = 0) ?(top_p = 1.0) ?(min_p = 0.0)
      ?(seed = 0) () =
    {
      temperature = Float.max 0.0 temperature;
      top_k = max 0 top_k;
      top_p = Float.max 0.0 (Float.min 1.0 top_p);
      min_p = Float.max 0.0 (Float.min 1.0 min_p);
      seed;
    }
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

  let on_device_all bytes =
    if Bytes.length bytes = 0 || Bytes.length bytes mod 4 <> 0 then
      Error
        (Printf.sprintf
           "on-device greedy token rows contain %d bytes; expected a positive multiple of 4"
           (Bytes.length bytes))
    else
      Ok
        (Array.init (Bytes.length bytes / 4) (fun index ->
             token_at bytes (index * 4)))

  external f16_argmax_native : bytes -> int -> int = "caml_llmopt_f16_argmax"

  let f16_last_row ?(suppress_token_ids = []) ~vocabulary bytes =
    let ( let* ) = Result.bind in
    let* row = Float16_logits.last_row ~vocabulary bytes in
    let* row =
      Float16_logits.suppress ~vocabulary ~suppress_token_ids row
    in
    try Ok (f16_argmax_native row vocabulary)
    with _ ->
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

  let f16_all_rows ?(suppress_token_ids = []) ~vocabulary bytes =
    let ( let* ) = Result.bind in
    let* _, row_bytes = Float16_logits.row_offset ~vocabulary bytes in
    let rows = Bytes.length bytes / row_bytes in
    let rec collect index output =
      if index = rows then Ok output
      else
        let row = Bytes.sub bytes (index * row_bytes) row_bytes in
        let* token = f16_last_row ~vocabulary ~suppress_token_ids row in
        output.(index) <- token;
        collect (index + 1) output
    in
    collect 0 (Array.make rows 0)
end

external f16_sample_native :
  bytes -> int -> float -> int -> float -> float -> int -> int
  = "caml_llmopt_sample_f16_bytecode" "caml_llmopt_sample_f16_native"

let sample ?(suppress_token_ids = []) ~params ~vocabulary bytes =
  let ( let* ) = Result.bind in
  let* row = Float16_logits.last_row ~vocabulary bytes in
  let* row = Float16_logits.suppress ~vocabulary ~suppress_token_ids row in
  if Params.is_greedy params then
    Greedy.f16_last_row ~vocabulary row
  else
    try
      Ok
        (f16_sample_native row vocabulary params.temperature params.top_k
           params.top_p params.min_p params.seed)
    with _ -> Greedy.f16_last_row ~vocabulary row
