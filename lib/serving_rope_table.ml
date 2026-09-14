let ( let* ) = Result.bind

type t = {
  cosine : Metal_runtime.Buffer.t;
  sine : Metal_runtime.Buffer.t;
  cosine_slot : Metal_runtime.Buffer.t;
  sine_slot : Metal_runtime.Buffer.t;
  positions : int;
  row_bytes : int;
}

let checked_product label factors =
  let rec multiply product = function
    | [] -> Ok product
    | factor :: rest ->
        if factor <= 0 || product > max_int / factor then
          Error (label ^ " size overflows")
        else multiply (product * factor) rest
  in
  multiply 1 factors

let round_f32 value =
  Int32.bits_of_float value |> Int32.float_of_bits

let float32_bits value =
  Int64.logand (Int64.of_int32 (Int32.bits_of_float (round_f32 value)))
    0xffff_ffffL

let round_shift value shift =
  let quotient = Int64.shift_right_logical value shift in
  let remainder =
    Int64.logand value (Int64.sub (Int64.shift_left 1L shift) 1L)
  in
  let halfway = Int64.shift_left 1L (shift - 1) in
  if
    remainder > halfway
    || (remainder = halfway && Int64.logand quotient 1L = 1L)
  then Int64.add quotient 1L
  else quotient

(* IEEE-754 binary32 to binary16, rounded to nearest, ties to even. *)
let float16_bits value =
  let bits = float32_bits value in
  let sign = Int64.logand (Int64.shift_right_logical bits 16) 0x8000L in
  let exponent =
    Int64.logand (Int64.shift_right_logical bits 23) 0xffL
    |> Int64.to_int
  in
  let mantissa = Int64.logand bits 0x7f_ffffL in
  if exponent = 0xff then
    if mantissa = 0L then Int64.logor sign 0x7c00L
    else Int64.logor sign 0x7e00L
  else if exponent > 142 then Int64.logor sign 0x7c00L
  else if exponent >= 113 then
    let half_exponent = exponent - 112 in
    let half_mantissa = round_shift mantissa 13 in
    if half_mantissa = 0x400L then
      if half_exponent = 30 then Int64.logor sign 0x7c00L
      else
        Int64.logor sign
          (Int64.logor
             (Int64.shift_left (Int64.of_int (half_exponent + 1)) 10)
             0L)
    else
      Int64.logor sign
        (Int64.logor
           (Int64.shift_left (Int64.of_int half_exponent) 10)
           half_mantissa)
  else
    let shift = 126 - exponent in
    if shift > 24 then sign
    else
      let mantissa =
        if exponent = 0 then mantissa else Int64.logor mantissa 0x80_0000L
      in
      let half_mantissa = round_shift mantissa shift in
      if half_mantissa >= 0x400L then Int64.logor sign 0x0400L
      else Int64.logor sign half_mantissa

let rope_half_dimension schedule =
  Serving_schedule.commands schedule
  |> List.find_map (fun command ->
         match Serving_schedule.Command.op command with
         | Ir.Op.Rms_rope config ->
             Some (Ir.Rms_rope.half_dimension config)
         | Ir.Op.Rotary_qk { half_dimension; _ } -> Some half_dimension
         | _ -> None)

let inv_freq_input schedule =
  Serving_schedule.tensor_inputs schedule
  |> List.find_opt (fun input ->
         String.ends_with ~suffix:"inv_freq_"
           (Serving_schedule.Tensor_input.key input)
         || String.ends_with ~suffix:"inv_freq"
           (Serving_schedule.Tensor_input.key input))

let create ~runtime ~package ~positions =
  let schedule = Serving_package.schedule package in
  match rope_half_dimension schedule with
  | None -> Ok None
  | Some half_dimension ->
      if positions <= 0 then Error "RoPE position table requires positive capacity"
      else if half_dimension <= 0 || half_dimension > max_int / 2 then
        Error "RoPE half dimension is invalid"
      else
        let width = 2 * half_dimension in
        let* row_bytes = checked_product "RoPE row" [ width; 2 ] in
        let* total_bytes = checked_product "RoPE table" [ positions; row_bytes ] in
        let* input =
          match inv_freq_input schedule with
          | None -> Error "decode package has no rotary inverse-frequency tensor"
          | Some input -> Ok input
        in
        let* inv_freq_bytes =
          checked_product "RoPE inverse-frequency" [ half_dimension; 4 ]
        in
        let value = Serving_schedule.Tensor_input.value input in
        let* () =
          if
            Ir.Value.dtype value = Ir.Dtype.Float32
            && Tensor_shape.dimensions (Ir.Value.logical_shape value)
               = [ half_dimension ]
          then Ok ()
          else Error "rotary inverse-frequency tensor has unexpected metadata"
        in
        let key = Serving_schedule.Tensor_input.key input in
        let* source, tensor = Metal_runtime.tensor runtime ~name:key in
        let* () =
          if
            Weight_archive.Tensor.dtype tensor = Weight_archive.Dtype.F32
            && Weight_archive.Tensor.shape tensor = [ half_dimension ]
            && Weight_archive.Tensor.byte_length tensor = inv_freq_bytes
          then Ok ()
          else Error "rotary inverse-frequency archive tensor has unexpected metadata"
        in
        let* source_bytes = Metal_runtime.Buffer.contents source in
        if Bytes.length source_bytes <> inv_freq_bytes then
          Error "rotary inverse-frequency tensor has an unexpected byte length"
        else
          let inv_freq =
            Array.init half_dimension (fun index ->
                Int32.float_of_bits
                  (Bytes.get_int32_le source_bytes (4 * index)))
          in
          if Array.exists (fun value -> not (Float.is_finite value)) inv_freq then
            Error "rotary inverse-frequency tensor contains a non-finite value"
          else
            let cosine_bytes = Bytes.create total_bytes in
            let sine_bytes = Bytes.create total_bytes in
            for position = 0 to positions - 1 do
              let position_f32 = round_f32 (Float.of_int position) in
              let row_offset = position * row_bytes in
              for index = 0 to half_dimension - 1 do
                let frequency =
                  round_f32 (inv_freq.(index) *. position_f32)
                in
                let cosine = float16_bits (round_f32 (Float.cos frequency)) in
                let sine = float16_bits (round_f32 (Float.sin frequency)) in
                let low_offset = row_offset + (2 * index) in
                let high_offset = row_offset + (2 * (half_dimension + index)) in
                Bytes.set_uint16_le cosine_bytes low_offset (Int64.to_int cosine);
                Bytes.set_uint16_le cosine_bytes high_offset (Int64.to_int cosine);
                Bytes.set_uint16_le sine_bytes low_offset (Int64.to_int sine);
                Bytes.set_uint16_le sine_bytes high_offset (Int64.to_int sine)
              done
            done;
            let* cosine = Metal_runtime.Buffer.of_bytes ~runtime cosine_bytes in
            let* sine = Metal_runtime.Buffer.of_bytes ~runtime sine_bytes in
            let* cosine_slot =
              Metal_runtime.Buffer.create ~runtime ~bytes:row_bytes
            in
            let* sine_slot =
              Metal_runtime.Buffer.create ~runtime ~bytes:row_bytes
            in
            Ok
              (Some
                 { cosine; sine; cosine_slot; sine_slot; positions; row_bytes })

let row table ~position =
  if position < 0 || position >= table.positions then
    Error
      (Printf.sprintf "RoPE position %d is outside [0,%d)" position
         table.positions)
  else
    let offset = position * table.row_bytes in
    let* cosine =
      Metal_runtime.Buffer.view ~parent:table.cosine ~offset
        ~bytes:table.row_bytes
    in
    let* sine =
      Metal_runtime.Buffer.view ~parent:table.sine ~offset
        ~bytes:table.row_bytes
    in
    Ok (cosine, sine)

let slice table ~position ~length =
  if position < 0 || length <= 0 || position + length > table.positions then
    Error
      (Printf.sprintf "RoPE slice [%d,%d) is outside [0,%d)" position
         (position + length) table.positions)
  else
    let offset = position * table.row_bytes in
    let bytes = length * table.row_bytes in
    let* cosine =
      Metal_runtime.Buffer.view ~parent:table.cosine ~offset ~bytes
    in
    let* sine =
      Metal_runtime.Buffer.view ~parent:table.sine ~offset ~bytes
    in
    Ok (cosine, sine)

let slot_for_position table ~position =
  let* cosine, sine = row table ~position in
  let* () =
    Metal_runtime.Buffer.copy ~source:cosine ~destination:table.cosine_slot
  in
  let* () =
    Metal_runtime.Buffer.copy ~source:sine ~destination:table.sine_slot
  in
  Ok (table.cosine_slot, table.sine_slot)
