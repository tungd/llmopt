let fail message =
  prerr_endline message;
  exit 2

let expect_ok = function
  | Ok value -> value
  | Error message -> fail message

let ( let* ) = Result.bind

let f16_bytes values =
  let bytes = Bytes.create (2 * Array.length values) in
  Array.iteri
    (fun index value -> Bytes.set_uint16_le bytes (2 * index) value)
    values;
  bytes

let f16_values bytes =
  Array.init (Bytes.length bytes / 2) (fun index ->
      Bytes.get_uint16_le bytes (2 * index))

let expect_kv = function
  | Ok value -> value
  | Error error -> fail (Kv_cache.error_to_string error)

let expect_buffer label expected buffer =
  let actual = expect_ok (Metal_runtime.Buffer.contents buffer) in
  if not (Bytes.equal actual expected) then
    let rec first_mismatch index =
      if index = Bytes.length expected || index = Bytes.length actual then index
      else if Bytes.get expected index <> Bytes.get actual index then index
      else first_mismatch (index + 1)
    in
    let index = first_mismatch 0 in
    let byte bytes =
      if index < Bytes.length bytes then
        Printf.sprintf "0x%02x" (Char.code (Bytes.get bytes index))
      else "end-of-buffer"
    in
    fail
      (Printf.sprintf
         "%s physical-cache round trip mismatch at byte %d: expected=%s actual=%s"
         label index (byte expected) (byte actual))

let cache_values count =
  Array.init count (fun index ->
      match index mod 4 with
      | 0 -> 0x57f0
      | 1 -> 0xd7f0
      | 2 -> 0x3c00
      | _ -> 0xbc00)

let opposite_cache_values values =
  Array.map
    (function
      | 0x57f0 -> 0xd7f0
      | 0xd7f0 -> 0x57f0
      | 0x3c00 -> 0xbc00
      | _ -> 0x3c00)
    values

let physical_cache_config format =
  let layout =
    Kv_cache.Layout.create ~format ~attention_layers:1 ~kv_heads:1
      ~head_dim:64 ~recurrent_layers:1 ~recurrent_width:64
      ~recurrent_window:1
    |> expect_ok
  in
  Kv_cache.Config.create ~layout ~token_capacity:4 ~checkpoint_capacity:2
  |> expect_ok

let exercise_cache runtime format =
  let config = physical_cache_config format in
  let ownership = Kv_cache.create config in
  let slots = Kv_cache.reserve_tokens ownership 2 |> expect_kv in
  let checkpoint = Kv_cache.reserve_checkpoint ownership |> expect_kv in
  let cache = Metal_runtime.Cache.create ~runtime ~config |> expect_ok in
  let key_values = cache_values 128 in
  let value_values = opposite_cache_values key_values in
  let checkpoint_values = cache_values 64 in
  let key_bytes = f16_bytes key_values in
  let value_bytes = f16_bytes value_values in
  let checkpoint_bytes = f16_bytes checkpoint_values in
  let buffer contents =
    Metal_runtime.Buffer.of_bytes ~runtime contents |> expect_ok
  in
  let destination bytes =
    Metal_runtime.Buffer.create ~runtime ~bytes |> expect_ok
  in
  let key_source = buffer key_bytes in
  let value_source = buffer value_bytes in
  let checkpoint_source = buffer checkpoint_bytes in
  let key_destination = destination (Bytes.length key_bytes) in
  let value_destination = destination (Bytes.length value_bytes) in
  let checkpoint_destination = destination (Bytes.length checkpoint_bytes) in
  let kernels =
    Metal_runtime.Cache.with_batch cache (fun batch ->
        let* pack_key =
          Metal_runtime.Cache.batch_pack_attention batch ~layer:0
            ~kind:Metal_runtime.Cache.Attention.Key ~slots ~source:key_source
        in
        let* pack_value =
          Metal_runtime.Cache.batch_pack_attention batch ~layer:0
            ~kind:Metal_runtime.Cache.Attention.Value ~slots ~source:value_source
        in
        let* unpack_key =
          Metal_runtime.Cache.batch_unpack_attention batch ~layer:0
            ~kind:Metal_runtime.Cache.Attention.Key ~slots
            ~destination:key_destination
        in
        let* unpack_value =
          Metal_runtime.Cache.batch_unpack_attention batch ~layer:0
            ~kind:Metal_runtime.Cache.Attention.Value ~slots
            ~destination:value_destination
        in
        let* pack_checkpoint =
          Metal_runtime.Cache.batch_pack_checkpoint batch ~layer:0 ~checkpoint
            ~source:checkpoint_source
        in
        let* unpack_checkpoint =
          Metal_runtime.Cache.batch_unpack_checkpoint batch ~layer:0
            ~checkpoint ~destination:checkpoint_destination
        in
        Ok
          [ pack_key; pack_value; unpack_key; unpack_value; pack_checkpoint;
            unpack_checkpoint ])
    |> expect_ok
  in
  expect_buffer (Kv_cache.Format.to_string format ^ " attention key") key_bytes
    key_destination;
  expect_buffer (Kv_cache.Format.to_string format ^ " attention value")
    value_bytes value_destination;
  expect_buffer (Kv_cache.Format.to_string format ^ " recurrent checkpoint")
    checkpoint_bytes checkpoint_destination;
  ( kernels,
    Metal_runtime.Cache.token_pool_bytes cache,
    Metal_runtime.Cache.checkpoint_pool_bytes cache )

let usage () =
  prerr_endline "usage: llmopt-ocaml-metal-smoke <package-directory>";
  exit 64

let () =
  if Array.length Sys.argv <> 2 then usage ();
  let root = Sys.argv.(1) in
  let package =
    expect_ok
      (Serving_package.of_file (Filename.concat root "package.llmopt"))
  in
  let runtime = expect_ok (Metal_runtime.load_package ~root package) in
  let input =
    [| 0x3c00; 0x4000; 0x4200; 0x4400;
       0x4000; 0x3c00; 0x0000; 0x3c00 |]
    |> f16_bytes
    |> Metal_runtime.Buffer.of_bytes ~runtime
    |> expect_ok
  in
  let first_execution, execution =
    Metal_runtime.with_execution_batch runtime (fun batch ->
        let schedule = Serving_package.schedule package in
        let* first =
          Metal_runtime.encode_schedule batch ~schedule
            ~inputs:[ "input", input ]
        in
        let* second =
          Metal_runtime.encode_schedule batch ~schedule
            ~inputs:[ "input", input ]
        in
        Ok (first, second))
    |> expect_ok
  in
  let output =
    match Metal_runtime.Execution.output execution ~name:"q8_linear" with
    | Some output -> output
    | None -> fail "OCaml Metal schedule did not produce q8_linear"
  in
  let actual = expect_ok (Metal_runtime.Buffer.contents output) |> f16_values in
  let expected_bits = [| 0x4300; 0x4800; 0x3c00; 0x3e00; 0x4400; 0x4000 |] in
  let first_output =
    match Metal_runtime.Execution.output first_execution ~name:"q8_linear" with
    | Some output -> output
    | None -> fail "first batched Metal schedule did not produce q8_linear"
  in
  let first_actual =
    expect_ok (Metal_runtime.Buffer.contents first_output) |> f16_values
  in
  if
    Array.length first_actual <> Array.length expected_bits
    || Array.length actual <> Array.length expected_bits
  then
    fail "OCaml Metal output length mismatch";
  Array.iteri
    (fun index expected_value ->
      if first_actual.(index) <> expected_value then
        fail
          (Printf.sprintf
             "first OCaml Metal output mismatch at %d: expected=0x%04x actual=0x%04x"
             index expected_value first_actual.(index));
      if actual.(index) <> expected_value then
        fail
          (Printf.sprintf
             "OCaml Metal output mismatch at %d: expected=0x%04x actual=0x%04x"
             index expected_value actual.(index)))
    expected_bits;
  let kernel =
    match Metal_runtime.Execution.kernels execution with
    | [ kernel ] -> kernel
    | kernels ->
        fail
          (Printf.sprintf "OCaml Metal schedule dispatched %d kernels"
             (List.length kernels))
  in
  let q8_kernels, q8_token_bytes, q8_checkpoint_bytes =
    exercise_cache runtime Kv_cache.Format.default
  in
  let q8_cache_kernels =
    match q8_kernels with
    | [ pack_key; pack_value; unpack_key; unpack_value; pack_checkpoint;
        unpack_checkpoint ] ->
        let attention = "llmopt_cache_pack_attention_q8_simd" in
        let checkpoint = "llmopt_cache_pack_checkpoint_q8_simd" in
        let attention_unpack = "llmopt_cache_unpack_attention_q8_vec4" in
        let checkpoint_unpack = "llmopt_cache_unpack_checkpoint_q8_vec4" in
        if pack_key <> attention || pack_value <> attention then
          fail "Q8 attention cache did not select the SIMD pack kernel";
        if pack_checkpoint <> checkpoint then
          fail "Q8 checkpoint cache did not select the SIMD pack kernel";
        if unpack_key <> attention_unpack || unpack_value <> attention_unpack then
          fail "Q8 attention cache did not select the vector unpack kernel";
        if unpack_checkpoint <> checkpoint_unpack then
          fail "Q8 checkpoint cache did not select the vector unpack kernel";
        String.concat ","
          [ attention; checkpoint; attention_unpack; checkpoint_unpack ]
    | _ -> fail "Q8 physical cache did not dispatch six kernels"
  in
  let f16_kernels, f16_token_bytes, f16_checkpoint_bytes =
    exercise_cache runtime Kv_cache.Format.f16
  in
  if List.length q8_kernels + List.length f16_kernels <> 12 then
    fail "physical cache did not dispatch twelve pack/unpack kernels";
  Printf.printf
    "device: %s\nstage: %s\ndispatch: ocaml-metal-schedule\nkernel: %s\nschedule-dispatches: 2\nschedule-submissions: 1\ncache-formats: q8-group-64,f16\nq8-vector-kernels: %s\ncache-dispatches: 12\ncache-submissions: 2\nq8-pools: %d token bytes, %d checkpoint bytes\nf16-pools: %d token bytes, %d checkpoint bytes\nattention: exact\ncheckpoint: exact\n"
    (Metal_runtime.device_name runtime)
    (Serving_package.Stage.to_string (Serving_package.stage package))
    kernel q8_cache_kernels q8_token_bytes q8_checkpoint_bytes f16_token_bytes
    f16_checkpoint_bytes
