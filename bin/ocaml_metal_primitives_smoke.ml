let fail message =
  prerr_endline message;
  exit 2

let expect_ok = function Ok value -> value | Error message -> fail message

let bytes_of_u16 values =
  let bytes = Bytes.create (2 * List.length values) in
  List.iteri (fun index value -> Bytes.set_uint16_le bytes (2 * index) value) values;
  bytes

let bytes_of_f32 values =
  let bytes = Bytes.create (4 * List.length values) in
  List.iteri
    (fun index value ->
      Bytes.set_int32_le bytes (4 * index) (Int32.bits_of_float value))
    values;
  bytes

let bytes_of_i64 values =
  let bytes = Bytes.create (8 * List.length values) in
  List.iteri
    (fun index value -> Bytes.set_int64_le bytes (8 * index) (Int64.of_int value))
    values;
  bytes

let bytes_of_i8 values =
  values
  |> List.map (fun value -> Char.chr (value land 0xff))
  |> List.to_seq |> Bytes.of_seq

let bytes_of_bool values =
  values |> List.map (fun value -> if value then '\001' else '\000')
  |> List.to_seq |> Bytes.of_seq

let input runtime name contents =
  name, Metal_runtime.Buffer.of_bytes ~runtime contents |> expect_ok

let output execution name =
  match Metal_runtime.Execution.output execution ~name with
  | Some buffer -> Metal_runtime.Buffer.contents buffer |> expect_ok
  | None -> fail ("native fixture did not produce output: " ^ name)

let hex bytes =
  let result = Buffer.create (2 * Bytes.length bytes) in
  Bytes.iter (fun byte -> Printf.bprintf result "%02x" (Char.code byte)) bytes;
  Buffer.contents result

let expect_bytes execution name expected =
  let actual = output execution name in
  if not (Bytes.equal actual expected) then
    fail
      (Printf.sprintf "%s mismatch: expected=%s actual=%s" name
         (hex expected) (hex actual))

let usage () =
  prerr_endline "usage: llmopt-ocaml-metal-primitives-smoke <package-directory>";
  exit 64

let () =
  if Array.length Sys.argv <> 2 then usage ();
  let root = Sys.argv.(1) in
  let package =
    Serving_package.of_file (Filename.concat root "package.llmopt") |> expect_ok
  in
  let runtime = Metal_runtime.load_package ~root package |> expect_ok in
  let execution =
    Metal_runtime.execute runtime
      ~inputs:
        [ input runtime "embedding_indices" (bytes_of_i64 [ 2; 0 ]);
          input runtime "embedding_weight"
            (bytes_of_u16 [ 0x3c00; 0x4000; 0x4200; 0x4400; 0x4500; 0x4600 ]);
          input runtime "diff_source" (bytes_of_i64 [ 0; 0; 2 ]);
          input runtime "diff_prepend" (bytes_of_i64 [ -1 ]);
          input runtime "packed" (bytes_of_bool [ true; false; true ]);
          input runtime "gather_source"
            (bytes_of_i64 [ 10; 11; 12; 20; 21; 22 ]);
          input runtime "first_index" (bytes_of_i64 [ 1 ]);
          input runtime "second_index" (bytes_of_i64 [ 2; 0; 1 ]);
          input runtime "rms_input"
            (bytes_of_f32 [ 1.; 0.; 0.; 0.; 0.; 1.; 0.; 0. ]);
          input runtime "rms_weight"
            (bytes_of_u16 [ 0x3c00; 0x3c00; 0x3c00; 0x3c00 ]);
          input runtime "short_input"
            (bytes_of_u16
               [ 0x3c00; 0x4000; 0x4200; 0x4400; 0x4500; 0x4600; 0x4700;
                 0x4800 ]);
          input runtime "short_weight"
            (bytes_of_u16 [ 0x3c00; 0x0000; 0xbc00; 0x3800; 0x3c00; 0x3800 ]);
          input runtime "attention_query"
            (bytes_of_u16 [ 0x3c00; 0x0000; 0x0000; 0x3c00 ]);
          input runtime "attention_key"
            (bytes_of_u16 [ 0x3c00; 0x0000; 0x0000; 0x3c00 ]);
          input runtime "attention_value"
            (bytes_of_u16 [ 0x4900; 0x0000; 0x0000; 0x4d00 ]);
          input runtime "attention_mask"
            (bytes_of_bool [ true; false; false; true ]);
          input runtime "cast_f16_input"
            (bytes_of_u16 [ 0x3c00; 0xc000; 0x3800 ]);
          input runtime "cast_f32_input" (bytes_of_f32 [ 1.; -2.; 0.5 ]);
          input runtime "cast_i64_input" (bytes_of_i64 [ 0; -3; 7 ]);
          input runtime "add_i64_input" (bytes_of_i64 [ 0; 1; 2 ]);
          input runtime "add_f16_left" (bytes_of_u16 [ 0x3c00; 0x4000 ]);
          input runtime "add_f16_right"
            (bytes_of_u16 [ 0x3c00; 0x4000; 0x4200 ]);
          input runtime "mul_f16_left" (bytes_of_u16 [ 0x3c00; 0x4000 ]);
          input runtime "mul_f16_right"
            (bytes_of_u16 [ 0x3c00; 0x4000; 0x4200 ]);
          input runtime "mul_f32_input" (bytes_of_f32 [ 2.; -4.; 0.5 ]);
          input runtime "le_i64_left" (bytes_of_i64 [ 1; 3 ]);
          input runtime "le_i64_right" (bytes_of_i64 [ 2; 1; 3 ]);
          input runtime "neg_f16_input"
            (bytes_of_u16 [ 0x3c00; 0xc000; 0x3800 ]);
          input runtime "silu_f16_input" (bytes_of_u16 [ 0x0000 ]);
          input runtime "cos_f32_input" (bytes_of_f32 [ 0. ]);
          input runtime "sin_f32_input" (bytes_of_f32 [ 0. ]);
          input runtime "transpose_f16_input"
            (bytes_of_u16 [ 0x3c00; 0x4000; 0x4200; 0x4400; 0x4500; 0x4600 ]);
          input runtime "transpose_f32_input"
            (bytes_of_f32 [ 1.; 2.; 3.; 4.; 5.; 6. ]);
          input runtime "index_f16_input"
            (bytes_of_u16 [ 0x3c00; 0x4000; 0x4200; 0x4400; 0x4500; 0x4600 ]);
          input runtime "index_f32_input"
            (bytes_of_f32 [ 1.; 2.; 3.; 4.; 5.; 6. ]);
          input runtime "index_i64_input" (bytes_of_i64 [ 1; 2; 3; 4; 5; 6 ]);
          input runtime "expand_f16_input" (bytes_of_u16 [ 0x3c00; 0x4000 ]);
          input runtime "expand_f32_input" (bytes_of_f32 [ 2.5 ]);
          input runtime "expand_bool_input" (bytes_of_bool [ true; false ]);
          input runtime "concat_f16_left" (bytes_of_u16 [ 0x3c00; 0x4000 ]);
          input runtime "concat_f16_right" (bytes_of_u16 [ 0x4200 ]);
          input runtime "concat_f32_left" (bytes_of_f32 [ 1.; 4. ]);
          input runtime "concat_f32_right" (bytes_of_f32 [ 2.; 3.; 5.; 6. ]);
          input runtime "roll_f16_input"
            (bytes_of_u16 [ 0x3c00; 0x4000; 0x4200; 0x4400 ]);
          input runtime "sum_f16_input"
            (bytes_of_u16 [ 0x3c00; 0x4000; 0x4200; 0x4400; 0x4500; 0x4600 ]);
          input runtime "update_destination"
            (bytes_of_u16
               [ 0x3c00; 0x4000; 0x4200; 0x4400; 0x4500; 0x4600; 0x4700;
                 0x4800 ]);
          input runtime "update_source"
            (bytes_of_u16 [ 0x4900; 0x4980; 0x4a00; 0x4a80 ]);
          input runtime "linear_f16_input"
            (bytes_of_u16
               [ 0x3c00; 0x4000; 0x4200; 0x4400; 0x4000; 0x3c00; 0x0000;
                 0x3c00 ]);
          input runtime "linear_f16_weight"
            (bytes_of_u16
               [ 0x3c00; 0x0000; 0x4000; 0xbc00; 0x0000; 0x3c00; 0xbc00;
                 0x4000; 0x4000; 0xc000; 0x0000; 0x3c00 ]);
          input runtime "q8_gemv_input"
            (bytes_of_u16 [ 0x3c00; 0x4000; 0x4200; 0x4400 ]);
          input runtime "q8_gemv_weight"
            (bytes_of_i8 [ 1; 0; 2; -1; 0; 1; -1; 2; 2; -2; 0; 1 ]);
          input runtime "q8_gemv_scale"
            (bytes_of_u16 [ 0x3800; 0x3c00; 0xb400 ]);
          input runtime "q8_gemv_bias"
            (bytes_of_u16 [ 0x3800; 0x3c00; 0xbc00 ]);
          input runtime "q8_add_residual"
            (bytes_of_u16 [ 0x3c00; 0xbc00; 0x3800 ]);
          input runtime "q8_mul_weight"
            (bytes_of_i8 [ 1; 0; 0; 0; 1; 0 ]);
          input runtime "q8_mul_scale" (bytes_of_u16 [ 0x3c00; 0x3c00 ]);
          input runtime "q8_mul_residual"
            (bytes_of_u16 [ 0x3c00; 0xbc00 ]);
          input runtime "matmul_lhs" (bytes_of_f32 [ 1.; 2.; 3.; 4.; 5.; 6. ]);
          input runtime "matmul_rhs" (bytes_of_f32 [ 1.; 2.; 0.; 1.; -1.; 0. ]) ]
    |> expect_ok
  in
  expect_bytes execution "embedding"
    (bytes_of_u16 [ 0x4500; 0x4600; 0x3c00; 0x4000 ]);
  expect_bytes execution "arange" (bytes_of_i64 [ 5; 3; 1 ]);
  expect_bytes execution "fill_bool" (bytes_of_bool [ true ]);
  expect_bytes execution "fill_f16" (bytes_of_u16 [ 0x3e00; 0x3e00; 0x3e00 ]);
  expect_bytes execution "fill_f32" (bytes_of_f32 [ -2.25; -2.25 ]);
  expect_bytes execution "diff" (bytes_of_i64 [ 1; 0; 2 ]);
  expect_bytes execution "cumsum" (bytes_of_i64 [ 1; 1; 2 ]);
  expect_bytes execution "gather2" (bytes_of_i64 [ 22; 20; 21 ]);
  expect_bytes execution "rms_norm"
    (bytes_of_u16 [ 0x4000; 0; 0; 0; 0; 0x4000; 0; 0 ]);
  expect_bytes execution "short_conv"
    (bytes_of_u16
       [ 0xbc00; 0xc000; 0xc000; 0xc000; 0x4200; 0x4400; 0x4100; 0x4800;
         0x4a00; 0x4b00; 0x49c0; 0x4400 ]);
  expect_bytes execution "attention"
    (bytes_of_u16 [ 0x4900; 0; 0; 0x4d00 ]);
  expect_bytes execution "cast_f16_f32" (bytes_of_f32 [ 1.; -2.; 0.5 ]);
  expect_bytes execution "cast_f32_f16"
    (bytes_of_u16 [ 0x3c00; 0xc000; 0x3800 ]);
  expect_bytes execution "cast_i64_f32" (bytes_of_f32 [ 0.; -3.; 7. ]);
  expect_bytes execution "add_i64" (bytes_of_i64 [ 6; 7; 8 ]);
  expect_bytes execution "add_f16"
    (bytes_of_u16 [ 0x4000; 0x4200; 0x4400; 0x4200; 0x4400; 0x4500 ]);
  expect_bytes execution "mul_f16"
    (bytes_of_u16 [ 0x3c00; 0x4000; 0x4200; 0x4000; 0x4400; 0x4600 ]);
  expect_bytes execution "mul_f32" (bytes_of_f32 [ 3.; -6.; 0.75 ]);
  expect_bytes execution "le_i64"
    (bytes_of_bool [ true; true; true; false; false; true ]);
  expect_bytes execution "neg_f16"
    (bytes_of_u16 [ 0xbc00; 0x4000; 0xb800 ]);
  expect_bytes execution "silu_f16" (bytes_of_u16 [ 0x0000 ]);
  expect_bytes execution "cos_f32" (bytes_of_f32 [ 1. ]);
  expect_bytes execution "sin_f32" (bytes_of_f32 [ 0. ]);
  expect_bytes execution "transpose_f16"
    (bytes_of_u16 [ 0x3c00; 0x4400; 0x4000; 0x4500; 0x4200; 0x4600 ]);
  expect_bytes execution "contiguous_f16"
    (bytes_of_u16 [ 0x3c00; 0x4400; 0x4000; 0x4500; 0x4200; 0x4600 ]);
  expect_bytes execution "transpose_f32" (bytes_of_f32 [ 1.; 4.; 2.; 5.; 3.; 6. ]);
  expect_bytes execution "index_f16" (bytes_of_u16 [ 0x4400; 0x4600 ]);
  expect_bytes execution "index_f32" (bytes_of_f32 [ 2.; 5. ]);
  expect_bytes execution "index_i64" (bytes_of_i64 [ 4; 6; 1; 3 ]);
  expect_bytes execution "expand_f16"
    (bytes_of_u16 [ 0x3c00; 0x3c00; 0x3c00; 0x4000; 0x4000; 0x4000 ]);
  expect_bytes execution "expand_f32" (bytes_of_f32 [ 2.5; 2.5; 2.5; 2.5 ]);
  expect_bytes execution "expand_bool"
    (bytes_of_bool [ true; false; true; false ]);
  expect_bytes execution "concat_f16"
    (bytes_of_u16 [ 0x3c00; 0x4000; 0x4200 ]);
  expect_bytes execution "concat_f32" (bytes_of_f32 [ 1.; 2.; 3.; 4.; 5.; 6. ]);
  expect_bytes execution "roll_f16"
    (bytes_of_u16 [ 0x4000; 0x4200; 0x4400; 0x3c00 ]);
  expect_bytes execution "sum_f16" (bytes_of_u16 [ 0x4600; 0x4b80 ]);
  expect_bytes execution "update_slice_f16"
    (bytes_of_u16
       [ 0x3c00; 0x4900; 0x4980; 0x4400; 0x4500; 0x4a00; 0x4a80; 0x4800 ]);
  expect_bytes execution "linear_f16"
    (bytes_of_u16 [ 0x4200; 0x4700; 0x4000; 0x3c00; 0x4200; 0x4200 ]);
  expect_bytes execution "q8_gemv"
    (bytes_of_u16 [ 0x4000; 0x4800; 0xbe00 ]);
  expect_bytes execution "q8_gemv_silu"
    (bytes_of_u16 [ 0x3f0c; 0x47ff; 0xb461 ]);
  expect_bytes execution "q8_gemv_silu_reference"
    (bytes_of_u16 [ 0x3f0c; 0x47ff; 0xb461 ]);
  let q8_fused = output execution "q8_gemv_silu" in
  let q8_reference = output execution "q8_gemv_silu_reference" in
  if not (Bytes.equal q8_fused q8_reference) then
    fail "fused Q8 SiLU differs from materialized Q8 plus standalone SiLU";
  expect_bytes execution "q8_gemv_add"
    (bytes_of_u16 [ 0x4200; 0x4700; 0xbc00 ]);
  expect_bytes execution "q8_gemv_add_reference"
    (bytes_of_u16 [ 0x4200; 0x4700; 0xbc00 ]);
  let q8_add_fused = output execution "q8_gemv_add" in
  let q8_add_reference = output execution "q8_gemv_add_reference" in
  if not (Bytes.equal q8_add_fused q8_add_reference) then
    fail "fused Q8 residual differs from materialized Q8 plus standalone add";
  expect_bytes execution "q8_gemv_mul_add"
    (bytes_of_u16 [ 0x4486; 0x53df ]);
  expect_bytes execution "q8_gemv_mul_add_reference"
    (bytes_of_u16 [ 0x4486; 0x53df ]);
  let q8_mul_add_fused = output execution "q8_gemv_mul_add" in
  let q8_mul_add_reference = output execution "q8_gemv_mul_add_reference" in
  if not (Bytes.equal q8_mul_add_fused q8_mul_add_reference) then
    fail
      "fused Q8 multiplied input differs from materialized multiply and residual";
  expect_bytes execution "matmul" (bytes_of_f32 [ -2.; 4.; -2.; 13. ]);
  let kernels = Metal_runtime.Execution.kernels execution in
  if List.length kernels <> 47 then
    fail
      (Printf.sprintf "native fixture dispatched %d kernels instead of 47"
         (List.length kernels));
  let workspace_bytes = Metal_runtime.Execution.workspace_bytes execution in
  if workspace_bytes <> 11_520 then
    fail
      (Printf.sprintf "native fixture workspace is %d bytes instead of 11520"
         workspace_bytes);
  Printf.printf
    "device: %s\ndispatch: binary-schedule\ncommands: %d\nkernels: %d\nworkspace: %d bytes\noutputs: 46 exact\nq8-decode-kernel: llmopt_q8_gemv_simd\nq8-silu-reference: exact\nq8-silu-decode-kernel: llmopt_q8_gemv_silu_simd\nq8-add-reference: exact\nq8-add-decode-kernel: llmopt_q8_gemv_add_simd\nq8-mul-add-reference: exact\nq8-mul-add-decode-kernel: llmopt_q8_gemv_mul_add_simd\n"
    (Metal_runtime.device_name runtime)
    (Serving_package.schedule package |> Serving_schedule.commands |> List.length)
    (List.length kernels) workspace_bytes
