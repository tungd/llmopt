let fail message =
  prerr_endline message;
  exit 2

let expect_ok = function
  | Ok value -> value
  | Error message -> fail message

let f32_bytes values =
  let bytes = Bytes.create (4 * Array.length values) in
  Array.iteri
    (fun index value ->
      Bytes.set_int32_le bytes (4 * index) (Int32.bits_of_float value))
    values;
  bytes

let f32_values bytes =
  Array.init (Bytes.length bytes / 4) (fun index ->
      Bytes.get_int32_le bytes (4 * index) |> Int32.float_of_bits)

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
  let m, n, k = 2, 3, 4 in
  let input =
    [| 1.; 2.; 3.; 4.;
       2.; 1.; 0.; 1. |]
    |> f32_bytes
    |> Metal_runtime.Buffer.of_bytes ~runtime
    |> expect_ok
  in
  let weight, weight_info =
    expect_ok (Metal_runtime.tensor runtime ~name:"weight_q8")
  in
  let scale, scale_info =
    expect_ok (Metal_runtime.tensor runtime ~name:"weight_scale")
  in
  let bias, bias_info = expect_ok (Metal_runtime.tensor runtime ~name:"bias") in
  if
    Weight_archive.Tensor.dtype weight_info <> Weight_archive.Dtype.I8
    || Weight_archive.Tensor.shape weight_info <> [ n; k ]
    || Weight_archive.Tensor.dtype scale_info <> Weight_archive.Dtype.F16
    || Weight_archive.Tensor.shape scale_info <> [ 1; n ]
    || Weight_archive.Tensor.dtype bias_info <> Weight_archive.Dtype.F16
    || Weight_archive.Tensor.shape bias_info <> [ 1; n ]
  then fail "binary Q8 weight fixture metadata mismatch";
  let output =
    expect_ok (Metal_runtime.Buffer.create ~runtime ~bytes:(m * n * 4))
  in
  let kernel =
    expect_ok
      (Metal_runtime.dispatch_q8_linear runtime ~dtype:Ir.Dtype.Float32
         ~input ~weight ~scale ~bias:(Some bias) ~output ~m ~n ~k)
  in
  let actual =
    expect_ok (Metal_runtime.Buffer.contents output) |> f32_values
  in
  let expected = [| 3.5; 8.; 1.; 1.5; 4.; 2. |] in
  if Array.length actual <> Array.length expected then
    fail "OCaml Metal output length mismatch";
  Array.iteri
    (fun index expected_value ->
      if actual.(index) <> expected_value then
        fail
          (Printf.sprintf
             "OCaml Metal output mismatch at %d: expected=%g actual=%g" index
             expected_value actual.(index)))
    expected;
  Yojson.Basic.pretty_to_channel stdout
    (`Assoc
      [ ("device", `String (Metal_runtime.device_name runtime));
        ("stage", `String (Serving_package.Stage.to_string (Serving_package.stage package)));
        ("kernel", `String kernel);
        ("shape", `List [ `Int m; `Int n; `Int k ]);
        ("output", `List (Array.to_list (Array.map (fun value -> `Float value) actual)));
        ("tensor_store", `String "weights.llmopt");
        ("dispatch", `String "ocaml-metal-direct") ]);
  output_char stdout '\n'
