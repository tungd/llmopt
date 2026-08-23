let fail message =
  prerr_endline message;
  exit 2

let expect_ok = function
  | Ok value -> value
  | Error message -> fail message

let f16_bytes values =
  let bytes = Bytes.create (2 * Array.length values) in
  Array.iteri
    (fun index value -> Bytes.set_uint16_le bytes (2 * index) value)
    values;
  bytes

let f16_values bytes =
  Array.init (Bytes.length bytes / 2) (fun index ->
      Bytes.get_uint16_le bytes (2 * index))

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
  let execution =
    expect_ok (Metal_runtime.execute runtime ~inputs:[ "input", input ])
  in
  let output =
    match Metal_runtime.Execution.output execution ~name:"q8_linear" with
    | Some output -> output
    | None -> fail "OCaml Metal schedule did not produce q8_linear"
  in
  let actual = expect_ok (Metal_runtime.Buffer.contents output) |> f16_values in
  let expected_bits = [| 0x4300; 0x4800; 0x3c00; 0x3e00; 0x4400; 0x4000 |] in
  let expected = [| 3.5; 8.; 1.; 1.5; 4.; 2. |] in
  if Array.length actual <> Array.length expected_bits then
    fail "OCaml Metal output length mismatch";
  Array.iteri
    (fun index expected_value ->
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
  Yojson.Basic.pretty_to_channel stdout
    (`Assoc
      [ ("device", `String (Metal_runtime.device_name runtime));
        ("stage", `String (Serving_package.Stage.to_string (Serving_package.stage package)));
        ("kernel", `String kernel);
        ("shape", `List [ `Int 2; `Int 3; `Int 4 ]);
        ( "output",
          `List
            (Array.to_list (Array.map (fun value -> `Float value) expected)) );
        ("tensor_store", `String "weights.llmopt");
        ("dispatch", `String "ocaml-metal-schedule") ]);
  output_char stdout '\n'
