let ( let* ) = Result.bind

let read_file path =
  try
    let channel = open_in_bin path in
    Fun.protect
      ~finally:(fun () -> close_in_noerr channel)
      (fun () ->
        really_input_string channel (in_channel_length channel)
        |> Bytes.of_string |> Result.ok)
  with Sys_error message -> Error message

let write_file path contents =
  try
    let channel = open_out_bin path in
    Fun.protect
      ~finally:(fun () -> close_out_noerr channel)
      (fun () -> output_bytes channel contents);
    Ok ()
  with Sys_error message -> Error message

let run root input_name input_path output_name output_path =
  let* package =
    Serving_package.of_file (Filename.concat root "package.llmopt")
  in
  let* runtime = Metal_runtime.load_package ~root package in
  let* input_bytes = read_file input_path in
  let* input = Metal_runtime.Buffer.of_bytes ~runtime input_bytes in
  let* execution =
    Metal_runtime.execute runtime ~inputs:[ input_name, input ]
  in
  let* output =
    match Metal_runtime.Execution.output execution ~name:output_name with
    | Some output -> Ok output
    | None -> Error ("package did not produce output: " ^ output_name)
  in
  let* contents = Metal_runtime.Buffer.contents output in
  let* () = write_file output_path contents in
  Printf.printf "device=%s kernels=%s output_bytes=%d\n"
    (Metal_runtime.device_name runtime)
    (String.concat "," (Metal_runtime.Execution.kernels execution))
    (Bytes.length contents);
  Ok ()

let usage () =
  prerr_endline
    "usage: llmopt-package-run <package-directory> <input-name> <input.bin> \
     <output-name> <output.bin>";
  exit 64

let () =
  match Array.to_list Sys.argv with
  | [ _; root; input_name; input_path; output_name; output_path ] ->
      (match run root input_name input_path output_name output_path with
      | Ok () -> ()
      | Error message ->
          prerr_endline message;
          exit 2)
  | _ -> usage ()
