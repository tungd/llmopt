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

let execute_once runtime input_name input output_name =
  let started_at = Unix.gettimeofday () in
  let* execution = Metal_runtime.execute runtime ~inputs:[ input_name, input ] in
  let* output =
    match Metal_runtime.Execution.output execution ~name:output_name with
    | Some output -> Ok output
    | None -> Error ("package did not produce output: " ^ output_name)
  in
  let* contents = Metal_runtime.Buffer.contents output in
  let elapsed_ms = (Unix.gettimeofday () -. started_at) *. 1000.0 in
  Ok (execution, contents, elapsed_ms)

let rec warmup runtime input_name input output_name remaining =
  if remaining = 0 then Ok ()
  else
    let* _ = execute_once runtime input_name input output_name in
    warmup runtime input_name input output_name (remaining - 1)

let rec measure runtime input_name input output_name remaining measurements last =
  if remaining = 0 then Ok (List.rev measurements, last)
  else
    let* execution, contents, elapsed_ms =
      execute_once runtime input_name input output_name
    in
    measure runtime input_name input output_name (remaining - 1)
      (elapsed_ms :: measurements) (Some (execution, contents))

let median values =
  let sorted = List.sort Float.compare values |> Array.of_list in
  let length = Array.length sorted in
  if length mod 2 = 1 then sorted.(length / 2)
  else (sorted.((length / 2) - 1) +. sorted.(length / 2)) /. 2.0

let run ~warmup_count ~repeat_count root input_name input_path output_name
    output_path =
  let* package =
    Serving_package.of_file (Filename.concat root "package.llmopt")
  in
  let* runtime = Metal_runtime.load_package ~root package in
  let* input_bytes = read_file input_path in
  let* input = Metal_runtime.Buffer.of_bytes ~runtime input_bytes in
  let* () = warmup runtime input_name input output_name warmup_count in
  let* measurements, last =
    measure runtime input_name input output_name repeat_count [] None
  in
  let execution, contents = Option.get last in
  let* () = write_file output_path contents in
  Printf.printf
    "device=%s dispatch_count=%d output_bytes=%d warmup=%d repeat=%d median_ms=%.6f min_ms=%.6f max_ms=%.6f\n"
    (Metal_runtime.device_name runtime)
    (List.length (Metal_runtime.Execution.kernels execution))
    (Bytes.length contents) warmup_count repeat_count (median measurements)
    (List.fold_left Float.min Float.infinity measurements)
    (List.fold_left Float.max Float.neg_infinity measurements);
  Ok ()

let count ~name ~minimum raw =
  let value = int_of_string raw in
  if value < minimum then
    invalid_arg (Printf.sprintf "%s must be at least %d" name minimum);
  value

let usage () =
  prerr_endline
    "usage: llmopt-package-run --warmup <count> --repeat <count> \
     <package-directory> <input-name> <input.bin> <output-name> <output.bin>";
  exit 64

let () =
  try
    let raw_warmup, raw_repeat, root, input_name, input_path, output_name,
        output_path =
      match Array.to_list Sys.argv with
      | [ _; "--warmup"; raw_warmup; "--repeat"; raw_repeat; root;
          input_name; input_path; output_name; output_path ] ->
          (raw_warmup, raw_repeat, root, input_name, input_path, output_name,
           output_path)
      | _ -> usage ()
    in
    let warmup_count = count ~name:"warmup" ~minimum:0 raw_warmup in
    let repeat_count = count ~name:"repeat" ~minimum:1 raw_repeat in
    match
      run ~warmup_count ~repeat_count root input_name input_path output_name
        output_path
    with
    | Ok () -> ()
    | Error message ->
        prerr_endline message;
        exit 2
  with Invalid_argument message ->
    prerr_endline message;
    exit 64
