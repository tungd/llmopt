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

type prepared_execution = {
  runtime : Metal_runtime.t;
  schedule : Serving_schedule.t;
  memory_plan : Serving_memory_plan.t;
  workspace : Metal_runtime.Buffer.t option;
}

let prepare_execution runtime =
  let schedule =
    Metal_runtime.package runtime |> Serving_package.schedule
  in
  let* memory_plan = Serving_memory_plan.create schedule in
  let workspace_bytes = Serving_memory_plan.workspace_bytes memory_plan in
  let* workspace =
    if workspace_bytes = 0 then Ok None
    else
      Metal_runtime.Buffer.create ~runtime ~bytes:workspace_bytes
      |> Result.map Option.some
  in
  Ok { runtime; schedule; memory_plan; workspace }

let execute_once prepared input_name input =
  let started_at = Unix.gettimeofday () in
  let* execution =
    Metal_runtime.execute_schedule ?workspace:prepared.workspace
      ~memory_plan:prepared.memory_plan prepared.runtime
      ~schedule:prepared.schedule ~inputs:[ input_name, input ]
  in
  let elapsed_ms = (Unix.gettimeofday () -. started_at) *. 1000.0 in
  Ok (execution, elapsed_ms)

let rec warmup prepared input_name input remaining =
  if remaining = 0 then Ok ()
  else
    let* _ = execute_once prepared input_name input in
    warmup prepared input_name input (remaining - 1)

let rec measure prepared input_name input remaining measurements last =
  if remaining = 0 then Ok (List.rev measurements, last)
  else
    let* execution, elapsed_ms = execute_once prepared input_name input in
    measure prepared input_name input (remaining - 1)
      (elapsed_ms :: measurements) (Some execution)

let median values =
  let sorted = List.sort Float.compare values |> Array.of_list in
  let length = Array.length sorted in
  if length mod 2 = 1 then sorted.(length / 2)
  else (sorted.((length / 2) - 1) +. sorted.(length / 2)) /. 2.0

let dump_kernel_histogram execution =
  match Sys.getenv_opt "LLMOPT_DUMP_KERNELS" with
  | Some "1" ->
      let counts = Hashtbl.create 64 in
      Metal_runtime.Execution.kernels execution
      |> List.iter (fun name ->
             Hashtbl.replace counts name
               (Option.value ~default:0 (Hashtbl.find_opt counts name) + 1));
      counts |> Hashtbl.to_seq |> List.of_seq
      |> List.sort (fun (left_name, left_count) (right_name, right_count) ->
             let by_count = Int.compare right_count left_count in
             if by_count <> 0 then by_count
             else String.compare left_name right_name)
      |> List.iter (fun (name, count) -> Printf.printf "%6d %s\n" count name)
  | _ -> ()

let run ~warmup_count ~repeat_count root input_name input_path output_name
    output_path =
  let* package =
    Serving_package.of_file (Filename.concat root "package.llmopt")
  in
  let* runtime = Metal_runtime.load_package ~root package in
  let* prepared = prepare_execution runtime in
  let* input_bytes = read_file input_path in
  let* input = Metal_runtime.Buffer.of_bytes ~runtime input_bytes in
  let* () = warmup prepared input_name input warmup_count in
  let* measurements, last =
    measure prepared input_name input repeat_count [] None
  in
  let execution = Option.get last in
  let* output =
    match Metal_runtime.Execution.output execution ~name:output_name with
    | Some output -> Ok output
    | None -> Error ("package did not produce output: " ^ output_name)
  in
  let* contents = Metal_runtime.Buffer.contents output in
  let* () = write_file output_path contents in
  Printf.printf
    "device=%s dispatch_count=%d output_bytes=%d warmup=%d repeat=%d median_ms=%.6f min_ms=%.6f max_ms=%.6f\n"
    (Metal_runtime.device_name runtime)
    (List.length (Metal_runtime.Execution.kernels execution))
    (Bytes.length contents) warmup_count repeat_count (median measurements)
    (List.fold_left Float.min Float.infinity measurements)
    (List.fold_left Float.max Float.neg_infinity measurements);
  dump_kernel_histogram execution;
  Ok ()

let run_all root input_name input_path output_directory =
  let* package =
    Serving_package.of_file (Filename.concat root "package.llmopt")
  in
  let* runtime = Metal_runtime.load_package ~root package in
  let* input_bytes = read_file input_path in
  let* input = Metal_runtime.Buffer.of_bytes ~runtime input_bytes in
  let started_at = Unix.gettimeofday () in
  let* execution = Metal_runtime.execute runtime ~inputs:[ input_name, input ] in
  let elapsed_ms = (Unix.gettimeofday () -. started_at) *. 1000.0 in
  let outputs = Metal_runtime.Execution.outputs execution in
  let* () =
    List.fold_left
      (fun result (name, output) ->
        let* () = result in
        let* contents = Metal_runtime.Buffer.contents output in
        write_file (Filename.concat output_directory (name ^ ".bin")) contents)
      (Ok ()) outputs
  in
  Printf.printf
    "device=%s dispatch_count=%d outputs=%d elapsed_ms=%.6f\n"
    (Metal_runtime.device_name runtime)
    (List.length (Metal_runtime.Execution.kernels execution))
    (List.length outputs) elapsed_ms;
  Ok ()

let count ~name ~minimum raw =
  let value = int_of_string raw in
  if value < minimum then
    invalid_arg (Printf.sprintf "%s must be at least %d" name minimum);
  value

let usage () =
  prerr_endline
    "usage: llmopt-package-run --warmup <count> --repeat <count> \
     <package-directory> <input-name> <input.bin> <output-name> <output.bin>\n\
     or: llmopt-package-run --all-outputs <package-directory> <input-name> \
     <input.bin> <output-directory>";
  exit 64

let () =
  try
    let result =
      match Array.to_list Sys.argv with
      | [ _; "--all-outputs"; root; input_name; input_path; output_directory ] ->
          run_all root input_name input_path output_directory
      | [ _; "--warmup"; raw_warmup; "--repeat"; raw_repeat; root;
          input_name; input_path; output_name; output_path ] ->
          let warmup_count = count ~name:"warmup" ~minimum:0 raw_warmup in
          let repeat_count = count ~name:"repeat" ~minimum:1 raw_repeat in
          run ~warmup_count ~repeat_count root input_name input_path output_name
            output_path
      | _ -> usage ()
    in
    match result with
    | Ok () -> ()
    | Error message ->
        prerr_endline message;
        exit 2
  with Invalid_argument message ->
    prerr_endline message;
    exit 64
