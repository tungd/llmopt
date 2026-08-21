let ensure_directory path =
  let rec create current =
    if current = "" || current = "." || Sys.file_exists current then ()
    else (
      create (Filename.dirname current);
      Unix.mkdir current 0o755)
  in
  create path

let write_file path contents =
  let channel = open_out path in
  Fun.protect
    ~finally:(fun () -> close_out_noerr channel)
    (fun () -> output_string channel contents)

let usage () =
  prerr_endline "usage: llmopt-fx <fx-manifest.json> <output-directory>";
  exit 64

let () =
  if Array.length Sys.argv <> 3 then usage ();
  let manifest = Sys.argv.(1) in
  let output_directory = Sys.argv.(2) in
  ensure_directory output_directory;
  match Fx.of_file manifest with
  | Error message ->
      prerr_endline message;
      exit 2
  | Ok fx_graph ->
      (match Fx_plan.plan fx_graph with
      | Error message ->
          prerr_endline message;
          exit 3
      | Ok graph ->
          let planned = Passes.optimize graph in
          let plan = Format.asprintf "%a" Ir.Graph.pp planned in
          write_file (Filename.concat output_directory "plan.txt") plan;
          (match Metal.emit planned with
          | Ok source -> write_file (Filename.concat output_directory "kernel.metal") source
          | Error message ->
              prerr_endline ("Metal emission skipped: " ^ message));
          (match Llvm_ir.emit planned with
          | Ok source -> write_file (Filename.concat output_directory "kernel.ll") source
          | Error message ->
            prerr_endline ("LLVM emission skipped: " ^ message));
          Printf.printf "planned %d FX nodes into %d IR nodes\n"
            (List.length (Fx.nodes fx_graph))
            (List.length (Ir.Graph.nodes planned)))
