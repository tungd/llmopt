let get = function Ok x -> x | Error e -> failwith e
let write path contents =
  let c = open_out_bin path in
  Fun.protect ~finally:(fun () -> close_out c) (fun () -> output_string c contents)

let () =
  if Array.length Sys.argv = 3 && Sys.argv.(1) = "--runtime-source" then
    (write Sys.argv.(2) Metal_exl3_codegen.runtime_source; exit 0);
  if Array.length Sys.argv <> 3 then (prerr_endline "usage: llmopt-exl3-probe-source archive output-directory"; exit 2);
  let archive = get (Weight_archive.of_file Sys.argv.(1)) in
  let bound = get (Exl3.Archive.bind archive ~prefix:"weight") in
  let storage = Exl3.Archive.storage bound in
  let dir = Sys.argv.(2) in
  let required name dtype shape =
    match Weight_archive.find archive name with
    | Some t when Weight_archive.Tensor.dtype t = dtype && Weight_archive.Tensor.shape t = shape -> t
    | _ -> failwith ("missing or malformed fixture component: " ^ name)
  in
  let rows = match Weight_archive.find archive "input" with
    | Some t -> (match Weight_archive.Tensor.shape t with [m; k] when m > 0 && k = Exl3.k storage -> m | _ -> failwith "fixture input shape")
    | None -> failwith "fixture input missing"
  in
  if rows > 0x7fffffff / max (Exl3.k storage) (Exl3.n storage) then failwith "fixture activation indexing exceeds u32 range";
  ignore (required "input" Weight_archive.Dtype.F16 [rows; Exl3.k storage]);
  ignore (required "expected" Weight_archive.Dtype.F32 [rows; Exl3.n storage]);
  ignore (required "decoded" Weight_archive.Dtype.F16 [Exl3.k storage; Exl3.n storage]);
  ignore (required "codebook" Weight_archive.Dtype.F16 [65536]);
  let schedules = Metal_exl3_codegen.Schedule.[get (vector ~rows:1); get (vector ~rows:4); get (matrix ~rows:8); get (matrix ~rows:16)] in
  let variants = List.map (fun schedule ->
    let name = Metal_exl3_codegen.Schedule.name schedule in
    let path = name ^ ".metal" in
    write (Filename.concat dir path) (Metal_exl3_codegen.emit storage schedule ~epilogue:Epilogue.none);
    `Assoc ["name", `String name; "source", `String path;
            "rows", `Int (Metal_exl3_codegen.Schedule.rows schedule);
            "threads", `Int (Metal_exl3_codegen.Schedule.threads schedule);
            "columns", `Int (Metal_exl3_codegen.Schedule.columns schedule)]
  ) schedules in
  (* Also compile and execute a composed epilogue after the output transform. *)
  let schedule = List.hd schedules in
  write (Filename.concat dir "vector_m1_silu.metal")
    (Metal_exl3_codegen.emit storage schedule ~epilogue:Epilogue.silu);
  let variants = variants @ [`Assoc ["name", `String "vector_m1_silu"; "source", `String "vector_m1_silu.metal";
    "rows", `Int 1; "threads", `Int (Metal_exl3_codegen.Schedule.threads schedule); "columns", `Int (Metal_exl3_codegen.Schedule.columns schedule); "silu", `Bool true]] in
  let tensors = Weight_archive.tensors archive |> List.map (fun t ->
    Weight_archive.Tensor.name t, `Assoc ["offset", `Int (Weight_archive.Tensor.offset t);
      "bytes", `Int (Weight_archive.Tensor.byte_length t)]) in
  let manifest = `Assoc ["archive", `String (Unix.realpath Sys.argv.(1));
    "k", `Int (Exl3.k storage); "n", `Int (Exl3.n storage); "rows", `Int rows;
    "bits", `Int (Exl3.bits storage); "variants", `List variants; "tensors", `Assoc tensors] in
  write (Filename.concat dir "probe.json") (Yojson.Basic.pretty_to_string manifest ^ "\n")
