let ( let* ) = Result.bind

let format_bytes bytes =
  let mb = float_of_int bytes /. (1024.0 *. 1024.0) in
  if mb >= 1024.0 then Printf.sprintf "%.2f GiB" (mb /. 1024.0)
  else Printf.sprintf "%.2f MiB" mb

let inspect_file path =
  Printf.printf "=================================================================\n";
  Printf.printf "  Inspecting GGUF Model: %s\n" (Filename.basename path);
  Printf.printf "=================================================================\n";
  match Gguf.of_file path with
  | Error msg ->
      Printf.printf "  [ERROR] Failed to parse GGUF: %s\n" msg
  | Ok model ->
      Printf.printf "  GGUF Version:     %d\n" model.version;
      Printf.printf "  Alignment:        %d bytes\n" model.alignment;
      Printf.printf "  Architecture:     %s\n" (Option.value ~default:"unknown" (Gguf.architecture model));
      Printf.printf "  Context Length:   %s\n"
        (Option.fold ~none:"default" ~some:string_of_int (Gguf.context_length model));
      Printf.printf "  Embedding Dim:    %s\n"
        (Option.fold ~none:"default" ~some:string_of_int (Gguf.embedding_length model));
      Printf.printf "  Attention Heads:  %s\n"
        (Option.fold ~none:"default" ~some:string_of_int (Gguf.head_count model));
      Printf.printf "  KV Heads:         %s\n"
        (Option.fold ~none:"default" ~some:string_of_int (Gguf.head_count_kv model));
      Printf.printf "  Feed Forward Dim: %s\n"
        (Option.fold ~none:"default" ~some:string_of_int (Gguf.feed_forward_length model));
      Printf.printf "  Total Tensors:    %d\n" (List.length model.tensors);

      (* Tensor breakdown by quant type *)
      let quant_counts = Hashtbl.create 16 in
      let quant_bytes = Hashtbl.create 16 in
      let total_bytes = ref 0 in
      List.iter
        (fun (info : Gguf.Tensor_info.t) ->
          let dtype_str =
            match info.dtype with
            | Weight_archive.Dtype.Quant Q8_0 -> "Q8_0 (Block-32)"
            | Weight_archive.Dtype.Quant Q5_0 -> "Q5_0 (Block-32)"
            | Weight_archive.Dtype.Quant Q4_0 -> "Q4_0 (Block-32)"
            | Weight_archive.Dtype.Quant Q4_K -> "Q4_K (Superblock-256)"
            | Weight_archive.Dtype.Quant Q5_K -> "Q5_K (Superblock-256)"
            | Weight_archive.Dtype.Quant Q6_K -> "Q6_K (Superblock-256)"
            | Weight_archive.Dtype.Quant IQ4_XS -> "IQ4_XS (Superblock-256)"
            | Weight_archive.Dtype.F16 -> "Float16 (F16)"
            | Weight_archive.Dtype.BF16 -> "Bfloat16 (BF16)"
            | Weight_archive.Dtype.F32 -> "Float32 (F32)"
            | Weight_archive.Dtype.U8 -> "Uint8 (U8)"
            | _ -> "Other"
          in
          let count = Hashtbl.find_opt quant_counts dtype_str |> Option.value ~default:0 in
          let b = Hashtbl.find_opt quant_bytes dtype_str |> Option.value ~default:0 in
          Hashtbl.replace quant_counts dtype_str (count + 1);
          Hashtbl.replace quant_bytes dtype_str (b + info.byte_length);
          total_bytes := !total_bytes + info.byte_length)
        model.tensors;

      Printf.printf "\n  Quantization Breakdown:\n";
      Hashtbl.iter
        (fun dtype count ->
          let b = Hashtbl.find quant_bytes dtype in
          Printf.printf "    • %-26s : %3d tensors  (%s)\n" dtype count (format_bytes b))
        quant_counts;
      Printf.printf "  Total Weight Size: %s\n" (format_bytes !total_bytes);

      (* Sample first 5 tensors *)
      Printf.printf "\n  First 5 Tensors:\n";
      List.iteri
        (fun i (info : Gguf.Tensor_info.t) ->
          if i < 5 then
            Printf.printf "    [%2d] %-40s %-16s %s\n" i info.name
              (String.concat "x" (List.map string_of_int info.shape))
              (format_bytes info.byte_length))
        model.tensors;
      Printf.printf "  Status: VALID GGUF MODEL\n\n%!"

let () =
  let args = List.tl (Array.to_list Sys.argv) in
  if args = [] then (
    prerr_endline "usage: llmopt-inspect-gguf <model1.gguf> [model2.gguf ...]";
    exit 64)
  else List.iter inspect_file args
