let ( let* ) = Result.bind

type context_handle
type library_handle
type buffer_handle

external create_context : unit -> context_handle
  = "caml_llmopt_metal_create_context"
external device_name : context_handle -> string
  = "caml_llmopt_metal_device_name"
external load_library : context_handle -> string -> library_handle
  = "caml_llmopt_metal_load_library"
external create_buffer : context_handle -> int -> buffer_handle
  = "caml_llmopt_metal_create_buffer"
external buffer_of_bytes : context_handle -> bytes -> buffer_handle
  = "caml_llmopt_metal_buffer_of_bytes"
external dispatch :
  library_handle * string * buffer_handle list * bytes * int * int * int * int * int * int -> unit
  = "caml_llmopt_metal_dispatch"

let benchmark_quant_kernel ~ctx ~lib ~kernel_name ~k ~n ~bytes_per_row ~quant_name =
  let m = 1 in
  let input_bytes = m * k * 2 in
  let weight_bytes = n * bytes_per_row in
  let output_bytes = m * n * 2 in
  
  let in_buf = create_buffer ctx input_bytes in
  let wt_buf = create_buffer ctx weight_bytes in
  let dummy_bias = create_buffer ctx (n * 2) in
  let out_buf = create_buffer ctx output_bytes in
  
  let params_bytes = Bytes.create 16 in
  Bytes.set_int32_le params_bytes 0 (Int32.of_int m);
  Bytes.set_int32_le params_bytes 4 (Int32.of_int n);
  Bytes.set_int32_le params_bytes 8 (Int32.of_int k);
  Bytes.set_int32_le params_bytes 12 0l;
  let params_buf = buffer_of_bytes ctx params_bytes in
  
  let threads_x = n * 32 in
  let tg_x = 32 in
  
  let run_once () =
    dispatch (lib, kernel_name, [in_buf; wt_buf; dummy_bias; out_buf; params_buf], Bytes.empty, threads_x, 1, 1, tg_x, 1, 1)
  in
  
  for _ = 1 to 50 do run_once () done;
  
  let iters = 200 in
  let t0 = Unix.gettimeofday () in
  for _ = 1 to iters do run_once () done;
  let t1 = Unix.gettimeofday () in
  
  let total_time_s = t1 -. t0 in
  let avg_latency_us = (total_time_s /. float_of_int iters) *. 1_000_000.0 in
  let total_transferred_bytes = float_of_int (input_bytes + weight_bytes + output_bytes) in
  let achieved_bw_gb_s = (total_transferred_bytes /. (avg_latency_us *. 1e-6)) /. (1024.0 *. 1024.0 *. 1024.0) in
  let tpot_tok_per_sec = 1_000_000.0 /. avg_latency_us in
  
  Printf.printf "  %-8s (K=%4d, N=%4d) | Latency: %6.2f µs | BW: %6.2f GB/s | Rate: %7.1f tok/s\n%!"
    quant_name k n avg_latency_us achieved_bw_gb_s tpot_tok_per_sec

let benchmark_ffn_block ~ctx ~lib ~k ~n ~bytes_per_row ~quant_name ~dual_name ~down_name =
  let m = 1 in
  let act_buf = create_buffer ctx (m * k * 2) in
  let gate_wt = create_buffer ctx (n * bytes_per_row) in
  let up_wt = create_buffer ctx (n * bytes_per_row) in
  let down_wt = create_buffer ctx (k * bytes_per_row) in
  let prod_buf = create_buffer ctx (m * n * 2) in
  let res_buf = create_buffer ctx (m * k * 2) in
  let out_buf = create_buffer ctx (m * k * 2) in

  let params_swiglu = Bytes.create 16 in
  Bytes.set_int32_le params_swiglu 0 (Int32.of_int m);
  Bytes.set_int32_le params_swiglu 4 (Int32.of_int n);
  Bytes.set_int32_le params_swiglu 8 (Int32.of_int k);
  Bytes.set_int32_le params_swiglu 12 0l;
  let p_swiglu_buf = buffer_of_bytes ctx params_swiglu in

  let params_down = Bytes.create 16 in
  Bytes.set_int32_le params_down 0 (Int32.of_int m);
  Bytes.set_int32_le params_down 4 (Int32.of_int k);
  Bytes.set_int32_le params_down 8 (Int32.of_int n);
  Bytes.set_int32_le params_down 12 0l;
  let p_down_buf = buffer_of_bytes ctx params_down in

  let run_fused_ffn () =
    dispatch (lib, dual_name, [act_buf; gate_wt; up_wt; prod_buf; p_swiglu_buf], Bytes.empty, n * 32, 1, 1, 32, 1, 1);
    dispatch (lib, down_name, [prod_buf; down_wt; res_buf; out_buf; p_down_buf], Bytes.empty, k * 32, 1, 1, 32, 1, 1)
  in

  for _ = 1 to 50 do run_fused_ffn () done;

  let iters = 200 in
  let t0 = Unix.gettimeofday () in
  for _ = 1 to iters do run_fused_ffn () done;
  let t1 = Unix.gettimeofday () in

  let total_time_s = t1 -. t0 in
  let avg_latency_us = (total_time_s /. float_of_int iters) *. 1_000_000.0 in
  let total_transferred_bytes = float_of_int ((2 * n * bytes_per_row) + (k * bytes_per_row)) in
  let achieved_bw_gb_s = (total_transferred_bytes /. (avg_latency_us *. 1e-6)) /. (1024.0 *. 1024.0 *. 1024.0) in

  Printf.printf "  %-8s (K=%4d, N=%4d) | Fused FFN Latency: %6.2f µs | BW: %6.2f GB/s | FFN Rate: %7.1f tok/s\n%!"
    quant_name k n avg_latency_us achieved_bw_gb_s (1_000_000.0 /. avg_latency_us)

let () =
  Printf.printf "=================================================================\n";
  Printf.printf "  llmopt Quantized Megakernel Microbenchmarks (Apple Metal GPU)\n";
  Printf.printf "=================================================================\n%!";
  
  let ctx = create_context () in
  Printf.printf "  Device: %s\n\n%!" (device_name ctx);
  
  let metal_code =
    "#include <metal_stdlib>\n#include <metal_matrix>\nusing namespace metal;\n"
    ^ {|
struct QuantBlock32Params { uint total_blocks; };
struct QuantBlock256Params { uint total_superblocks; };
struct QuantLinearParams { uint m; uint n; uint k; uint has_bias; };
|}
    ^ Metal.q8_0_source ^ Metal.q5_0_source ^ Metal.q4_0_source
    ^ Metal.q4_k_source ^ Metal.q5_k_source ^ Metal.q6_k_source
  in
  let tmp_metal = Filename.temp_file "llmopt_kernels" ".metal" in
  let tmp_metallib = Filename.temp_file "llmopt_kernels" ".metallib" in
  let out_ch = open_out tmp_metal in
  output_string out_ch metal_code;
  close_out out_ch;
  
  let cmd = Printf.sprintf "xcrun metal -O3 %s -o %s" (Filename.quote tmp_metal) (Filename.quote tmp_metallib) in
  if Sys.command cmd <> 0 then (
    prerr_endline "Failed to compile Metal shaders";
    exit 1
  );
  let lib = load_library ctx tmp_metallib in
  
  Printf.printf "  --- Individual Single-Projection GEMV Latencies ---\n";
  benchmark_quant_kernel ~ctx ~lib ~kernel_name:"llmopt_q4_0_linear_f16" ~k:1024 ~n:4608 ~bytes_per_row:(1024 / 32 * 18) ~quant_name:"Q4_0";
  benchmark_quant_kernel ~ctx ~lib ~kernel_name:"llmopt_q4_k_linear_f16" ~k:1024 ~n:4608 ~bytes_per_row:(1024 / 256 * 144) ~quant_name:"Q4_K";
  benchmark_quant_kernel ~ctx ~lib ~kernel_name:"llmopt_q5_k_linear_f16" ~k:1024 ~n:4608 ~bytes_per_row:(1024 / 256 * 176) ~quant_name:"Q5_K";
  benchmark_quant_kernel ~ctx ~lib ~kernel_name:"llmopt_q6_k_linear_f16" ~k:1024 ~n:4608 ~bytes_per_row:(1024 / 256 * 210) ~quant_name:"Q6_K";
  benchmark_quant_kernel ~ctx ~lib ~kernel_name:"llmopt_q8_0_linear_f16" ~k:1024 ~n:4608 ~bytes_per_row:(1024 / 32 * 34) ~quant_name:"Q8_0";

  Printf.printf "\n  --- Complete Fused SwiGLU FFN Block (Gate+Up+SiLU+Down+Add) ---\n";
  benchmark_ffn_block ~ctx ~lib ~k:1024 ~n:4608 ~bytes_per_row:(1024 / 32 * 18) ~quant_name:"Q4_0"
    ~dual_name:"llmopt_q4_0_dual_swiglu_f16" ~down_name:"llmopt_q4_0_down_add_f16";
  benchmark_ffn_block ~ctx ~lib ~k:1024 ~n:4608 ~bytes_per_row:(1024 / 256 * 144) ~quant_name:"Q4_K"
    ~dual_name:"llmopt_q4_k_dual_swiglu_f16" ~down_name:"llmopt_q4_k_down_add_f16";
  benchmark_ffn_block ~ctx ~lib ~k:1024 ~n:4608 ~bytes_per_row:(1024 / 32 * 34) ~quant_name:"Q8_0"
    ~dual_name:"llmopt_q8_0_dual_swiglu_f16" ~down_name:"llmopt_q8_0_down_add_f16";

  Printf.printf "\n  --- Llama-3.2-1B Complete Fused FFN Block (K=2048, N=8192) ---\n";
  benchmark_ffn_block ~ctx ~lib ~k:2048 ~n:8192 ~bytes_per_row:(2048 / 256 * 144) ~quant_name:"Q4_K"
    ~dual_name:"llmopt_q4_k_dual_swiglu_f16" ~down_name:"llmopt_q4_k_down_add_f16";
  benchmark_ffn_block ~ctx ~lib ~k:2048 ~n:8192 ~bytes_per_row:(2048 / 32 * 34) ~quant_name:"Q8_0"
    ~dual_name:"llmopt_q8_0_dual_swiglu_f16" ~down_name:"llmopt_q8_0_down_add_f16";

  Printf.printf "\n  --- Gemma-4-E2B Complete Fused FFN Block (K=1536, N=8960) ---\n";
  benchmark_ffn_block ~ctx ~lib ~k:1536 ~n:8960 ~bytes_per_row:(1536 / 256 * 144) ~quant_name:"Q4_K"
    ~dual_name:"llmopt_q4_k_dual_swiglu_f16" ~down_name:"llmopt_q4_k_down_add_f16";
  benchmark_ffn_block ~ctx ~lib ~k:1536 ~n:8960 ~bytes_per_row:(1536 / 32 * 34) ~quant_name:"Q8_0"
    ~dual_name:"llmopt_q8_0_dual_swiglu_f16" ~down_name:"llmopt_q8_0_down_add_f16";

  Printf.printf "\n=================================================================\n%!"
