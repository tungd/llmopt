let get = function Ok x -> x | Error e -> failwith e
let write path text = let c = open_out_bin path in
  Fun.protect ~finally:(fun () -> close_out c) (fun () -> output_string c text)
let tensor = Tensor_shape.of_ints_exn
let static g name shape dtype = Ir.Graph.tensor_input g ~name
  ~source:(Ir.Input_source.Tensor_store {key=name}) ~shape:(tensor shape) ~dtype
let op g op inputs shape =
  let output = Ir.Graph.fresh_tensor_value g ~shape:(tensor shape) ~dtype:Ir.Dtype.Float16 in
  Ir.Graph.append g ~op ~inputs ~output:(Some output); output
let graph m =
  let g = Ir.Graph.create () in
  let x = Ir.Graph.tensor_input g ~name:"input" ~source:Ir.Input_source.Runtime
    ~shape:(tensor [m;128]) ~dtype:Ir.Dtype.Float16 in
  let router = Ir.Graph.tensor_input g ~name:"router" ~source:Ir.Input_source.Runtime
    ~shape:(tensor [m;3]) ~dtype:Ir.Dtype.Float32 in
  let q = Ir.Dtype.Quant Ir.Dtype.Exl3_4 in
  let w = static g "linear" [256;128] q in
  let gate = static g "gate" [3;256;128] q in
  let up = static g "up" [3;256;128] q in
  let down = static g "down" [3;128;256] q in
  let linear = op g (Ir.Op.Linear {m;n=256;k=128;bias=false}) [x;w] [m;256] in
  let moe = op g (Ir.Op.Moe_swiglu_ffn {m;n_experts=3;top_k=2;hidden_dim=128;
      intermediate_dim=256;norm_topk_prob=true}) [x;router;gate;up;down] [m;128] in
  Ir.Graph.add_output g ~name:"linear" linear; Ir.Graph.add_output g ~name:"moe" moe;g
let half bytes i =
  let u = Bytes.get_uint16_le bytes (i*2) in
  let sign = if u land 0x8000 = 0 then 1. else -1. in
  let exp = (u lsr 10) land 31 and frac = u land 1023 in
  sign *. (if exp=0 then Float.ldexp (float frac) (-24)
    else if exp=31 then nan else Float.ldexp (float (1024+frac)) (exp-25))
let () =
  if Array.length Sys.argv <> 2 then failwith "usage: exl3-runtime-test fixture-directory";
  let root = Sys.argv.(1) in
  List.iter (fun m ->
    let g=graph m in
    let program=get (Metal.lower g) in
    write (Filename.concat root "kernel.metal") (Metal.Program.source program);
    write (Filename.concat root "kernel.metallib") "";
    let artifact x=get (Serving_package.Artifact.create x) in
    let schedule=get (Serving_schedule.of_graph g) in
    let package=get (Serving_package.serving
      ~files:(Serving_package.Files.create ~metal_library:(artifact "kernel.metallib"))
      ~kernels:(Metal.Program.kernels program) ~schedule
      ~tensor_store:(Serving_package.Tensor_store.weights ~file:(artifact "weights.llmopt"))
      ~cache:Serving_package.Cache.default ()) in
    (* Exercise schedule/dtype serialization as the real package loader does. *)
    let package=get (Serving_package.of_bytes (Serving_package.to_bytes package)) in
    let runtime=get (Metal_runtime.load_package ~root package) in
    let input name count =
      let buffer,_=get (Metal_native.tensor runtime ~name) in
      get (Metal_native.Buffer.view ~parent:buffer ~offset:0 ~bytes:count) in
    let inputs=["input",input "input" (m*128*2); "router",input "router" (m*3*4)] in
    List.iter (fun mode ->
    let execution = match mode with
      | `Direct -> get (Metal_runtime.execute runtime ~inputs)
      | `Prebaked ->
        let dispatches,execution=get (Metal_runtime.bind_schedule_plan runtime ~schedule ~inputs) in
        let dummy=get (Metal_native.Buffer.create ~runtime ~bytes:8) in
        let plan=get (Metal_runtime.Prebaked.create ~runtime ~dispatches
          ~token_buffer:dummy ~output_buffer:dummy) in
        ignore (get (Metal_runtime.Prebaked.execute plan ~token:0 ~past_tokens:0 ~checkpoint:0));
        execution in
    let label = match mode with `Direct -> "direct" | `Prebaked -> "prebaked" in
    List.iter (fun (name,n) ->
      let actual=Option.get (Metal_runtime.Execution.output execution ~name) |> Metal_native.Buffer.contents |> get in
      let expected,_=get (Metal_native.tensor runtime ~name:("expected."^name)) in
      let expected=get (Metal_native.Buffer.contents expected) in
      let worst=ref 0. in
      for i=0 to m*n-1 do
        let a=half actual i and e=half expected i in
        let error=abs_float(a-.e) in worst:=max !worst error;
        if not(Float.is_finite a) || error > 0.001 +. 0.002 *. abs_float e then
          failwith (Printf.sprintf "%s %s m=%d i=%d actual=%g expected=%g" label name m i a e)
      done;
      Printf.printf "%s %s m=%d max_abs_error=%g\n%!" label name m !worst
    ) ["linear",256;"moe",128]
    ) [`Direct;`Prebaked]
  ) [1;3;17;64]
