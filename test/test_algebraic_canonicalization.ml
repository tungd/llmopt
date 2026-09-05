(** Unit tests for Pass_algebraic_canonicalization *)

let run () =
  Printf.printf "Running Algebraic Canonicalization pass tests...\n%!";

  (* 1. Identity Cast Elision *)
  let g1 = Ir.Graph.create () in
  let shape = Tensor_shape.of_ints_exn [ 2; 4 ] in
  let v_in = Ir.Graph.fresh_tensor_value g1 ~shape ~dtype:Ir.Dtype.Float16 in
  let v_cast = Ir.Graph.fresh_tensor_value g1 ~shape ~dtype:Ir.Dtype.Float16 in
  let cast_node = Ir.node_create ~id:1 ~op:(Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float16))
    ~inputs:[ v_in ] ~output:(Some v_cast) in
  let v_out = Ir.Graph.fresh_tensor_value g1 ~shape ~dtype:Ir.Dtype.Float16 in
  let add_node = Ir.node_create ~id:2 ~op:(Ir.Op.Primitive (Ir.Primitive.Pointwise (Ir.Pointwise.Binary (Ir.Pointwise.Add, Ir.Pointwise.Tensor v_cast, Ir.Pointwise.Tensor v_cast))))
    ~inputs:[ v_cast; v_cast ] ~output:(Some v_out) in
  let g1 = Ir.Graph.with_nodes_and_outputs g1 [ cast_node; add_node ] [ ("out", v_out) ] in

  let g1_opt = Pass_algebraic_canonicalization.run g1 in
  let nodes1 = Ir.Graph.nodes g1_opt in
  assert (List.length nodes1 = 1);
  let remaining_node = List.hd nodes1 in
  assert (Ir.node_id remaining_node = 2);
  let inputs1 = Ir.node_inputs remaining_node in
  assert (List.for_all (fun inp -> Ir.Value.id inp = Ir.Value.id v_in) inputs1);
  Printf.printf "  [PASS] Identity cast elision\n%!";

  (* 2. Chained Precision Roundtrip Bypass: f16 -> f32 -> f16 *)
  let g2 = Ir.Graph.create () in
  let v_in2 = Ir.Graph.fresh_tensor_value g2 ~shape ~dtype:Ir.Dtype.Float16 in
  let v_f32 = Ir.Graph.fresh_tensor_value g2 ~shape ~dtype:Ir.Dtype.Float32 in
  let v_f16 = Ir.Graph.fresh_tensor_value g2 ~shape ~dtype:Ir.Dtype.Float16 in
  let cast1 = Ir.node_create ~id:1 ~op:(Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float32))
    ~inputs:[ v_in2 ] ~output:(Some v_f32) in
  let cast2 = Ir.node_create ~id:2 ~op:(Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float16))
    ~inputs:[ v_f32 ] ~output:(Some v_f16) in
  let v_final = Ir.Graph.fresh_tensor_value g2 ~shape ~dtype:Ir.Dtype.Float16 in
  let silu = Ir.node_create ~id:3 ~op:(Ir.Op.Primitive (Ir.Primitive.Pointwise (Ir.Pointwise.Unary (Ir.Pointwise.Silu, v_f16))))
    ~inputs:[ v_f16 ] ~output:(Some v_final) in
  let g2 = Ir.Graph.with_nodes_and_outputs g2 [ cast1; cast2; silu ] [ ("out", v_final) ] in

  let g2_opt = Pass_algebraic_canonicalization.run g2 in
  let nodes2 = Ir.Graph.nodes g2_opt in
  let silu_opt = List.find (fun n -> Ir.node_id n = 3) nodes2 in
  assert (Ir.Value.id (List.hd (Ir.node_inputs silu_opt)) = Ir.Value.id v_in2);
  Printf.printf "  [PASS] Chained precision roundtrip bypass\n%!";

  (* 3. Identity Movement Elision: View with same shape *)
  let g3 = Ir.Graph.create () in
  let v_in3 = Ir.Graph.fresh_tensor_value g3 ~shape ~dtype:Ir.Dtype.Float32 in
  let v_view = Ir.Graph.fresh_tensor_value g3 ~shape ~dtype:Ir.Dtype.Float32 in
  let view_node = Ir.node_create ~id:1 ~op:(Ir.Op.Primitive (Ir.Primitive.Movement Ir.Movement.View))
    ~inputs:[ v_in3 ] ~output:(Some v_view) in
  let g3 = Ir.Graph.with_nodes_and_outputs g3 [ view_node ] [ ("out", v_view) ] in

  let g3_opt = Pass_algebraic_canonicalization.run g3 in
  assert (List.length (Ir.Graph.nodes g3_opt) = 0);
  let (_, out_val) = List.hd (Ir.Graph.outputs g3_opt) in
  assert (Ir.Value.id out_val = Ir.Value.id v_in3);
  Printf.printf "  [PASS] Identity movement elision\n%!";

  (* 4. Movement CSE: duplicate unsqueeze *)
  let g4 = Ir.Graph.create () in
  let shape_1d = Tensor_shape.of_ints_exn [ 4 ] in
  let shape_2d = Tensor_shape.of_ints_exn [ 1; 4 ] in
  let v_in4 = Ir.Graph.fresh_tensor_value g4 ~shape:shape_1d ~dtype:Ir.Dtype.Float32 in
  let v_unsq1 = Ir.Graph.fresh_tensor_value g4 ~shape:shape_2d ~dtype:Ir.Dtype.Float32 in
  let v_unsq2 = Ir.Graph.fresh_tensor_value g4 ~shape:shape_2d ~dtype:Ir.Dtype.Float32 in
  let unsq1 = Ir.node_create ~id:1 ~op:(Ir.Op.Primitive (Ir.Primitive.Movement (Ir.Movement.Unsqueeze 0)))
    ~inputs:[ v_in4 ] ~output:(Some v_unsq1) in
  let unsq2 = Ir.node_create ~id:2 ~op:(Ir.Op.Primitive (Ir.Primitive.Movement (Ir.Movement.Unsqueeze 0)))
    ~inputs:[ v_in4 ] ~output:(Some v_unsq2) in
  let v_out4 = Ir.Graph.fresh_tensor_value g4 ~shape:shape_2d ~dtype:Ir.Dtype.Float32 in
  let add4 = Ir.node_create ~id:3 ~op:(Ir.Op.Primitive (Ir.Primitive.Pointwise (Ir.Pointwise.Binary (Ir.Pointwise.Add, Ir.Pointwise.Tensor v_unsq1, Ir.Pointwise.Tensor v_unsq2))))
    ~inputs:[ v_unsq1; v_unsq2 ] ~output:(Some v_out4) in
  let g4 = Ir.Graph.with_nodes_and_outputs g4 [ unsq1; unsq2; add4 ] [ ("out", v_out4) ] in

  let g4_opt = Pass_algebraic_canonicalization.run g4 in
  let nodes4 = Ir.Graph.nodes g4_opt in
  assert (List.length nodes4 = 2); (* only one unsqueeze + one add *)
  let add_opt = List.find (fun n -> Ir.node_id n = 3) nodes4 in
  let add_inps = Ir.node_inputs add_opt in
  assert (Ir.Value.id (List.nth add_inps 0) = Ir.Value.id (List.nth add_inps 1));
  Printf.printf "  [PASS] Movement CSE\n%!";

  Printf.printf "All Algebraic Canonicalization pass tests passed successfully!\n%!"
