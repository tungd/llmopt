(** Unit tests for Pass_algebraic_canonicalization *)

(* Values consumed inside a node's pointwise op payload (they are separate from
   [node_inputs] and must be rewritten too when producers are eliminated). *)
let embedded_pointwise_values node =
  match Ir.node_op node with
  | Ir.Op.Primitive (Ir.Primitive.Pointwise operation) ->
      Ir.Pointwise.values operation
  | _ -> []

let assert_all_same expected = function
  | [] -> assert false
  | values -> List.iter (fun value -> assert (Ir.Value.equal expected value)) values

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
  let embedded1 = embedded_pointwise_values remaining_node in
  assert (List.length embedded1 = 2);
  assert_all_same v_in embedded1;
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
  let embedded2 = embedded_pointwise_values silu_opt in
  assert (List.length embedded2 = 1);
  (* The silu payload must no longer reference the eliminated f16 value. *)
  assert (Ir.Value.equal v_in2 (List.hd embedded2));
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
  let survivor = List.hd add_inps in
  let embedded4 = embedded_pointwise_values add_opt in
  assert (List.length embedded4 = 2);
  (* Both operands embed the surviving value; the duplicate was rewritten. *)
  assert_all_same survivor embedded4;
  Printf.printf "  [PASS] Movement CSE\n%!";

  (* 5. End-to-end regression: when a cast chain is eliminated, the values
     embedded inside the surviving consumer's pointwise op must be rewritten
     too.  Otherwise Compute_plan.of_graph reports the pipeline crash
     "node N reads value M with no producer" on real graphs. *)
  let g5 = Ir.Graph.create () in
  let v_in5 =
    Ir.Graph.tensor_input g5 ~name:"x" ~source:Ir.Input_source.Runtime ~shape
      ~dtype:Ir.Dtype.Float16
  in
  let v_f32b =
    Ir.Graph.fresh_tensor_value g5 ~shape ~dtype:Ir.Dtype.Float32
  in
  let v_f16b =
    Ir.Graph.fresh_tensor_value g5 ~shape ~dtype:Ir.Dtype.Float16
  in
  let cast5a =
    Ir.node_create ~id:1
      ~op:(Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float32))
      ~inputs:[ v_in5 ] ~output:(Some v_f32b)
  in
  let cast5b =
    Ir.node_create ~id:2
      ~op:(Ir.Op.Primitive (Ir.Primitive.Cast Ir.Dtype.Float16))
      ~inputs:[ v_f32b ] ~output:(Some v_f16b)
  in
  let v_out5 =
    Ir.Graph.fresh_tensor_value g5 ~shape ~dtype:Ir.Dtype.Float16
  in
  let add5 =
    Ir.node_create ~id:3
      ~op:
        (Ir.Op.Primitive
           (Ir.Primitive.Pointwise
              (Ir.Pointwise.Binary
                 ( Ir.Pointwise.Add, Ir.Pointwise.Tensor v_f16b,
                   Ir.Pointwise.Tensor v_f16b ))))
      ~inputs:[ v_f16b; v_f16b ] ~output:(Some v_out5)
  in
  (* tensor_input registered the Input node (id 0); keep it so every value in
     the graph has a producer, exactly like a real pipeline graph. *)
  let g5 =
    Ir.Graph.with_nodes_and_outputs g5
      (Ir.Graph.nodes g5 @ [ cast5a; cast5b; add5 ])
      [ ("out", v_out5) ]
  in
  let g5_opt = Pass_algebraic_canonicalization.run g5 in
  let add5_opt =
    List.find (fun n -> Ir.node_id n = 3) (Ir.Graph.nodes g5_opt)
  in
  let embedded5 = embedded_pointwise_values add5_opt in
  assert (List.length embedded5 = 2);
  assert_all_same v_in5 embedded5;
  assert_all_same v_in5 (Ir.node_inputs add5_opt);
  (match Compute_plan.of_graph g5_opt with
  | Ok _ -> ()
  | Error message ->
      failwith
        ("canonicalized graph must stay compute-plan valid \
          (embedded-operand rewrite): " ^ message));
  Printf.printf "  [PASS] Embedded operand rewrite keeps compute plan valid\n%!";

  Printf.printf "All Algebraic Canonicalization pass tests passed successfully!\n%!"
