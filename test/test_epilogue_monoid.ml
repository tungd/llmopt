(** Unit tests and property-based verification for:
    1. Affine Coordinate Category (The Index Monoid)
    2. Free Pointwise Applicative Functor
    3. Contraction Epilogue Monoid (E, ⋄, ε) *)

let run () =
  Printf.printf "Running Epilogue Monoid and Coordinate Category tests...\n%!";

  (* Helper to create dummy values *)
  let g = Ir.Graph.create () in
  let shape_1d = Tensor_shape.of_ints_exn [ 64 ] in
  let v_bias = Ir.Graph.fresh_tensor_value g ~shape:shape_1d ~dtype:Ir.Dtype.Float16 in
  let v_res = Ir.Graph.fresh_tensor_value g ~shape:shape_1d ~dtype:Ir.Dtype.Float16 in
  let v_scale = Ir.Graph.fresh_tensor_value g ~shape:shape_1d ~dtype:Ir.Dtype.Float16 in

  (* ========================================================================= *)
  (* 1. Coordinate Category Tests                                              *)
  (* ========================================================================= *)
  let id = Coordinate_map.identity in
  assert (Coordinate_map.is_identity id);

  (* Law: id ∘ T = T *)
  let perm = Coordinate_map.Permute [ 1; 0 ] in
  let id_perm = Coordinate_map.compose id perm in
  assert (Coordinate_map.equal id_perm perm);

  (* Law: T ∘ id = T *)
  let perm_id = Coordinate_map.compose perm id in
  assert (Coordinate_map.equal perm_id perm);

  (* Law: Involution (transpose ∘ transpose = id) *)
  let double_perm = Coordinate_map.compose perm perm in
  assert (Coordinate_map.is_identity double_perm);

  (* MSL index emission *)
  let idx_id = Coordinate_map.emit_msl_index id ~index_var:"col" ~shape:[ 32; 64 ] in
  assert (idx_id = "col");

  let idx_perm = Coordinate_map.emit_msl_index perm ~index_var:"col" ~shape:[ 32; 64 ] in
  assert (idx_perm = "((col % 64u) * 32u + (col / 64u))");

  Printf.printf "  [PASS] Coordinate Category axioms (identity, composition, involution)\n%!";

  (* ========================================================================= *)
  (* 2. Free Pointwise Applicative Functor Tests                               *)
  (* ========================================================================= *)
  let open Applicative_expr in

  (* Law: fmap id = id (cancels identity casts) *)
  let read_f16 = Read { value = v_bias; coord = Coordinate_map.identity } in
  let cast_f16 = fmap (Cast Ir.Dtype.Float16) read_f16 in
  assert (cast_f16 = read_f16); (* Identity cast eliminated at construction *)

  (* Law: Algebraic simplification of identities *)
  let x_plus_zero = simplify (lift2 Add read_f16 (pure 0.0)) in
  assert (x_plus_zero = read_f16);

  let x_mul_one = simplify (lift2 Mul read_f16 (pure 1.0)) in
  assert (x_mul_one = read_f16);

  let x_mul_zero = simplify (lift2 Mul read_f16 (pure 0.0)) in
  assert (x_mul_zero = Const 0.0);

  (* Constant folding *)
  let c2_plus_c3 = simplify (lift2 Add (pure 2.0) (pure 3.0)) in
  assert (c2_plus_c3 = Const 5.0);

  Printf.printf "  [PASS] Pointwise Applicative Functor laws (cast cancellation, constant folding)\n%!";

  (* ========================================================================= *)
  (* 3. Epilogue Monoid Tests (E, ⋄, ε)                                        *)
  (* ========================================================================= *)
  let eps = Epilogue.Monoid.empty in
  assert (Epilogue.Monoid.is_empty eps);

  let e_bias = Epilogue.to_expr (Epilogue.bias v_bias) in
  let e_res = Epilogue.to_expr (Epilogue.residual_add ~residual:v_res) in
  let e_silu = Epilogue.to_expr Epilogue.silu in

  (* Law 1: Left Identity: eps ⋄ e = e *)
  let left_id = Epilogue.Monoid.append eps e_bias in
  assert (Epilogue.Monoid.equal left_id e_bias);

  (* Law 2: Right Identity: e ⋄ eps = e *)
  let right_id = Epilogue.Monoid.append e_bias eps in
  assert (Epilogue.Monoid.equal right_id e_bias);

  (* Law 3: Associativity: (e1 ⋄ e2) ⋄ e3 = e1 ⋄ (e2 ⋄ e3) *)
  let left_assoc = Epilogue.Monoid.append (Epilogue.Monoid.append e_bias e_silu) e_res in
  let right_assoc = Epilogue.Monoid.append e_bias (Epilogue.Monoid.append e_silu e_res) in
  assert (Epilogue.Monoid.equal left_assoc right_assoc);

  Printf.printf "  [PASS] Epilogue Monoid strict laws (Left Identity, Right Identity, Associativity)\n%!";

  (* ========================================================================= *)
  (* 4. Arbitrary Composite Epilogue Fusion (e.g. Gemma 4)                     *)
  (* ========================================================================= *)
  (* Model chain: Linear -> Silu -> Residual_Add -> Scale *)
  let epi_silu = Epilogue.silu in
  let epi_bias = Epilogue.bias v_bias in
  let epi_res = Epilogue.residual_add ~residual:v_res in
  let epi_scale = Epilogue.of_expr
    (Binop (Mul, Slot Epilogue.Acc, Read { value = v_scale; coord = Coordinate_map.identity }))
  in

  (* Compose silu with bias *)
  let comp_res = Epilogue.compose ~producer:epi_silu ~consumer:epi_bias in
  assert (Result.is_ok comp_res);
  let fused_epi = Result.get_ok comp_res in

  (* Further compose with residual_add *)
  let comp_res2 = Epilogue.compose ~producer:fused_epi ~consumer:epi_res in
  assert (Result.is_ok comp_res2);
  let multi_fused = Result.get_ok comp_res2 in

  (* Further compose with scale *)
  let comp_res3 = Epilogue.compose ~producer:multi_fused ~consumer:epi_scale in
  assert (Result.is_ok comp_res3);
  let gemma_chain = Result.get_ok comp_res3 in

  let msl = Epilogue.emit_msl_writeback ~acc:"acc0" ~idx:"out_idx" gemma_chain in
  assert (String.length msl > 10);
  Printf.printf "  [PASS] Full Gemma composite epilogue chain emitted MSL: %s\n%!" msl;

  (* ========================================================================= *)
  (* 5. Dynamic Expr-epilogue buffer binding in parametric Metal emitters       *)
  (* ========================================================================= *)
  let contains haystack needle =
    let hlen = String.length haystack in
    let nlen = String.length needle in
    let rec loop i = i + nlen <= hlen && (String.sub haystack i nlen = needle || loop (i + 1)) in
    nlen = 0 || loop 0
  in
  let expect_contains label haystack needle =
    if not (contains haystack needle) then
      failwith (Printf.sprintf "%s: expected generated MSL to contain %S" label needle)
  in

  (* [gemma_chain] is an [Epilogue.Expr] reading three distinct f16 tensors;
     the parametric emitters must now declare one buffer per required value
     with the same deterministic buf_<Value_id> name the write-back
     expression uses, in stable required_values order. *)
  let chain_inputs = Epilogue.inputs gemma_chain in
  expect_contains "w4a16 Expr epilogue declares all three buffers"
    (Metal.emit_parametric_w4a16_linear ~name:"test_w4_expr" ~epilogue:gemma_chain)
    (List.mapi
       (fun i value ->
         Printf.sprintf "device const half* buf_%d [[buffer(%d)]]"
           (Ir.Value_id.to_int (Ir.Value.id value)) (6 + i))
       chain_inputs
    |> String.concat ",\n    ");
  expect_contains "q4_k Expr epilogue declares all three buffers"
    (Metal.emit_parametric_q4_k_linear ~name:"test_q4_expr" ~epilogue:gemma_chain)
    (List.mapi
       (fun i value ->
         Printf.sprintf "device const half* buf_%d [[buffer(%d)]]"
           (Ir.Value_id.to_int (Ir.Value.id value)) (6 + i))
       chain_inputs
    |> String.concat ",\n    ");
  expect_contains "simdgroup_matrix Expr epilogue declares all three buffers"
    (Metal.emit_simdgroup_matrix_gemm ~name:"test_gemm_expr" ~epilogue:gemma_chain)
    (List.mapi
       (fun i value ->
         Printf.sprintf "device const half* buf_%d [[buffer(%d)]]"
           (Ir.Value_id.to_int (Ir.Value.id value)) (4 + i))
       chain_inputs
    |> String.concat ",\n    ");
  (match chain_inputs with
  | first :: _ ->
      let first_id = Ir.Value_id.to_int (Ir.Value.id first) in
      expect_contains "w4a16 write-back references declared buffer with index expr"
        (Metal.emit_parametric_w4a16_linear ~name:"test_w4_expr2" ~epilogue:gemma_chain)
        (Printf.sprintf "(float)buf_%d[col]" first_id);
      expect_contains "simdgroup write-back references declared buffer with index expr"
        (Metal.emit_simdgroup_matrix_gemm ~name:"test_gemm_expr2" ~epilogue:gemma_chain)
        (Printf.sprintf "(float)buf_%d[c]" first_id)
  | [] -> failwith "gemma_chain epilogue unexpectedly reads no values");

  (* A Float32 read must declare a float (not half) buffer *)
  let v_f32 = Ir.Graph.fresh_tensor_value g ~shape:shape_1d ~dtype:Ir.Dtype.Float32 in
  let f32_expr =
    Epilogue.of_expr
      (Applicative_expr.Read { value = v_f32; coord = Coordinate_map.identity })
  in
  let f32_id = Ir.Value_id.to_int (Ir.Value.id v_f32) in
  expect_contains "Float32 Expr read declares float buffer"
    (Metal.emit_parametric_w4a16_linear ~name:"test_w4_expr_f32" ~epilogue:f32_expr)
    (Printf.sprintf "device const float* buf_%d [[buffer(6)]]" f32_id);

  Printf.printf "  [PASS] Expr epilogues bind dynamic buf_<Value_id> buffers in parametric emitters\n%!";

  (* ========================================================================= *)
  (* 6. Property-based Monoid laws over 10,000 synthetic expression trees      *)
  (* ========================================================================= *)
  Random.init 0x0CC1E;
  let shape_2d = Tensor_shape.of_ints_exn [ 4; 4 ] in
  let pv_list =
    Array.init 4 (fun _ ->
        Ir.Graph.fresh_tensor_value g ~shape:shape_2d ~dtype:Ir.Dtype.Float16)
  in
  let pv_env = Hashtbl.create 8 in
  Array.iter
    (fun value ->
      let id = Ir.Value_id.to_int (Ir.Value.id value) in
      let arr =
        Array.init 16 (fun j ->
            0.5 *. sin (float_of_int (id * 7 + j))
            +. 0.25 *. cos (float_of_int (id * 3 + 2 * j)))
      in
      Hashtbl.replace pv_env id arr)
    pv_list;
  let lookup_value value =
    Hashtbl.find pv_env (Ir.Value_id.to_int (Ir.Value.id value))
  in
  let coord_index value coord index =
    match coord with
    | Coordinate_map.Identity -> index
    | Coordinate_map.Permute [ 1; 0 ] -> (
        let dims = Tensor_shape.dimensions (Ir.Value.logical_shape value) in
        match dims with
        | m :: n :: _ -> (index mod n) * m + (index / n)
        | _ -> index)
    | _ -> index
  in
  (* IEEE-sound constant domain: strictly positive and nonzero. The algebraic
     simplifier folds Const leaves (Mul(_, Const 0.0) -> Const 0.0 and
     Log/Rsqrt over Const), which is only IEEE-exact when every folded
     operand is finite. Reads and probes may drive arbitrary subtrees to
     nan/inf at runtime, so any Const 0.0 (or non-positive Log domain) in a
     generated tree would make simplify diverge from the raw IEEE evaluation
     we compare against. Negative runtime values still occur through the
     sin/cos-valued reads and Sub/Min/Neg operators. *)
  let consts = [| 0.0625; 0.125; 0.25; 0.5; 1.0; 2.0; 4.0 |] in
  let unops =
    [| Applicative_expr.Neg; Applicative_expr.Abs; Applicative_expr.Silu;
       Applicative_expr.Gelu; Applicative_expr.Exp; Applicative_expr.Log;
       Applicative_expr.Rsqrt; Applicative_expr.Cast Ir.Dtype.Float16 |]
  in
  let binops =
    [| Applicative_expr.Add; Applicative_expr.Sub; Applicative_expr.Mul;
       Applicative_expr.Div; Applicative_expr.Min; Applicative_expr.Max |]
  in
  let triops = [| Applicative_expr.Clamp; Applicative_expr.Where |] in
  let pick arr = arr.(Random.int (Array.length arr)) in
  let rec gen_atom () =
    let r = Random.int 10 in
    if r < 4 then Applicative_expr.Slot Epilogue.Acc
    else if r = 4 then Applicative_expr.Const (pick consts)
    else
      let value = pick pv_list in
      let coord =
        if Random.bool () then Coordinate_map.identity else Coordinate_map.Permute [ 1; 0 ]
      in
      Applicative_expr.Read { value; coord }
  in
  let rec gen_tree depth =
    if depth <= 0 then gen_atom ()
    else
      let r = Random.int 100 in
      if r < 25 then gen_atom ()
      else if r < 60 then Applicative_expr.Unop (pick unops, gen_tree (depth - 1))
      else if r < 90 then
        Applicative_expr.Binop (pick binops, gen_tree (depth - 1), gen_tree (depth - 1))
      else
        Applicative_expr.Triop
          (pick triops, gen_tree (depth - 1), gen_tree (depth - 1), gen_tree (depth - 1))
  in
  let rec eval_tree ~index ~x = function
    | Applicative_expr.Slot Epilogue.Acc -> x
    | Applicative_expr.Const c -> c
    | Applicative_expr.Read { value; coord } ->
        (lookup_value value).(coord_index value coord index)
    | Applicative_expr.Unop (op, a) ->
        let v = eval_tree ~index ~x a in
        (match op with
        | Applicative_expr.Neg -> -.v
        | Applicative_expr.Abs -> abs_float v
        | Applicative_expr.Silu -> v /. (1.0 +. exp (-.v))
        | Applicative_expr.Gelu ->
            0.5 *. v
            *. (1.0 +. tanh (0.79788456 *. (v +. 0.044715 *. v *. v *. v)))
        | Applicative_expr.Exp -> exp v
        | Applicative_expr.Log -> log v
        | Applicative_expr.Rsqrt -> 1.0 /. sqrt v
        | Applicative_expr.Cast _ -> v)
    | Applicative_expr.Binop (op, a, b) ->
        let va = eval_tree ~index ~x a in
        let vb = eval_tree ~index ~x b in
        (match op with
        | Applicative_expr.Add -> va +. vb
        | Applicative_expr.Sub -> va -. vb
        | Applicative_expr.Mul -> va *. vb
        | Applicative_expr.Div -> va /. vb
        | Applicative_expr.Min -> min va vb
        | Applicative_expr.Max -> max va vb)
    | Applicative_expr.Triop (op, a, b, c) ->
        let va = eval_tree ~index ~x a in
        let vb = eval_tree ~index ~x b in
        let vc = eval_tree ~index ~x c in
        (match op with
        | Applicative_expr.Clamp -> max vb (min va vc)
        | Applicative_expr.Where -> if va <> 0.0 then vb else vc)
  in
  let close l r =
    l = r
    || (Float.is_nan l && Float.is_nan r)
    || abs_float (l -. r) <= 1e-6 *. (1.0 +. abs_float l)
  in
  let indices = [| 0; 3; 7; 12; 15 |] in
  let probes = [| -1.0; -0.25; 0.0; 0.5; 1.3 |] in
  let assert_same_denotation label left right =
    Array.iter
      (fun index ->
        Array.iter
          (fun x ->
            let l = eval_tree ~index ~x left in
            let r = eval_tree ~index ~x right in
            if not (close l r) then
              failwith
                (Printf.sprintf "%s violated at trial boundary index=%d x=%.3f (lhs=%.6f rhs=%.6f)"
                   label index x l r))
          probes)
      indices
  in
  let trials = 10000 in
  for trial = 1 to trials do
    (* Identity laws: eps ⋄ e = e and e ⋄ eps = e *)
    let e = gen_tree 3 in
    let left_id = Epilogue.Monoid.append Epilogue.Monoid.empty e in
    let right_id = Epilogue.Monoid.append e Epilogue.Monoid.empty in
    if not (Epilogue.Monoid.equal left_id e) then
      failwith
        (Printf.sprintf "structural left identity violated at trial %d:\n  e         = %s\n  eps ⋄ e   = %s" trial
           (Epilogue.Monoid.to_string e) (Epilogue.Monoid.to_string left_id));
    if not (Epilogue.Monoid.equal right_id e) then
      failwith (Printf.sprintf "structural right identity violated at trial %d" trial);
    assert_same_denotation "left identity" left_id e;
    assert_same_denotation "right identity" right_id e;
    (* Associativity: (a ⋄ b) ⋄ c = a ⋄ (b ⋄ c) *)
    let a = gen_tree 3 in
    let b = gen_tree 3 in
    let c = gen_tree 3 in
    let left_assoc = Epilogue.Monoid.append (Epilogue.Monoid.append a b) c in
    let right_assoc = Epilogue.Monoid.append a (Epilogue.Monoid.append b c) in
    if not (Epilogue.Monoid.equal left_assoc right_assoc) then
      failwith
        (Printf.sprintf
           "structural associativity violated at trial %d:\n  (a⋄b)⋄c = %s\n  a⋄(b⋄c) = %s" trial
           (Epilogue.Monoid.to_string left_assoc) (Epilogue.Monoid.to_string right_assoc));
    assert_same_denotation "associativity" left_assoc right_assoc
  done;
  Printf.printf
    "  [PASS] Property-based Monoid laws over %d synthetic expression trees (identity + associativity, structural and denotational)\n%!"
    trials;

  Printf.printf "All Epilogue Monoid & Coordinate Category tests passed successfully!\n%!"
