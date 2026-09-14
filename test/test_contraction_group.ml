(** G4P12-01: scalar-DAG elaboration + typed-group normal form.

    Pins for the new additive algebra core [Contraction_group]:

    A. Chain-region equivalence: for every chain-shaped region the legacy
       [Kernel_ir.Fused_tactic.fold_linear_epilogue] accepts today, the
       elaborated group's single-accumulator epilogue is [Epilogue.equal] to
       the legacy fold result AND to the pinned monoid tree (orientation
       canonical, tags reconstructed from the step list).
    B. Exercise (a): a [y*y] single-slot square region (rejected by the
       legacy fold because both operands are folded) forms a group whose
       single-acc epilogue is [Expr (Mul (Slot Acc, Slot Acc))].
    C. Exercise (b): two projections over a shared input followed by an
       arbitrary pointwise operation over both slots form a two-contraction
       group (not the legacy gated template, and not a foldable chain).
    D. Exercise (c): an exported intermediate feeding a fused consumer forms
       a group with two results and an explicit use obligation.
    E. Conservation where the legacy recognisers cut: sigmoid chains and
       regions with an unreferenced binding stay outside group formation;
       multi-export fan-out and Cast chains form groups (a documented
       superset of the legacy fold) but never reconstruct single-acc. *)

let fail message = raise (Failure message)

let expect condition message = if not condition then fail message

let expect_ok = function
  | Ok value -> value
  | Error message -> fail message

let expect_ok_kernel = function
  | Ok value -> value
  | Error error -> fail (Kernel_region.error_to_string error)

module Fixture = struct
  let shape_1_64 = Tensor_shape.of_ints_exn [ 1; 64 ]
  let shape_64_64 = Tensor_shape.of_ints_exn [ 64; 64 ]

  let value id shape dtype =
    Ir.Value.make_tensor ~id ~shape ~dtype

  let linear_primitive =
    Kernel_region.Primitive.linear ~m:1 ~n:64 ~k:64 ~bias:false
      ~storage:(Ir.Linear_storage.Dense Ir.Dtype.Float16)

  (* steps: (primitive, operand slot ints, output value).  Region inputs take
     slots 0..n-1 in order; step i binds slot (n + i). *)

  let make_region_with_members ~member_node_ids ~name ~inputs ~steps ~results =
    let inputs =
      List.mapi
        (fun index value ->
          ({ Kernel_region.slot = Kernel_region.Slot.of_int_exn index; value }
            : Kernel_region.input))
        inputs
    in
    let base = List.length inputs in
    let bindings =
      fst
        (List.fold_left
           (fun (bindings, count) (primitive, operand_slots, output_value) ->
             let slot = Kernel_region.Slot.of_int_exn (base + count) in
             let binding =
               Kernel_region.make_binding
                 ~outputs:
                   [
                     {
                       Kernel_region.slot;
                       shape = Ir.Value.logical_shape output_value;
                       dtype = Ir.Value.dtype output_value;
                       value = Some output_value;
                     };
                   ]
                 ~primitive
                 ~inputs:(List.map Kernel_region.Slot.of_int_exn operand_slots)
               |> expect_ok
             in
             (binding :: bindings, count + 1))
           ([], 0) steps)
    in
    let bindings = List.rev bindings in
    let results =
      List.map
        (fun (slot_index, result_value) ->
          {
            Kernel_region.slot = Kernel_region.Slot.of_int_exn slot_index;
            value = result_value;
            storage = Kernel_region.Fresh;
          })
        results
    in
    let input_accesses =
      List.map
        (fun ({ Kernel_region.slot; _ } : Kernel_region.input) ->
          {
            Kernel_region.Effect.slot;
            mode = Kernel_region.Effect.Read;
            alias = Kernel_region.Effect.Distinct;
          })
        inputs
    in
    let result_accesses =
      List.map
        (fun ({ Kernel_region.slot; _ } : Kernel_region.result) ->
          {
            Kernel_region.Effect.slot;
            mode = Kernel_region.Effect.Write;
            alias = Kernel_region.Effect.Distinct;
          })
        results
    in
    let effects =
      Kernel_region.Effect.create ~accesses:(input_accesses @ result_accesses) ()
      |> expect_ok
    in
    Kernel_region.create ~name
      ~member_node_ids
      ~inputs ~bindings ~results ~effects ~resource:Kernel_region.Resource.zero
    |> expect_ok_kernel

  let make_region ~name ~inputs ~steps ~results =
    make_region_with_members ~member_node_ids:[ 1; 2; 3; 4; 5; 6; 7; 8 ]
      ~name ~inputs ~steps ~results
  let single_result_region ~name ~inputs ~steps ~result_index result_value =
    make_region ~name ~inputs ~steps ~results:[ (result_index, result_value) ]
end

open Fixture
open Contraction_group

let chain_region_equiv ~name ~inputs ~steps ~result_index result_value expected =
  let region =
    single_result_region ~name ~inputs ~steps ~result_index result_value
  in
  match Kernel_ir.Fused_tactic.fold_linear_epilogue region with
  | None -> fail (name ^ ": legacy fold rejected a chain-shaped region")
  | Some legacy ->
      expect
        (Epilogue.equal legacy expected)
        (name ^ ": legacy fold pinned epilogue");
      (match Contraction_group.of_region region with
      | None -> fail (name ^ ": elaboration rejected a fold-accepted region")
      | Some group -> (
          match Contraction_group.single_acc_epilogue group with
          | None ->
              fail (name ^ ": single-accumulator reconstruction failed")
          | Some actual ->
              expect
                (Epilogue.equal actual legacy)
                (name ^ ": elaborated group equals legacy fold result");
              Printf.printf "  [PASS] %s\n%!" name))

let run () =
  Printf.printf "Running typed contraction-group algebra tests...\n%!";
  let f16 = Ir.Dtype.Float16 in
  let x = value 1 shape_1_64 f16 in
  let w = value 2 shape_64_64 f16 in
  let residual = value 3 shape_1_64 f16 in
  let up = value 4 shape_1_64 f16 in
  let o_linear = value 5 shape_1_64 f16 in
  let o_linear2 = value 6 shape_1_64 f16 in
  let o_silu = value 7 shape_1_64 f16 in
  let o_mul = value 8 shape_1_64 f16 in
  let o_add = value 9 shape_1_64 f16 in
  let linear_step = (linear_primitive, [ 0; 1 ], o_linear) in
  let silu_step operand output =
    (Kernel_region.Primitive.unary Kernel_region.Primitive.Silu, [ operand ], output)
  in
  let gelu_step operand output =
    (Kernel_region.Primitive.unary Kernel_region.Primitive.Gelu, [ operand ], output)
  in
  let add_step left right output =
    (Kernel_region.Primitive.binary Kernel_region.Primitive.Add, [ left; right ], output)
  in
  let mul_step left right output =
    (Kernel_region.Primitive.binary Kernel_region.Primitive.Mul, [ left; right ], output)
  in
  let inputs_2 = [ x; w ] in
  let inputs_4 = [ x; w; residual; up ] in

  Printf.printf "A. Chain-region equivalence vs the legacy single-slot fold\n%!";
  (* Bare linear: the fold returns the identity epilogue. *)
  chain_region_equiv ~name:"bare-linear" ~inputs:inputs_2
    ~steps:[ linear_step ] ~result_index:2 o_linear Epilogue.none;
  (* A. flat Linear + Add(residual): the residual-add tag. *)
  chain_region_equiv ~name:"flat-linear-add" ~inputs:inputs_4
    ~steps:[ linear_step; add_step 4 2 o_add ] ~result_index:5 o_add
    (Epilogue.residual_add ~residual);
  (* A'. Add with the external residual on the left: orientation must
     canonicalise to the same residual-add epilogue. *)
  chain_region_equiv ~name:"flipped-linear-add" ~inputs:inputs_4
    ~steps:[ linear_step; add_step 2 4 o_add ] ~result_index:5 o_add
    (Epilogue.residual_add ~residual);
  (* Linear -> Silu -> Add(residual): monoid [Expr] tree, accumulator first. *)
  chain_region_equiv ~name:"silu-residual-add" ~inputs:inputs_4
    ~steps:[ linear_step; silu_step 4 o_silu; add_step 5 2 o_add ]
    ~result_index:6 o_add
    (Epilogue.of_expr
       (Applicative_expr.Binop
          ( Applicative_expr.Add,
            Applicative_expr.Unop
              (Applicative_expr.Silu, Applicative_expr.Slot Epilogue.Acc),
            Applicative_expr.Read { value = residual; coord = Coordinate_map.identity } )));
  (* Same chain with the Add operands flipped. *)
  chain_region_equiv ~name:"silu-residual-add-flipped" ~inputs:inputs_4
    ~steps:[ linear_step; silu_step 4 o_silu; add_step 2 5 o_add ]
    ~result_index:6 o_add
    (Epilogue.of_expr
       (Applicative_expr.Binop
          ( Applicative_expr.Add,
            Applicative_expr.Unop
              (Applicative_expr.Silu, Applicative_expr.Slot Epilogue.Acc),
            Applicative_expr.Read { value = residual; coord = Coordinate_map.identity } )));
  (* C. Linear * up: total scale in pure [Expr] monoid form. *)
  chain_region_equiv ~name:"mul-up" ~inputs:inputs_4
    ~steps:[ linear_step; mul_step 4 3 o_mul ] ~result_index:5 o_mul
    (Epilogue.of_expr
       (Applicative_expr.Binop
          ( Applicative_expr.Mul,
            Applicative_expr.Slot Epilogue.Acc,
            Applicative_expr.Read { value = up; coord = Coordinate_map.identity } )));
  (* D. Linear -> Silu -> Mul(up) -> Add(residual). *)
  chain_region_equiv ~name:"silu-mul-residual" ~inputs:inputs_4
    ~steps:
      [ linear_step; silu_step 4 o_silu; mul_step 5 3 o_mul; add_step 6 2 o_add ]
    ~result_index:7 o_add
    (Epilogue.of_expr
       (Applicative_expr.Binop
          ( Applicative_expr.Add,
            Applicative_expr.Binop
              ( Applicative_expr.Mul,
                Applicative_expr.Unop
                  (Applicative_expr.Silu, Applicative_expr.Slot Epilogue.Acc),
                Applicative_expr.Read { value = up; coord = Coordinate_map.identity } ),
            Applicative_expr.Read { value = residual; coord = Coordinate_map.identity } )));
  (* H. Stacked residual adds stay associative in monoid [Expr] form. *)
  chain_region_equiv ~name:"double-residual-add" ~inputs:inputs_4
    ~steps:[ linear_step; add_step 4 2 o_add; add_step 5 3 o_linear2 ]
    ~result_index:6 o_linear2
    (Epilogue.of_expr
       (Applicative_expr.Binop
          ( Applicative_expr.Add,
            Applicative_expr.Binop
              ( Applicative_expr.Add,
                Applicative_expr.Slot Epilogue.Acc,
                Applicative_expr.Read { value = residual; coord = Coordinate_map.identity } ),
            Applicative_expr.Read { value = up; coord = Coordinate_map.identity } )));
  (* Bare Silu over the accumulator keeps the legacy tag. *)
  chain_region_equiv ~name:"silu-only" ~inputs:inputs_2
    ~steps:[ linear_step; silu_step 2 o_silu ] ~result_index:3 o_silu Epilogue.silu;
  (* Bare Gelu over the accumulator keeps the legacy tag. *)
  chain_region_equiv ~name:"gelu-only" ~inputs:inputs_2
    ~steps:[ linear_step; gelu_step 2 o_silu ] ~result_index:3 o_silu Epilogue.gelu;

  Printf.printf "B. Exercise (a): y*y single-slot square\n%!";
  (* The region computes out = L * L where L is the single linear output.
     The legacy fold rejects it because Mul consumes two folded operands;
     the elaborated group admits it as [Expr (Mul (Slot Acc, Slot Acc))]. *)
  let y2_region =
    single_result_region ~name:"y-square" ~inputs:inputs_2
      ~steps:[ linear_step; mul_step 2 2 o_mul ] ~result_index:3 o_mul
  in
  expect
    (Kernel_ir.Fused_tactic.fold_linear_epilogue y2_region = None)
    "y*y stays outside the legacy single-slot fold today";
  expect
    (Kernel_ir.Fused_tactic.select y2_region = None)
    "y*y stays outside the legacy tactic selection today";
  (match Contraction_group.of_region y2_region with
  | None -> fail "y*y region must elaborate into a group"
  | Some group -> (
      expect
        (List.length group.contractions = 1 && List.length group.exports = 1)
        "y*y group holds one contraction and one export";
      expect
        (Contraction_group.use_count group (Kernel_region.Slot.of_int_exn 2) = 2)
        "y*y fan-in: the contraction slot is consumed twice";
      match Contraction_group.single_acc_epilogue group with
      | None -> fail "y*y must reconstruct a single-accumulator epilogue"
      | Some epilogue -> (
          match epilogue with
          | Epilogue.Expr expr ->
              expect
                (Epilogue.equal epilogue
                   (Epilogue.of_expr
                      (Applicative_expr.Binop
                         ( Applicative_expr.Mul,
                           Applicative_expr.Slot Epilogue.Acc,
                           Applicative_expr.Slot Epilogue.Acc ))))
                "y*y normal form is Expr (Mul (Slot Acc, Slot Acc))";
              (match expr with
              | Applicative_expr.Binop
                  ( Applicative_expr.Mul,
                    Applicative_expr.Slot left,
                    Applicative_expr.Slot right ) ->
                  expect
                    (left = Epilogue.Acc && right = Epilogue.Acc)
                    "y*y references the accumulator on both sides"
              | _ ->
                  fail "y*y reconstruction is not structurally Mul (acc, acc)")
          | _ -> fail "y*y must not be a typed tag")));
  Printf.printf "  [PASS] y*y group\n%!";
  (* G4P12-02: the consumer-facing select/fold route through the elaboration
     widens admission to this single-slot branch shape: [select] returns one
     [Linear_epilogue] whose epilogue is the pure monoid
     [Expr (Mul (Acc, Acc))], and the widened [fold_linear_epilogue] returns
     the same epilogue. *)
  (match Contraction_group.select y2_region with
  | Some
      (Kernel_ir.Fused_tactic.Linear_epilogue
        { linear; epilogue; output }) ->
      expect
        (linear.Kernel_ir.Fused_tactic.m = 1
        && linear.Kernel_ir.Fused_tactic.n = 64
        && linear.Kernel_ir.Fused_tactic.k = 64
        && Ir.Value.equal output o_mul
        && Epilogue.equal epilogue
             (Epilogue.of_expr
                (Applicative_expr.Binop
                   ( Applicative_expr.Mul,
                     Applicative_expr.Slot Epilogue.Acc,
                     Applicative_expr.Slot Epilogue.Acc ))))
        "widened select admits y*y as Expr (Mul (Acc, Acc))"
  | Some _ -> fail "y*y widened select must be Linear_epilogue"
  | None -> fail "widened select rejected the y*y square");
  (match Contraction_group.fold_linear_epilogue y2_region with
  | Some epilogue ->
      expect
        (Epilogue.equal epilogue
           (Epilogue.of_expr
              (Applicative_expr.Binop
                 ( Applicative_expr.Mul,
                   Applicative_expr.Slot Epilogue.Acc,
                   Applicative_expr.Slot Epilogue.Acc ))))
        "widened fold yields the same monoid epilogue for y*y"
  | None -> fail "widened fold rejected the y*y square");
  Printf.printf "  [PASS] widened y*y select/fold\n%!";

  Printf.printf "C. Exercise (b): two projections + arbitrary pointwise\n%!";
  (* g = x*wg, u = x*wu over a shared input x, then a = silu(g), and
     z = a + u: an arbitrary pointwise op over two internal slots.  This is
     not the gated Mul template and not a single chain. *)
  let wg = value 10 shape_64_64 f16 in
  let wu = value 11 shape_64_64 f16 in
  let o_g = value 12 shape_1_64 f16 in
  let o_u = value 13 shape_1_64 f16 in
  let o_a = value 14 shape_1_64 f16 in
  let o_z = value 15 shape_1_64 f16 in
  let inputs_gated = [ x; wg; wu ] in
  let g_step = (linear_primitive, [ 0; 1 ], o_g) in
  let u_step = (linear_primitive, [ 0; 2 ], o_u) in
  let two_slot_region =
    single_result_region ~name:"two-slot-pointwise" ~inputs:inputs_gated
      ~steps:[ g_step; u_step; silu_step 3 o_a; add_step 5 4 o_z ]
      ~result_index:6 o_z
  in
  expect
    (Kernel_ir.Fused_tactic.fold_linear_epilogue two_slot_region = None)
    "two projections are outside the single-slot fold today";
  expect
    (Kernel_ir.Fused_tactic.select two_slot_region = None)
    "two projections are outside the legacy recognizers today";
  (match Contraction_group.of_region two_slot_region with
  | None -> fail "two-slot region must elaborate into a group"
  | Some group -> (
      expect
        (List.length group.contractions = 2)
        "two-slot group holds both projections";
      (* Shared input tensor and compatible domains. *)
      List.iter
        (fun contraction ->
          expect
            (Ir.Value.equal contraction.input x)
            "both projections read the shared activation input";
          expect
            (contraction.m = 1 && contraction.n = 64 && contraction.k = 64)
            "projection geometry is compatible";
          expect
            (Tensor_shape.equal contraction.shape shape_1_64)
            "projection domain is the fused tensor space")
        group.contractions;
      (* The arbitrary pointwise step combines both slots. *)
      (match
         Contraction_group.slot_expr group (Kernel_region.Slot.of_int_exn 6)
       with
      | Some
          (Binop (Applicative_expr.Add, Slot left, Slot right)) ->
          expect
            (Kernel_region.Slot.to_int left = 5 && Kernel_region.Slot.to_int right = 4)
            "pointwise add references both internal slots"
      | _ -> fail "two-slot pointwise step is not Add over both projection slots");
      (* Multi-contraction groups never collapse to a single accumulator. *)
      expect
        (Contraction_group.single_acc_epilogue group = None)
        "two-slot group has no single-accumulator epilogue";
      (* But they reconstruct a multi-accumulator expression tree. *)
      (match Contraction_group.multi_acc_epilogue group with
      | Some epilogue ->
          expect
            (Applicative_expr.equal ( = ) epilogue
               (Applicative_expr.Binop
                  ( Applicative_expr.Add,
                    Applicative_expr.Unop
                      (Applicative_expr.Silu, Applicative_expr.Slot 0),
                    Applicative_expr.Slot 1 )))
            "multi-accumulator expression is Add(Silu(0), 1)"
      | None -> fail "two-slot group must have multi-accumulator epilogue")));
  (* Test selection: select_contraction_group and widened select *)
  (match Contraction_group.select_contraction_group two_slot_region with
  | Some (Kernel_ir.Fused_tactic.Contraction_group { contractions; epilogue; output }) ->
      expect (List.length contractions = 2) "group tactic has 2 contractions";
      expect (Ir.Value.equal output o_z) "group tactic output matches";
      expect
        (Applicative_expr.equal ( = ) epilogue
           (Applicative_expr.Binop
              ( Applicative_expr.Add,
                Applicative_expr.Unop
                  (Applicative_expr.Silu, Applicative_expr.Slot 0),
                Applicative_expr.Slot 1 )))
        "group tactic epilogue is Add(Silu(0), 1)"
  | _ -> fail "select_contraction_group must select two_slot_region");
  (match Contraction_group.select two_slot_region with
  | Some (Kernel_ir.Fused_tactic.Contraction_group { contractions; _ }) ->
      expect (List.length contractions = 2) "widened select admits two_slot_region as Contraction_group"
  | _ -> fail "widened select must return Contraction_group for two_slot_region");
  (* Gated product (SwiGLU) region: g = x*wg, u = x*wu, a = silu(g), z = a * u *)
  let swiglu_region =
    single_result_region ~name:"two-slot-swiglu" ~inputs:inputs_gated
      ~steps:[ g_step; u_step; silu_step 3 o_a; mul_step 5 4 o_z ]
      ~result_index:6 o_z
  in
  (match Contraction_group.select_contraction_group swiglu_region with
  | Some (Kernel_ir.Fused_tactic.Contraction_group { contractions; epilogue; output }) ->
      expect (List.length contractions = 2) "SwiGLU group has 2 contractions";
      expect (Ir.Value.equal output o_z) "SwiGLU group output matches";
      expect
        (Applicative_expr.equal ( = ) epilogue
           (Applicative_expr.Binop
              ( Applicative_expr.Mul,
                Applicative_expr.Unop
                  (Applicative_expr.Silu, Applicative_expr.Slot 0),
                Applicative_expr.Slot 1 )))
        "SwiGLU epilogue is Mul(Silu(0), 1)"
  | _ -> fail "select_contraction_group must select swiglu_region");
  Printf.printf "  [PASS] two-slot group\n%!";

  Printf.printf "D. Exercise (c): exported intermediate + fused consumer\n%!";
  (* S = silu(L) is exported AND consumed by M = S * up, then R = M +
     residual.  The group has two results and one explicit use obligation on
     the shared intermediate S. *)
  let o_r = value 16 shape_1_64 f16 in
  let export_region =
    make_region ~name:"exported-intermediate" ~inputs:inputs_4
      ~steps:
        [ linear_step; silu_step 4 o_silu; mul_step 5 3 o_mul; add_step 6 2 o_r ]
      ~results:[ (5, o_silu); (7, o_r) ]
  in
  expect
    (Kernel_ir.Fused_tactic.fold_linear_epilogue export_region = None)
    "two-result regions stay outside the single-slot fold today";
  (match Contraction_group.of_region export_region with
  | None -> fail "exported-intermediate region must elaborate into a group"
  | Some group -> (
      expect
        (List.length group.contractions = 1)
        "exported-intermediate group holds the single linear";
      expect
        (List.length group.exports = 2)
        "exported-intermediate group holds both region results";
      let slot5 = Kernel_region.Slot.of_int_exn 5 in
      let slot7 = Kernel_region.Slot.of_int_exn 7 in
      expect
        (Contraction_group.is_exported group slot5)
        "the intermediate S escapes as a region result";
      expect
        (Contraction_group.is_exported group slot7)
        "the fused consumer R escapes as a region result";
      expect
        (Contraction_group.use_count group slot5 = 1)
        "S is consumed exactly once inside the group";
      expect
        (Contraction_group.single_acc_epilogue group = None)
        "two-export group has no single-accumulator epilogue";
      );
      Printf.printf "  [PASS] exported-intermediate group\n%!");

  Printf.printf "E. Fan-out and conservative cuts\n%!";
  (* Multi-export fan-out (two consumers of one Silu step) forms a group with
     a two-fold use obligation, though it never reconstructs single-acc. *)
  let fanout_region =
    make_region ~name:"fanout-two-results" ~inputs:inputs_4
      ~steps:
        [ linear_step; silu_step 4 o_silu; add_step 5 2 o_add; mul_step 5 3 o_mul ]
      ~results:[ (6, o_add); (7, o_mul) ]
  in
  expect
    (Kernel_ir.Fused_tactic.fold_linear_epilogue fanout_region = None)
    "fan-out stays outside the single-slot fold today";
  (match Contraction_group.of_region fanout_region with
  | None -> fail "fan-out region must elaborate into a group"
  | Some group ->
      expect
        (Contraction_group.use_count group (Kernel_region.Slot.of_int_exn 5) = 2)
        "fan-out records the two-fold use obligation");
  Printf.printf "  [PASS] fan-out group\n%!";
  (* Sigmoid leaves the scalar grammar: the region stays outside groups. *)
  let o_sigmoid = value 17 shape_1_64 f16 in
  let sigmoid_region =
    single_result_region ~name:"sigmoid-cut" ~inputs:inputs_4
      ~steps:
        [
          linear_step;
          ( Kernel_region.Primitive.unary Kernel_region.Primitive.Sigmoid,
            [ 4 ],
            o_sigmoid );
        ]
      ~result_index:5 o_sigmoid
  in
  expect
    (Contraction_group.of_region sigmoid_region = None)
    "sigmoid chains stay outside group formation";
  (* An unreferenced binding (a second linear that feeds no result) keeps the
     region outside group formation, mirroring the fold's whole-region rule. *)
  let unreferenced_region =
    single_result_region ~name:"unreferenced-binding" ~inputs:inputs_2
      ~steps:[ linear_step; (linear_primitive, [ 0; 1 ], o_linear2) ]
      ~result_index:3 o_linear2
  in
  expect
    (Kernel_ir.Fused_tactic.fold_linear_epilogue unreferenced_region = None)
    "legacy fold rejects the unreferenced-binding region";
  expect
    (Contraction_group.of_region unreferenced_region = None)
    "unreferenced bindings keep the region outside group formation";
  (* Cast is inside the scalar grammar, so a Cast chain elaborates (a
     documented superset of the legacy fold, which cuts on casts).  The cast
     target dtype is Float32, so the output tensor must be Float32. *)
  let o_cast = value 18 shape_1_64 Ir.Dtype.Float32 in
  let cast_region =
    single_result_region ~name:"cast-chain" ~inputs:inputs_4
      ~steps:
        [
          linear_step;
          (Kernel_region.Primitive.cast Ir.Dtype.Float32, [ 4 ], o_cast);
        ]
      ~result_index:5 o_cast
  in
  expect
    (Kernel_ir.Fused_tactic.fold_linear_epilogue cast_region = None)
    "legacy fold cuts on Cast today";
  expect
    (match Contraction_group.of_region cast_region with
    | Some _ -> true
    | None -> false)
    "Cast chains elaborate (Cast is inside the scalar grammar)";
  (match Contraction_group.of_region cast_region with
  | None -> fail "cast chain must elaborate into a group"
  | Some group -> (
      match Contraction_group.single_acc_epilogue group with
      | Some epilogue ->
          expect
            (Epilogue.equal epilogue
               (Epilogue.of_expr
                  (Applicative_expr.Unop
                     ( Applicative_expr.Cast Ir.Dtype.Float32,
                       Applicative_expr.Slot Epilogue.Acc ))))
            "cast chain reconstructs Expr (Cast (Float32, acc))"
      | None -> fail "cast chain must reconstruct a single-acc Expr epilogue"));
  Printf.printf "  [PASS] conservative cuts\n%!";

  Printf.printf "F. Multi-slot parametric kernel emission (G4P12-06-2)\n%!";
  let q4_0_schema =
    match
      Block_schema.of_storage (Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_0)
    with
    | Ok schema -> schema
    | Error err -> fail (Block_schema.Error.to_string err)
  in
  let q8_0_schema =
    match
      Block_schema.of_storage (Ir.Linear_storage.Block_quantized Ir.Dtype.Q8_0)
    with
    | Ok schema -> schema
    | Error err -> fail (Block_schema.Error.to_string err)
  in
  let q4_k_schema =
    match
      Block_schema.of_storage (Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_K)
    with
    | Ok schema -> schema
    | Error err -> fail (Block_schema.Error.to_string err)
  in
  let contains_substr haystack needle =
    let hlen = String.length haystack in
    let nlen = String.length needle in
    let rec check i =
      if i + nlen > hlen then false
      else if String.sub haystack i nlen = needle then true
      else check (i + 1)
    in
    if nlen = 0 then true else check 0
  in
  let compile_msl_kernel kernel_text =
    let preamble =
      "#include <metal_stdlib>\n\
       using namespace metal;\n\n\
       struct QuantLinearParams {\n\
           uint m;\n\
           uint n;\n\
           uint k;\n\
           uint has_bias;\n\
       };\n\n"
    in
    let full_src = preamble ^ kernel_text in
    let tmp_src = Filename.temp_file "test_multi_linear" ".metal" in
    let tmp_air = Filename.temp_file "test_multi_linear" ".air" in
    let oc = open_out tmp_src in
    output_string oc full_src;
    close_out oc;
    let cmd =
      Printf.sprintf "xcrun -sdk macosx metal -c %s -o %s >/dev/null 2>&1"
        (Filename.quote tmp_src) (Filename.quote tmp_air)
    in
    let ok = Sys.command cmd = 0 in
    (try Sys.remove tmp_src with _ -> ());
    (try Sys.remove tmp_air with _ -> ());
    ok
  in
  (* 1. SwiGLU two-slot kernel *)
  let swiglu_epilogue =
    Applicative_expr.Binop
      ( Applicative_expr.Mul,
        Applicative_expr.Unop (Applicative_expr.Silu, Applicative_expr.Slot 0),
        Applicative_expr.Slot 1 )
  in
  let swiglu_kernel_name =
    Contraction_group.algebraic_kernel_name
      ~base:"llmopt_q4_0_multi_linear_f16_x8" swiglu_epilogue
  in
  let swiglu_src =
    expect_ok
      (Metal_block_codegen.q4_0_multi_linear_f16_x8_kernel
         ~name:swiglu_kernel_name ~epilogue:swiglu_epilogue
         ~target:Target_hardware.default q4_0_schema)
  in
  expect
    (contains_substr swiglu_src ("kernel void " ^ swiglu_kernel_name))
    "swiglu kernel name matches digest";
  expect
    (contains_substr swiglu_src "device const uchar* weight_0 [[buffer(1)]]")
    "swiglu buffer 1 is weight_0";
  expect
    (contains_substr swiglu_src "device const uchar* weight_1 [[buffer(2)]]")
    "swiglu buffer 2 is weight_1";
  expect
    (contains_substr swiglu_src "device half* output [[buffer(3)]]")
    "swiglu buffer 3 is output";
  expect
    (contains_substr swiglu_src "constant QuantLinearParams& params [[buffer(4)]]")
    "swiglu buffer 4 is params";
  expect
    (contains_substr swiglu_src "acc0_0 = simd_sum(acc0_0);")
    "swiglu slot 0 reduction";
  expect
    (contains_substr swiglu_src "acc1_0 = simd_sum(acc1_0);")
    "swiglu slot 1 reduction";
  expect
    (contains_substr swiglu_src
       "out_row[base_col] = half(((acc0_0 * (1.0f / (1.0f + metal::exp(-acc0_0)))) * acc1_0));")
    "swiglu writeback matches Silu(0)*1";
  expect
    (compile_msl_kernel swiglu_src)
    "swiglu multi-slot kernel compiles cleanly with xcrun metal";
  Printf.printf "  [PASS] two-slot SwiGLU parametric emission\n%!";

  (* 2. Two-slot arbitrary pointwise silu(g) + u *)
  let add_epilogue =
    Applicative_expr.Binop
      ( Applicative_expr.Add,
        Applicative_expr.Unop (Applicative_expr.Silu, Applicative_expr.Slot 0),
        Applicative_expr.Slot 1 )
  in
  let add_src =
    expect_ok
      (Metal_block_codegen.q4_0_multi_linear_f16_x8_kernel
         ~name:"test_silu_add" ~epilogue:add_epilogue
         ~target:Target_hardware.default q4_0_schema)
  in
  expect
    (contains_substr add_src
       "out_row[base_col] = half(((acc0_0 * (1.0f / (1.0f + metal::exp(-acc0_0)))) + acc1_0));")
    "silu_add writeback matches Silu(0)+1";
  expect
    (compile_msl_kernel add_src)
    "silu_add multi-slot kernel compiles cleanly with xcrun metal";
  Printf.printf "  [PASS] two-slot arbitrary pointwise parametric emission\n%!";

  (* 3. Two-slot GeGLU with auxiliary buffer (residual add) *)
  let residual =
    Ir.Value.make_tensor ~id:99 ~shape:Fixture.shape_1_64 ~dtype:Ir.Dtype.Float16
  in
  let geglu_res_epilogue =
    Applicative_expr.Binop
      ( Applicative_expr.Add,
        Applicative_expr.Binop
          ( Applicative_expr.Mul,
            Applicative_expr.Unop (Applicative_expr.Gelu, Applicative_expr.Slot 0),
            Applicative_expr.Slot 1 ),
        Applicative_expr.Read
          { value = residual; coord = Coordinate_map.identity } )
  in
  let res_src =
    expect_ok
      (Metal_block_codegen.q4_0_multi_linear_f16_x8_kernel
         ~name:"test_geglu_residual" ~epilogue:geglu_res_epilogue
         ~target:Target_hardware.default q4_0_schema)
  in
  expect
    (contains_substr res_src "device const half* buf_0 [[buffer(3)]]")
    "geglu residual has auxiliary buffer 3";
  expect
    (contains_substr res_src "device half* output [[buffer(4)]]")
    "geglu residual has output buffer 4";
  expect
    (contains_substr res_src "constant QuantLinearParams& params [[buffer(5)]]")
    "geglu residual has params buffer 5";
  expect
    (contains_substr res_src "in_buf_0")
    "geglu residual indexes auxiliary buffer via in_buf_0";
  expect
    (compile_msl_kernel res_src)
    "geglu residual multi-slot kernel compiles cleanly with xcrun metal";
  Printf.printf "  [PASS] two-slot GeGLU + residual auxiliary buffer emission\n%!";

  (* 4. Three-slot contraction group *)
  let three_slot_epilogue =
    Applicative_expr.Binop
      ( Applicative_expr.Add,
        Applicative_expr.Binop
          ( Applicative_expr.Add,
            Applicative_expr.Slot 0,
            Applicative_expr.Slot 1 ),
        Applicative_expr.Slot 2 )
  in
  let three_src =
    expect_ok
      (Metal_block_codegen.q4_0_multi_linear_f16_x8_kernel
         ~name:"test_three_slot" ~epilogue:three_slot_epilogue ~num_slots:3
         ~target:Target_hardware.default q4_0_schema)
  in
  expect
    (contains_substr three_src "device const uchar* weight_0 [[buffer(1)]]")
    "three-slot has weight_0";
  expect
    (contains_substr three_src "device const uchar* weight_1 [[buffer(2)]]")
    "three-slot has weight_1";
  expect
    (contains_substr three_src "device const uchar* weight_2 [[buffer(3)]]")
    "three-slot has weight_2";
  expect
    (contains_substr three_src "device half* output [[buffer(4)]]")
    "three-slot has output buffer 4";
  expect
    (contains_substr three_src "constant QuantLinearParams& params [[buffer(5)]]")
    "three-slot has params buffer 5";
  expect
    (contains_substr three_src "acc2_0 = simd_sum(acc2_0);")
    "three-slot slot 2 reduction";
  expect
    (contains_substr three_src
       "out_row[base_col] = half(((acc0_0 + acc1_0) + acc2_0));")
    "three-slot writeback matches (0+1)+2";
  expect
    (compile_msl_kernel three_src)
    "three-slot multi-slot kernel compiles cleanly with xcrun metal";
  Printf.printf "  [PASS] three-slot parametric emission\n%!";

  (* 5. Q8_0 two-slot SwiGLU kernel *)
  let q8_swiglu_name =
    Contraction_group.algebraic_kernel_name
      ~base:"llmopt_q8_0_linear_f16" swiglu_epilogue
  in
  let q8_swiglu_src =
    expect_ok
      (Metal_block_codegen.q8_0_multi_linear_kernel
         ~name:q8_swiglu_name ~epilogue:swiglu_epilogue
         ~target:Target_hardware.default q8_0_schema)
  in
  let q8_linear_src =
    expect_ok
      (Metal_block_codegen.linear_kernels ~target:Target_hardware.default q8_0_schema)
  in
  expect
    (contains_substr q8_swiglu_src ("kernel void " ^ q8_swiglu_name))
    "q8_0 swiglu kernel name matches digest";
  expect
    (contains_substr q8_swiglu_src "device const uchar* weight_0 [[buffer(1)]]")
    "q8_0 swiglu buffer 1 is weight_0";
  expect
    (contains_substr q8_swiglu_src "device const uchar* weight_1 [[buffer(2)]]")
    "q8_0 swiglu buffer 2 is weight_1";
  expect
    (contains_substr q8_swiglu_src "device half* output [[buffer(3)]]")
    "q8_0 swiglu buffer 3 is output";
  expect
    (contains_substr q8_swiglu_src "constant QuantLinearParams& params [[buffer(4)]]")
    "q8_0 swiglu buffer 4 is params";
  expect
    (contains_substr q8_swiglu_src "llmopt_schema_q8_0_value(row_blocks_0")
    "q8_0 swiglu calls llmopt_schema_q8_0_value for slot 0";
  expect
    (contains_substr q8_swiglu_src "llmopt_schema_q8_0_value(row_blocks_1")
    "q8_0 swiglu calls llmopt_schema_q8_0_value for slot 1";
  expect
    (contains_substr q8_swiglu_src "acc0 = simd_sum(acc0);")
    "q8_0 swiglu reduces slot 0";
  expect
    (contains_substr q8_swiglu_src "acc1 = simd_sum(acc1);")
    "q8_0 swiglu reduces slot 1";
  expect
    (compile_msl_kernel (q8_linear_src ^ q8_swiglu_src))
    "q8_0 swiglu kernel compiles cleanly with xcrun metal";
  Printf.printf "  [PASS] Q8_0 two-slot SwiGLU parametric emission\n%!";

  (* 6. Q4_K two-slot GeGLU + residual kernel *)
  let q4_k_res_name =
    Contraction_group.algebraic_kernel_name
      ~base:"llmopt_q4_k_linear_f16" geglu_res_epilogue
  in
  let q4_k_res_src =
    expect_ok
      (Metal_block_codegen.q4_k_multi_linear_kernel
         ~name:q4_k_res_name ~epilogue:geglu_res_epilogue
         ~target:Target_hardware.default q4_k_schema)
  in
  let q4_k_linear_src =
    expect_ok
      (Metal_block_codegen.linear_kernels ~target:Target_hardware.default q4_k_schema)
  in
  expect
    (contains_substr q4_k_res_src ("kernel void " ^ q4_k_res_name))
    "q4_k geglu residual kernel declaration matches";
  expect
    (contains_substr q4_k_res_src "device const uchar* weight_0 [[buffer(1)]]")
    "q4_k geglu residual has weight_0";
  expect
    (contains_substr q4_k_res_src "device const uchar* weight_1 [[buffer(2)]]")
    "q4_k geglu residual has weight_1";
  expect
    (contains_substr q4_k_res_src "device const half* buf_0 [[buffer(3)]]")
    "q4_k geglu residual has auxiliary buffer buf_0";
  expect
    (contains_substr q4_k_res_src "device half* output [[buffer(4)]]")
    "q4_k geglu residual has output buffer 4";
  expect
    (contains_substr q4_k_res_src "constant QuantLinearParams& params [[buffer(5)]]")
    "q4_k geglu residual has params buffer 5";
  expect
    (contains_substr q4_k_res_src "llmopt_schema_q4_k_value(packed_0")
    "q4_k geglu residual calls llmopt_schema_q4_k_value for slot 0";
  expect
    (contains_substr q4_k_res_src "llmopt_schema_q4_k_value(packed_1")
    "q4_k geglu residual calls llmopt_schema_q4_k_value for slot 1";
  expect
    (contains_substr q4_k_res_src "in_buf_0[column]")
    "q4_k geglu residual reads auxiliary residual buffer";
  expect
    (compile_msl_kernel (q4_k_linear_src ^ q4_k_res_src))
    "q4_k geglu residual kernel compiles cleanly with xcrun metal";
  Printf.printf "  [PASS] Q4_K two-slot GeGLU + residual superblock parametric emission\n%!";

  Printf.printf "G. G4P12-07: multi-export contraction group selection\n%!";
  (* Two-contraction, two-export region:
     g = x*wg, u = x*wu, a = silu(g), z = a + u.
     Export both a (the intermediate) and z (the final result).
     This requires 2 contractions sharing input x and 2 exports. *)
  let multi_export_g = value 20 shape_1_64 f16 in
  let multi_export_u = value 21 shape_1_64 f16 in
  let multi_export_a = value 22 shape_1_64 f16 in
  let multi_export_z = value 23 shape_1_64 f16 in
  let multi_export_region =
    make_region ~name:"multi-export-fanout" ~inputs:inputs_gated
      ~steps:
        [ (linear_primitive, [ 0; 1 ], multi_export_g);
          (linear_primitive, [ 0; 2 ], multi_export_u);
          silu_step 3 multi_export_a;
          add_step 5 4 multi_export_z ]
      ~results:[ (5, multi_export_a); (6, multi_export_z) ]
  in
  (match Contraction_group.of_region multi_export_region with
  | None -> fail "multi-export region must elaborate into a group"
  | Some group ->
      expect
        (List.length group.contractions = 2)
        "multi-export group has 2 contractions";
      expect
        (List.length group.exports = 2)
        "multi-export group has 2 exports";
      (match Contraction_group.multi_export_epilogues group with
      | Some epilogues ->
          expect
            (List.length epilogues = 2)
            "multi_export_epilogues returns 2 epilogues";
          (* First export is a = silu(g) = Silu(Slot 0) *)
          (match List.hd epilogues with
          | Applicative_expr.Unop
              (Applicative_expr.Silu, Applicative_expr.Slot 0) -> ()
          | _ -> fail "first epilogue must be Silu(Slot 0)");
          (* Second export is z = silu(g) + u = Add(Silu(Slot 0), Slot 1) *)
          (match List.nth epilogues 1 with
          | Applicative_expr.Binop
              ( Applicative_expr.Add,
                Applicative_expr.Unop
                  (Applicative_expr.Silu, Applicative_expr.Slot 0),
                Applicative_expr.Slot 1 ) -> ()
          | _ -> fail "second epilogue must be Add(Silu(Slot 0), Slot 1)")
      | None -> fail "multi_export_epilogues must return Some for 2-contraction 2-export group"));
  (* select_contraction_group must now return Multi_export_contraction_group *)
  (match Contraction_group.select_contraction_group multi_export_region with
  | Some (Kernel_ir.Fused_tactic.Multi_export_contraction_group
             { contractions; epilogues; outputs }) ->
      expect (List.length contractions = 2) "multi-export tactic has 2 contractions";
      expect (List.length epilogues = 2) "multi-export tactic has 2 epilogues";
      expect (List.length outputs = 2) "multi-export tactic has 2 outputs";
      expect
        (Ir.Value.equal (List.hd outputs) multi_export_a)
        "first output is the intermediate a";
      expect
        (Ir.Value.equal (List.nth outputs 1) multi_export_z)
        "second output is the final z"
  | Some _ -> fail "select_contraction_group must return Multi_export_contraction_group"
  | None -> fail "select_contraction_group must select multi-export region");
  (* Widened select must also return Multi_export_contraction_group *)
  (match Contraction_group.select multi_export_region with
  | Some (Kernel_ir.Fused_tactic.Multi_export_contraction_group _) -> ()
  | Some _ -> fail "widened select must return Multi_export_contraction_group for multi-export region"
  | None -> fail "widened select must admit multi-export region");
  Printf.printf "  [PASS] multi-export contraction group selection\n%!";

  Printf.printf "H. G4P12-07: multi-export kernel emission\n%!";
  (* Multi-export kernel: two epilogues (Silu(0) and Add(Silu(0), 1)),
     two output buffers.  The kernel must compile cleanly. *)
  let me_epilogue_0 =
    Applicative_expr.Unop (Applicative_expr.Silu, Applicative_expr.Slot 0)
  in
  let me_epilogue_1 =
    Applicative_expr.Binop
      ( Applicative_expr.Add,
        Applicative_expr.Unop (Applicative_expr.Silu, Applicative_expr.Slot 0),
        Applicative_expr.Slot 1 )
  in
  let me_kernel_name =
    Contraction_group.algebraic_kernel_name
      ~base:"llmopt_q4_0_multi_export_f16_x8" me_epilogue_1
  in
  let me_src =
    expect_ok
      (Metal_block_codegen.q4_0_multi_export_linear_f16_x8_kernel
         ~name:me_kernel_name ~epilogues:[ me_epilogue_0; me_epilogue_1 ]
         ~target:Target_hardware.default q4_0_schema)
  in

  let () =
    expect
      (contains_substr me_src ("kernel void " ^ me_kernel_name))
      "multi-export kernel name matches digest";
    expect
      (contains_substr me_src "device const uchar* weight_0 [[buffer(1)]]")
      "multi-export buffer 1 is weight_0";
    expect
      (contains_substr me_src "device const uchar* weight_1 [[buffer(2)]]")
      "multi-export buffer 2 is weight_1";
    expect
      (contains_substr me_src "device half* output_0 [[buffer(3)]]")
      "multi-export buffer 3 is output_0";
    expect
      (contains_substr me_src "device half* output_1 [[buffer(4)]]")
      "multi-export buffer 4 is output_1";
    expect
      (contains_substr me_src "constant QuantLinearParams& params [[buffer(5)]]")
      "multi-export buffer 5 is params";
    expect
      (contains_substr me_src "out_row_0[base_col]")
      "multi-export has writeback for output_0";
    expect
      (contains_substr me_src "out_row_1[base_col]")
      "multi-export has writeback for output_1";
    expect
      (compile_msl_kernel me_src)
      "multi-export kernel compiles cleanly with xcrun metal";
    Printf.printf "  [PASS] multi-export kernel emission
%!"
  in

  let () = Printf.printf "I. Multi-export block32 & superblock kernel emission
%!" in
  let me_kernel_q8_name =
    Contraction_group.algebraic_kernel_name
      ~base:"llmopt_q8_0_multi_export_f16" me_epilogue_1
  in
  let me_q8_src =
    expect_ok
      (Metal_block_codegen.q8_0_multi_export_linear_kernel
         ~name:me_kernel_q8_name ~epilogues:[ me_epilogue_0; me_epilogue_1 ]
         ~target:Target_hardware.default q8_0_schema)
  in
  let () = expect (compile_msl_kernel (q8_linear_src ^ me_q8_src)) "Q8_0 multi-export kernel compiles cleanly with xcrun metal" in

  let me_kernel_q4k_name =
    Contraction_group.algebraic_kernel_name
      ~base:"llmopt_q4_k_multi_export_f16" me_epilogue_1
  in
  let me_q4k_src =
    expect_ok
      (Metal_block_codegen.q4_k_multi_export_linear_kernel
         ~name:me_kernel_q4k_name ~epilogues:[ me_epilogue_0; me_epilogue_1 ]
         ~target:Target_hardware.default q4_k_schema)
  in
  let () =
    expect (compile_msl_kernel (q4_k_linear_src ^ me_q4k_src)) "Q4_K multi-export kernel compiles cleanly with xcrun metal";
    Printf.printf "  [PASS] multi-export block32 & superblock kernel emission\n%!"
  in

  let () = Printf.printf "J. Multi-export schedule, memory plan, and dispatch assignment
%!" in
  let graph = Ir.Graph.create () in
  let val_x = Ir.Graph.fresh_tensor_value graph ~shape:shape_1_64 ~dtype:Ir.Dtype.Float16 in
  let val_w1 = Ir.Graph.fresh_tensor_value graph ~shape:shape_64_64 ~dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0) in
  let val_w2 = Ir.Graph.fresh_tensor_value graph ~shape:shape_64_64 ~dtype:(Ir.Dtype.Quant Ir.Dtype.Q4_0) in
  let () =
    Ir.Graph.append graph ~op:(Ir.Op.Input { name = "x"; source = Ir.Input_source.Runtime }) ~inputs:[] ~output:(Some val_x);
    Ir.Graph.append graph ~op:(Ir.Op.Input { name = "w1"; source = Ir.Input_source.Runtime }) ~inputs:[] ~output:(Some val_w1);
    Ir.Graph.append graph ~op:(Ir.Op.Input { name = "w2"; source = Ir.Input_source.Runtime }) ~inputs:[] ~output:(Some val_w2)
  in

  let val_g = Ir.Graph.fresh_tensor_value graph ~shape:shape_1_64 ~dtype:Ir.Dtype.Float16 in
  let () = Ir.Graph.append graph ~op:(Ir.Op.Linear { m = 1; n = 64; k = 64; bias = false }) ~inputs:[ val_x; val_w1 ] ~output:(Some val_g) in
  let val_u = Ir.Graph.fresh_tensor_value graph ~shape:shape_1_64 ~dtype:Ir.Dtype.Float16 in
  let () = Ir.Graph.append graph ~op:(Ir.Op.Linear { m = 1; n = 64; k = 64; bias = false }) ~inputs:[ val_x; val_w2 ] ~output:(Some val_u) in
  let val_a = Ir.Graph.fresh_tensor_value graph ~shape:shape_1_64 ~dtype:Ir.Dtype.Float16 in
  let () = Ir.Graph.append graph ~op:(Ir.Op.Primitive (Ir.Primitive.Pointwise (Ir.Pointwise.Unary (Ir.Pointwise.Silu, val_g)))) ~inputs:[ val_g ] ~output:(Some val_a) in
  let val_z = Ir.Graph.fresh_tensor_value graph ~shape:shape_1_64 ~dtype:Ir.Dtype.Float16 in
  let () = Ir.Graph.append graph ~op:(Ir.Op.Primitive (Ir.Primitive.Pointwise (Ir.Pointwise.Binary (Ir.Pointwise.Add, Ir.Pointwise.Tensor val_a, Ir.Pointwise.Tensor val_u)))) ~inputs:[ val_a; val_u ] ~output:(Some val_z) in

  let () =
    Ir.Graph.append graph ~op:(Ir.Op.Output { name = "out_a" }) ~inputs:[ val_a ] ~output:None;
    Ir.Graph.append graph ~op:(Ir.Op.Output { name = "out_z" }) ~inputs:[ val_z ] ~output:None
  in

  let find_node_id_by_output out_val =
    Ir.Graph.nodes graph
    |> List.find_map (fun node ->
           match Ir.node_output node with
           | Some out when Ir.Value.equal out out_val -> Some (Ir.node_id node)
           | _ -> None)
    |> Option.get
  in
  let node_g = find_node_id_by_output val_g in
  let node_u = find_node_id_by_output val_u in
  let node_a = find_node_id_by_output val_a in
  let node_z = find_node_id_by_output val_z in

  let me_sched_region =
    make_region_with_members ~member_node_ids:[ node_g; node_u; node_a; node_z ]
      ~name:"multi-export-schedule-region"
      ~inputs:[ val_x; val_w1; val_w2 ]
      ~steps:
        [ (Kernel_region.Primitive.linear ~m:1 ~n:64 ~k:64 ~bias:false
             ~storage:(Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_0),
           [ 0; 1 ], val_g);
          (Kernel_region.Primitive.linear ~m:1 ~n:64 ~k:64 ~bias:false
             ~storage:(Ir.Linear_storage.Block_quantized Ir.Dtype.Q4_0),
           [ 0; 2 ], val_u);
          silu_step 3 val_a;
          add_step 5 4 val_z ]
      ~results:[ (5, val_a); (6, val_z) ]
  in

  let schedule =
    Serving_schedule.of_graph ~fusion_regions:[ me_sched_region ] graph
    |> expect_ok
  in
  let commands = Serving_schedule.commands schedule in
  let carrier_opt =
    List.find_opt
      (fun cmd ->
        match Serving_schedule.Command.op cmd with
        | Ir.Op.Fused_region { extra_outputs } ->
            List.exists (Ir.Value.equal val_z) extra_outputs
            && (match Serving_schedule.Command.output cmd with Some out -> Ir.Value.equal val_a out | None -> false)
        | _ -> false)
      commands
  in
  let () = expect (Option.is_some carrier_opt) "carrier command with Fused_region { extra_outputs = [val_z] } and output = Some val_a exists in schedule" in
  let carrier = Option.get carrier_opt in
  let () = expect (Serving_schedule.Command.fusion_region carrier = Some me_sched_region) "carrier command carries the fusion region" in
  let () = expect (Serving_schedule.Command.node_id carrier = node_a) "carrier inherits the matched node id" in

  (* Verify member node collapsing: node_g, node_u, node_z should not be separate commands *)
  let collapsed_command_count =
    List.filter
      (fun cmd ->
        let id = Serving_schedule.Command.node_id cmd in
        id = node_g || id = node_u || id = node_z)
      commands
    |> List.length
  in
  let () = expect (collapsed_command_count = 0) "non-carrier member nodes were collapsed out of schedule commands" in

  let () = expect (Result.is_ok (Serving_schedule.of_graph ~fusion_regions:[ me_sched_region ] graph)) "multi-export fusion command passes validation" in

  let roundtrip_sched =
    schedule |> Serving_schedule.to_bytes |> Serving_schedule.of_bytes |> expect_ok
  in
  let () =
    expect
      (Serving_schedule.commands roundtrip_sched
       |> List.exists (fun cmd ->
              match Serving_schedule.Command.op cmd with
              | Ir.Op.Fused_region { extra_outputs } ->
                  List.exists (Ir.Value.equal val_z) extra_outputs
              | _ -> false))
      "Fused_region survives schedule serialization codec roundtrip"
  in

  let memory_plan = Serving_memory_plan.create schedule |> expect_ok in
  let () =
    expect (Option.is_some (Serving_memory_plan.allocation memory_plan val_a)) "memory plan allocates storage for primary output val_a";
    expect (Option.is_some (Serving_memory_plan.allocation memory_plan val_z)) "memory plan allocates storage for extra output val_z";
    expect (Serving_memory_plan.check_disjoint memory_plan [ val_a; val_z ]) "memory plan allocations for val_a and val_z are disjoint"
  in

  let () =
    match Metal_dispatch_assignment.assignment_execution carrier with
    | Some (Metal_dispatch_assignment.Assignment_multi_export_contraction_group { contractions; epilogues; outputs }) ->
        expect (List.length contractions = 2) "assigned multi-export has 2 contractions";
        expect (List.length epilogues = 2) "assigned multi-export has 2 epilogues";
        expect (List.length outputs = 2) "assigned multi-export has 2 outputs";
        expect (Ir.Value.equal (List.hd outputs) val_a) "first output is val_a";
        expect (Ir.Value.equal (List.nth outputs 1) val_z) "second output is val_z"
    | _ -> fail "assignment_execution on carrier must yield Assignment_multi_export_contraction_group"
  in

  let metal_program = Metal.lower ~fusion_regions:[ me_sched_region ] graph |> expect_ok in
  let p_src = Metal.Program.source metal_program in
  let () =
    expect (contains_substr p_src "kernel void llmopt_q4_0_linear_f16_x8") "Metal program contains the specialized multi-export kernel";
    Printf.printf "  [PASS] multi-export schedule, memory plan, and dispatch assignment\n%!"
  in

  let () = Printf.printf "K. Multi-export on-device Metal execution and numerical parity\n%!" in
  let tmp_dir = Filename.temp_file "metal_me_test" "" in
  (try Sys.remove tmp_dir with _ -> ());
  Unix.mkdir tmp_dir 0o755;
  let metal_file = Filename.concat tmp_dir "kernel.metal" in
  let air_file = Filename.concat tmp_dir "kernel.air" in
  let metallib_file = Filename.concat tmp_dir "kernel.metallib" in
  let oc = open_out metal_file in
  output_string oc p_src;
  close_out oc;
  let compile_cmd =
    Printf.sprintf "xcrun -sdk macosx metal -c %s -o %s && xcrun -sdk macosx metallib %s -o %s"
      (Filename.quote metal_file) (Filename.quote air_file)
      (Filename.quote air_file) (Filename.quote metallib_file)
  in
  let () = expect (Sys.command compile_cmd = 0) "compiled multi-export kernel to metallib" in
  let artifact = expect_ok (Serving_package.Artifact.create "kernel.metallib") in
  let files = Serving_package.Files.create ~metal_library:artifact in
  let package =
    Serving_package.compiled_graph ~files
      ~kernels:(Metal.Program.kernels metal_program)
      ~schedule
      ~cache:Serving_package.Cache.default ()
    |> expect_ok
  in
  let runtime = Metal_runtime.load_package ~root:tmp_dir package |> expect_ok in
  let device_name = Metal_runtime.device_name runtime in
  Printf.printf "     Running on device: %s\n%!" device_name;

  (* Prepare inputs *)
  (* x: [1, 64] float16 values of 0.5 (0x3800) *)
  let x_bytes = Bytes.create (64 * 2) in
  for i = 0 to 63 do
    Bytes.set_uint16_le x_bytes (i * 2) 0x3800
  done;

  (* w1: [64, 64] Q4_0: 64 columns, each 2 blocks of 32 elements.
     Each block: scale = 1.0 (0x3c00), 16 bytes of 0xaa (nibble 10 -> decoded weight 2.0).
     Each column: 2 * 18 = 36 bytes. Total = 64 * 36 = 2304 bytes. *)
  let w1_bytes = Bytes.create (64 * 36) in
  for col = 0 to 63 do
    for blk = 0 to 1 do
      let blk_offset = (col * 36) + (blk * 18) in
      Bytes.set_uint16_le w1_bytes blk_offset 0x3c00;
      for b = 0 to 15 do
        Bytes.set_uint8 w1_bytes (blk_offset + 2 + b) 0xaa
      done
    done
  done;

  (* w2: [64, 64] Q4_0: scale = 1.0 (0x3c00), 16 bytes of 0x99 (nibble 9 -> decoded weight 1.0). *)
  let w2_bytes = Bytes.create (64 * 36) in
  for col = 0 to 63 do
    for blk = 0 to 1 do
      let blk_offset = (col * 36) + (blk * 18) in
      Bytes.set_uint16_le w2_bytes blk_offset 0x3c00;
      for b = 0 to 15 do
        Bytes.set_uint8 w2_bytes (blk_offset + 2 + b) 0x99
      done
    done
  done;

  let buf_x = Metal_runtime.Buffer.of_bytes ~runtime x_bytes |> expect_ok in
  let buf_w1 = Metal_runtime.Buffer.of_bytes ~runtime w1_bytes |> expect_ok in
  let buf_w2 = Metal_runtime.Buffer.of_bytes ~runtime w2_bytes |> expect_ok in

  let execution =
    Metal_runtime.with_execution_batch runtime (fun batch ->
        Metal_runtime.encode_schedule batch ~schedule
          ~inputs:[ ("x", buf_x); ("w1", buf_w1); ("w2", buf_w2) ])
    |> expect_ok
  in

  let out_a_buf =
    Metal_runtime.Execution.output execution ~name:"out_a"
    |> Option.to_result ~none:"missing out_a output in execution"
    |> expect_ok
  in
  let out_z_buf =
    Metal_runtime.Execution.output execution ~name:"out_z"
    |> Option.to_result ~none:"missing out_z output in execution"
    |> expect_ok
  in

  let out_a_bytes = Metal_runtime.Buffer.contents out_a_buf |> expect_ok in
  let out_z_bytes = Metal_runtime.Buffer.contents out_z_buf |> expect_ok in

  let () = expect (Bytes.length out_a_bytes = 128) "out_a output is 128 bytes (64 f16 elements)" in
  let () = expect (Bytes.length out_z_bytes = 128) "out_z output is 128 bytes (64 f16 elements)" in

  (* Verify out_a: sum(0.5 * 2.0) = 64.0 -> silu(64.0) = 64.0 (0x5400) *)
  for i = 0 to 63 do
    let v = Bytes.get_uint16_le out_a_bytes (i * 2) in
    if v <> 0x5400 then
      fail (Printf.sprintf "out_a mismatch at column %d: expected 0x5400 (64.0), got 0x%04x" i v)
  done;

  (* Verify out_z: sum(0.5 * 1.0) = 32.0 -> z = a + u = 64.0 + 32.0 = 96.0 (0x5600) *)
  for i = 0 to 63 do
    let v = Bytes.get_uint16_le out_z_bytes (i * 2) in
    if v <> 0x5600 then
      fail (Printf.sprintf "out_z mismatch at column %d: expected 0x5600 (96.0), got 0x%04x" i v)
  done;

  let executed_kernels = Metal_runtime.Execution.kernels execution in
  let () =
    expect
      (List.exists (fun k -> contains_substr k "llmopt_q4_0_linear_f16_x8") executed_kernels)
      "multi-export contraction group kernel was executed by GPU"
  in

  (* Benchmark on-device execution: 10 warmup runs, 100 timed iterations *)
  for _ = 1 to 10 do
    let _ =
      Metal_runtime.with_execution_batch runtime (fun batch ->
          Metal_runtime.encode_schedule batch ~schedule
            ~inputs:[ ("x", buf_x); ("w1", buf_w1); ("w2", buf_w2) ])
    in
    ()
  done;

  let iters = 100 in
  let t0 = Unix.gettimeofday () in
  for _ = 1 to iters do
    let _ =
      Metal_runtime.with_execution_batch runtime (fun batch ->
          Metal_runtime.encode_schedule batch ~schedule
            ~inputs:[ ("x", buf_x); ("w1", buf_w1); ("w2", buf_w2) ])
    in
    ()
  done;
  let elapsed = Unix.gettimeofday () -. t0 in
  let us_per_dispatch = (elapsed /. float_of_int iters) *. 1_000_000.0 in
  Printf.printf "     Benchmark (100 dispatches): %.2f us / dispatch (%.1f dispatches/sec)\n%!"
    us_per_dispatch (float_of_int iters /. elapsed);

  (* Clean up temporary directory *)
  (try
     Sys.remove metal_file;
     Sys.remove air_file;
     Sys.remove metallib_file;
     Unix.rmdir tmp_dir
   with _ -> ());

  Printf.printf "  [PASS] multi-export on-device Metal execution (bit-exact 64/64 parity for out_a and out_z)\n%!";

  Printf.printf "All typed contraction-group algebra tests passed!\n%!"
