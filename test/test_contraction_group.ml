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
  let make_region ~name ~inputs ~steps ~results =
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
      ~member_node_ids:[ 1; 2; 3; 4; 5; 6; 7; 8 ]
      ~inputs ~bindings ~results ~effects ~resource:Kernel_region.Resource.zero
    |> expect_ok_kernel

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
        "two-slot group has no single-accumulator epilogue"));
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
        "two-export group has no single-accumulator epilogue");
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

  Printf.printf "All typed contraction-group algebra tests passed!\n%!"
