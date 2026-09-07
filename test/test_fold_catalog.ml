open Test_support
(* 13. Fused-tactic Linear_epilogue fold catalog.

   [Kernel_ir.Fused_tactic.fold_linear_epilogue] is the algebraic core that
   folds the single-linear pointwise consumer chain of a region into an
   [Epilogue.t] through associative [Epilogue.compose].  Selection remains
   admission-gated to the flat legacy residual-add form so existing decode /
   prefill dispatch is byte-identical; richer chains surface in the same
   change that teaches code emission and runtime dispatch their writebacks.

   Regions here are built directly through the public [Kernel_ir] API with
   the same slot/effect discipline as [Constructive_lowering]'s executable
   regions: region inputs first, one slot per step output, read effects on
   inputs and write effects on results. *)
  let shape_1_64 = Tensor_shape.of_ints_exn [ 1; 64 ]
  let shape_64_64 = Tensor_shape.of_ints_exn [ 64; 64 ]

  let value id dtype =
    Ir.Value.make_tensor ~id ~shape:shape_1_64 ~dtype

  let linear_primitive =
    Kernel_ir.Primitive.linear ~m:1 ~n:64 ~k:64 ~bias:false
      ~storage:(Ir.Linear_storage.Dense Ir.Dtype.Float16)

  (* steps: (primitive, operand slot ints, output value).  Region inputs take
     slots 0..n-1 in order; step i binds slot (n + i). *)
  let expect_ok_kernel = function
    | Ok value -> value
    | Error error -> fail (Kernel_ir.error_to_string error)

  let make_region ~name ~inputs ~steps ~results =
    let inputs =
      List.mapi
        (fun index value ->
          ({ Kernel_ir.slot = Kernel_ir.Slot.of_int_exn index; value }
            : Kernel_ir.input))
        inputs
    in
    let base = List.length inputs in
    let bindings =
      fst
        (List.fold_left
           (fun (bindings, count) (primitive, operand_slots, output_value) ->
             let slot = Kernel_ir.Slot.of_int_exn (base + count) in
             let binding =
               Kernel_ir.make_binding
                 ~outputs:
                   [
                     {
                       Kernel_ir.slot;
                       shape = Ir.Value.logical_shape output_value;
                       dtype = Ir.Value.dtype output_value;
                       value = Some output_value;
                     };
                   ]
                 ~primitive
                 ~inputs:(List.map Kernel_ir.Slot.of_int_exn operand_slots)
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
            Kernel_ir.slot = Kernel_ir.Slot.of_int_exn slot_index;
            value = result_value;
            storage = Kernel_ir.Fresh;
          })
        results
    in
    let input_accesses =
      List.map
        (fun ({ Kernel_ir.slot; _ } : Kernel_ir.input) ->
          {
            Kernel_ir.Effect.slot;
            mode = Kernel_ir.Effect.Read;
            alias = Kernel_ir.Effect.Distinct;
          })
        inputs
    in
    let result_accesses =
      List.map
        (fun ({ Kernel_ir.slot; _ } : Kernel_ir.result) ->
          {
            Kernel_ir.Effect.slot;
            mode = Kernel_ir.Effect.Write;
            alias = Kernel_ir.Effect.Distinct;
          })
        results
    in
    let effects =
      Kernel_ir.Effect.create ~accesses:(input_accesses @ result_accesses) ()
      |> expect_ok
    in
    Kernel_ir.create ~name
      ~member_node_ids:[ 1; 2; 3; 4; 5; 6; 7; 8 ]
      ~inputs ~bindings ~results ~effects ~resource:Kernel_ir.Resource.zero
    |> expect_ok_kernel

  let single_result_region ~name ~inputs ~steps ~result_index result_value =
    make_region ~name ~inputs ~steps ~results:[ (result_index, result_value) ]

  let fold_matches ~name ~inputs ~steps ~result_index result_value expected =
    let region =
      single_result_region ~name ~inputs ~steps ~result_index result_value
    in
    expect
      (match Kernel_ir.Fused_tactic.fold_linear_epilogue region with
      | Some epilogue -> Epilogue.equal epilogue expected
      | None -> false)
      name

  let fold_is_none ~name ~inputs ~steps ~result_index result_value =
    let region =
      single_result_region ~name ~inputs ~steps ~result_index result_value
    in
    expect
      (Kernel_ir.Fused_tactic.fold_linear_epilogue region = None)
      name

  let select_is_none ~name ~inputs ~steps ~result_index result_value =
    let region =
      single_result_region ~name ~inputs ~steps ~result_index result_value
    in
    expect (Kernel_ir.Fused_tactic.select region = None) name

  let select_matches ~name ~inputs ~steps ~result_index result_value epilogue =
    let region =
      single_result_region ~name ~inputs ~steps ~result_index result_value
    in
    expect
      (match Kernel_ir.Fused_tactic.select region with
      | Some
          (Kernel_ir.Fused_tactic.Linear_epilogue { linear; epilogue = ep;
                                                    output }) ->
          linear.Kernel_ir.Fused_tactic.m = 1
          && linear.n = 64
          && linear.k = 64
          && Epilogue.equal ep epilogue
          && Ir.Value.equal output result_value
      | Some _ -> false
      | None -> false)
      name

  let run () =
    Printf.printf "Running fused Linear_epilogue fold catalog tests...\n%!";
    (* Unique metadata per fixture.  Ids never meet across regions, so the
       small id range below is only scoped per value. *)
    let x =
      Ir.Value.make_tensor ~id:1 ~shape:shape_1_64 ~dtype:Ir.Dtype.Float16
    in
    let w =
      Ir.Value.make_tensor ~id:2 ~shape:shape_64_64 ~dtype:Ir.Dtype.Float16
    in
    let residual = value 3 Ir.Dtype.Float16 in
    let up = value 4 Ir.Dtype.Float16 in
    let o_linear = value 5 Ir.Dtype.Float16 in
    let o_linear2 = value 6 Ir.Dtype.Float16 in
    let o_silu = value 7 Ir.Dtype.Float16 in
    let o_mul = value 8 Ir.Dtype.Float16 in
    let o_add = value 9 Ir.Dtype.Float16 in
    let o_sigmoid = value 10 Ir.Dtype.Float16 in
    let o_cast = value 11 Ir.Dtype.Float16 in
    let linear_step = (linear_primitive, [ 0; 1 ], o_linear) in
    let add_step left right output =
      (Kernel_ir.Primitive.binary Kernel_ir.Primitive.Add, [ left; right ], output)
    in
    let mul_step left right output =
      (Kernel_ir.Primitive.binary Kernel_ir.Primitive.Mul, [ left; right ], output)
    in
    (* Region input layout for the chain fixtures: x=0, w=1, residual=2,
       up=3, so the linear binds slot 4 and step outputs follow at 5, 6, 7. *)
    let inputs_4 = [ x; w; residual; up ] in

    (* A. Flat legacy class: Linear + Add(linear, external residual).  The
       fold returns the residual-add epilogue and selection still admits the
       region, preserving the exact legacy decode/prefill dispatch class. *)
    let flat_region =
      single_result_region ~name:"flat-linear-add" ~inputs:inputs_4
        ~steps:[ linear_step; add_step 4 2 o_add ]
        ~result_index:5 o_add
    in
    (match Kernel_ir.Fused_tactic.fold_linear_epilogue flat_region with
    | Some epilogue ->
        expect
          (Epilogue.equal epilogue (Epilogue.residual_add ~residual))
          "fold flattens Linear+Add to the residual-add epilogue"
    | None -> fail "fold rejected the flat Linear+Add region");
    (match Kernel_ir.Fused_tactic.select flat_region with
    | Some
        (Kernel_ir.Fused_tactic.Linear_epilogue
          { linear; epilogue; output }) ->
        expect
          (linear.Kernel_ir.Fused_tactic.m = 1
          && linear.n = 64
          && linear.k = 64
          && linear.storage = Ir.Linear_storage.Dense Ir.Dtype.Float16
          && Ir.Value.equal linear.input x
          && Ir.Value.equal linear.weight w
          && linear.parameters = []
          && Epilogue.equal epilogue (Epilogue.residual_add ~residual)
          && Ir.Value.equal output o_add)
          "select admits the flat Linear+Add region with the legacy epilogue shape"
    | _ -> fail "select rejected the flat Linear+Add region");

    (* A'. Operand orientation must not matter: Add(external, linear). *)
    fold_matches ~name:"fold tolerates Add(residual, linear) orientation"
      ~inputs:inputs_4
      ~steps:[ linear_step; add_step 2 4 o_add ]
      ~result_index:5 o_add (Epilogue.residual_add ~residual);

    (* B. Linear -> Silu -> Add(external residual): the chain folds into a
       composite epilogue and is now selectable; selection hands the region to
       the digest-keyed emitter whose writeback honors the full chain, while
       the runtime keeps flat residual-add regions on the legacy _add path. *)
    let silu_residual =
      Epilogue.compose ~producer:Epilogue.silu
        ~consumer:(Epilogue.residual_add ~residual)
    in
    fold_matches ~name:"fold composes Silu over the flat residual add"
      ~inputs:inputs_4
      ~steps:
        [
          linear_step;
          (Kernel_ir.Primitive.unary Kernel_ir.Primitive.Silu, [ 4 ], o_silu);
          add_step 5 2 o_add;
        ]
      ~result_index:6 o_add silu_residual;
    select_matches ~name:"select admits the Silu-residual chain as one folded epilogue"
      ~inputs:inputs_4
      ~steps:
        [
          linear_step;
          (Kernel_ir.Primitive.unary Kernel_ir.Primitive.Silu, [ 4 ], o_silu);
          add_step 5 2 o_add;
        ]
      ~result_index:6 o_add silu_residual;

    (* C. Linear -> Mul(linear, external up): the pointwise scale by an
       external tensor (an MLP gate's up projection) folds in place and is
       selectable as a one-step scale epilogue. *)
    fold_matches ~name:"fold folds the external-tensor scale (mul by up)"
      ~inputs:inputs_4
      ~steps:[ linear_step; mul_step 4 3 o_mul ]
      ~result_index:5 o_mul (Epilogue.mul ~factor:up);
    select_matches ~name:"select admits the external-tensor scale epilogue"
      ~inputs:inputs_4
      ~steps:[ linear_step; mul_step 4 3 o_mul ]
      ~result_index:5 o_mul (Epilogue.mul ~factor:up);

    (* D. Linear -> Silu -> Mul(up) -> Add(residual): the full gated-MLP tail
       collapses to one associative epilogue expression that selection now
       admits in full. *)
    let silu_mul_residual =
      Epilogue.compose ~producer:Epilogue.silu
        ~consumer:(Epilogue.mul ~factor:up)
      |> fun producer ->
      Epilogue.compose ~producer ~consumer:(Epilogue.residual_add ~residual)
    in
    fold_matches ~name:"fold collapses the Silu-mul-residual chain"
      ~inputs:inputs_4
      ~steps:
        [
          linear_step;
          (Kernel_ir.Primitive.unary Kernel_ir.Primitive.Silu, [ 4 ], o_silu);
          mul_step 5 3 o_mul;
          add_step 6 2 o_add;
        ]
      ~result_index:7 o_add silu_mul_residual;
    select_matches ~name:"select admits the full Silu-mul-residual chain"
      ~inputs:inputs_4
      ~steps:
        [
          linear_step;
          (Kernel_ir.Primitive.unary Kernel_ir.Primitive.Silu, [ 4 ], o_silu);
          mul_step 5 3 o_mul;
          add_step 6 2 o_add;
        ]
      ~result_index:7 o_add silu_mul_residual;

    (* E. Regions without a unique linear contraction do not fold or select. *)
    fold_is_none ~name:"fold rejects a two-linear region"
      ~inputs:inputs_4
      ~steps:
        [
          linear_step;
          (linear_primitive, [ 0; 1 ], o_linear2);
        ]
      ~result_index:5 o_linear2;
    select_is_none ~name:"select rejects a two-linear region"
      ~inputs:inputs_4
      ~steps:
        [
          linear_step;
          (linear_primitive, [ 0; 1 ], o_linear2);
        ]
      ~result_index:5 o_linear2;

    (* F. Out-of-grammar unaries (Sigmoid has no Applicative-expr counterpart
       today) cut the chain rather than folding silently. *)
    fold_is_none ~name:"fold cuts on out-of-grammar unary (Sigmoid)"
      ~inputs:inputs_4
      ~steps:
        [
          linear_step;
          (Kernel_ir.Primitive.unary Kernel_ir.Primitive.Sigmoid, [ 4 ], o_sigmoid);
        ]
      ~result_index:5 o_sigmoid;
    select_is_none ~name:"select keeps the Sigmoid chain out of dispatch"
      ~inputs:inputs_4
      ~steps:
        [
          linear_step;
          (Kernel_ir.Primitive.unary Kernel_ir.Primitive.Sigmoid, [ 4 ], o_sigmoid);
        ]
      ~result_index:5 o_sigmoid;

    (* G. Bare single-linear region: the fold reports the identity epilogue
       (a pure contraction) but selection refuses it -- today that class is
       not a fused tactic and must not be claimed by Linear_epilogue. *)
    let bare_region =
      single_result_region ~name:"bare-linear" ~inputs:[ x; w ]
        ~steps:[ linear_step ] ~result_index:2 o_linear
    in
    (match Kernel_ir.Fused_tactic.fold_linear_epilogue bare_region with
    | Some epilogue ->
        expect (Epilogue.equal epilogue Epilogue.none)
          "fold reports the identity epilogue for a bare linear region"
    | None -> fail "fold rejected a bare single-linear region");
    expect (Kernel_ir.Fused_tactic.select bare_region = None)
      "select refuses the bare linear region";

    (* H. Double residual add does not reduce to the flat tag but the algebra
       still folds it associatively, and selection now admits the composite as
       one digest-keyed epilogue (runtime dispatch keeps the flat single-add
       region byte-identical on the legacy _add path). *)
    let double_add =
      Epilogue.compose ~producer:(Epilogue.residual_add ~residual)
        ~consumer:(Epilogue.residual_add ~residual:up)
    in
    fold_matches ~name:"fold keeps stacked residual adds associative"
      ~inputs:inputs_4
      ~steps:[ linear_step; add_step 4 2 o_add; add_step 5 3 o_mul ]
      ~result_index:6 o_mul double_add;
    select_matches ~name:"select admits stacked residual adds as one folded epilogue"
      ~inputs:inputs_4
      ~steps:[ linear_step; add_step 4 2 o_add; add_step 5 3 o_mul ]
      ~result_index:6 o_mul double_add;

    (* I. A cast on the accumulator is not yet part of the foldable grammar:
       conservative cut, no silent mis-fuse. *)
    fold_is_none ~name:"fold cuts on an intermediate cast"
      ~inputs:inputs_4
      ~steps:
        [
          linear_step;
          (Kernel_ir.Primitive.cast Ir.Dtype.Float16, [ 4 ], o_cast);
          add_step 5 2 o_add;
        ]
      ~result_index:6 o_add;
    select_is_none ~name:"select keeps the cast chain out of dispatch"
      ~inputs:inputs_4
      ~steps:
        [
          linear_step;
          (Kernel_ir.Primitive.cast Ir.Dtype.Float16, [ 4 ], o_cast);
          add_step 5 2 o_add;
        ]
      ~result_index:6 o_add;

    (* J. Fan-out of the chain to a second result is not one epilogue and must
       not be claimed by either the fold or selection. *)
    let fanout_region =
      make_region ~name:"fanout" ~inputs:inputs_4
        ~steps:
          [
            linear_step;
            (Kernel_ir.Primitive.unary Kernel_ir.Primitive.Silu, [ 4 ], o_silu);
            add_step 5 2 o_add;
            mul_step 5 3 o_mul;
          ]
        ~results:[ (6, o_add); (7, o_mul) ]
    in
    expect
      (Kernel_ir.Fused_tactic.fold_linear_epilogue fanout_region = None
      && Kernel_ir.Fused_tactic.select fanout_region = None)
      "fold and select reject fan-out chains with two results";

    (* K. G4P12-02 widening: a single-slot branch shape ([y*y] -- the
       pointwise Mul consumes the single accumulator slot twice) stays
       outside the legacy recognizers, but the elaboration admits it as one
       [Linear_epilogue] whose epilogue is the pure monoid
       [Expr (Mul (Acc, Acc))].  This is the shape the digest-keyed emission
       path serves for block-quantized contractions. *)
    let y2_region =
      single_result_region ~name:"y-square" ~inputs:[ x; w ]
        ~steps:[ linear_step; mul_step 2 2 o_mul ] ~result_index:3 o_mul
    in
    expect
      (Kernel_ir.Fused_tactic.fold_linear_epilogue y2_region = None
      && Kernel_ir.Fused_tactic.select y2_region = None)
      "y*y stays outside the legacy recognizers today";
    (match Contraction_group.select y2_region with
    | Some
        (Kernel_ir.Fused_tactic.Linear_epilogue
          { linear; epilogue; output }) ->
        expect
          (linear.Kernel_ir.Fused_tactic.m = 1
          && linear.Kernel_ir.Fused_tactic.n = 64
          && linear.Kernel_ir.Fused_tactic.k = 64
          && Ir.Value.equal linear.Kernel_ir.Fused_tactic.input x
          && Ir.Value.equal linear.Kernel_ir.Fused_tactic.weight w
          && linear.Kernel_ir.Fused_tactic.parameters = []
          && Ir.Value.equal output o_mul
          && Epilogue.equal epilogue
               (Epilogue.of_expr
                  (Applicative_expr.Binop
                     ( Applicative_expr.Mul,
                       Applicative_expr.Slot Epilogue.Acc,
                       Applicative_expr.Slot Epilogue.Acc ))))
          "elaborated select admits y*y as Expr (Mul (Acc, Acc))"
    | Some _ -> fail "y*y must select as Linear_epilogue only"
    | None -> fail "elaborated select rejected the y*y square");
    (match Contraction_group.fold_linear_epilogue y2_region with
    | Some epilogue ->
        expect
          (Epilogue.equal epilogue
             (Epilogue.of_expr
                (Applicative_expr.Binop
                   ( Applicative_expr.Mul,
                     Applicative_expr.Slot Epilogue.Acc,
                     Applicative_expr.Slot Epilogue.Acc ))))
          "elaborated fold yields the same monoid epilogue for y*y"
    | None -> fail "elaborated fold rejected the y*y square");

    print_endline "llmopt fused Linear_epilogue fold catalog passed"
