(** Typed contraction groups over the shared assignment graph.

    Implementation notes (see [contraction_group.mli] for the contract):

    - Elaboration classifies every region binding once -- [Linear] (bias-free)
      -> contraction, [Unary]/[Binary]/[Cast] expressible in the
      [Applicative_expr] grammar -> scalar SSA step, everything else ->
      opaque -- and then assembles the group from the region results'
      dependency closure.  There is no chain/topology matching.
    - Scalar steps keep SSA slot references (no tree inlining at group level),
      so shared producers are explicit; fan-out shows up as use obligations.
    - Add/Mul steps with exactly one internal operand are canonicalised to
      internal-first orientation, mirroring the legacy [Epilogue] consumer
      templates.  This commutes IEEE-commutative operations only; it never
      reassociates floating-point terms.
    - [single_acc_epilogue] reconstructs tags as canonical normal forms over
      the step list (identity, bare Silu/Gelu, flat external Add) and returns
      the pure monoid [Expr] for every longer chain. *)

module Slot_map = Map.Make (Kernel_region.Slot)

type expr =
  | Slot of Kernel_region.Slot.t
  | Const of float
  | Read of Ir.Value.t
  | Unop of Applicative_expr.unop * expr
  | Binop of Applicative_expr.binop * expr * expr

type contraction = {
  slot : Kernel_region.Slot.t;
  m : int;
  n : int;
  k : int;
  storage : Ir.Linear_storage.layout;
  input : Ir.Value.t;
  weight : Ir.Value.t;
  parameters : Ir.Value.t list;
  value : Ir.Value.t;
  shape : Tensor_shape.t;
}

type step = {
  slot : Kernel_region.Slot.t;
  expr : expr;
  value : Ir.Value.t;
  shape : Tensor_shape.t;
  dtype : Ir.Dtype.t;
}

type export = {
  slot : Kernel_region.Slot.t;
  value : Ir.Value.t;
}

type t = {
  name : string;
  domain : Tensor_shape.t;
  contractions : contraction list;
  steps : step list;
  exports : export list;
}

(* ------------------------------------------------------------------------ *)
(* Leaf classification                                                       *)
(* ------------------------------------------------------------------------ *)

type classified =
  | Contraction of contraction
  | Scalar of step
  | Opaque

let unary_of = function
  | Kernel_region.Primitive.Silu -> Some Applicative_expr.Silu
  | Kernel_region.Primitive.Gelu -> Some Applicative_expr.Gelu
  | Kernel_region.Primitive.Neg -> Some Applicative_expr.Neg
  | Kernel_region.Primitive.Rsqrt -> Some Applicative_expr.Rsqrt
  | Kernel_region.Primitive.Exp -> Some Applicative_expr.Exp
  | (Kernel_region.Primitive.Cos | Kernel_region.Primitive.Sin
    | Kernel_region.Primitive.Tanh | Kernel_region.Primitive.Sigmoid
    | Kernel_region.Primitive.Softplus | Kernel_region.Primitive.Pow _) ->
      None

let binary_of = function
  | Kernel_region.Primitive.Mul -> Some Applicative_expr.Mul
  | Kernel_region.Primitive.Add -> Some Applicative_expr.Add
  | Kernel_region.Primitive.Sub -> Some Applicative_expr.Sub
  | Kernel_region.Primitive.Div -> Some Applicative_expr.Div
  | (Kernel_region.Primitive.Logical_and | Kernel_region.Primitive.Equal
    | Kernel_region.Primitive.Not_equal | Kernel_region.Primitive.Less_equal
    | Kernel_region.Primitive.Greater) ->
      None

let operand_expr input_values producers slot =
  match Slot_map.find_opt slot input_values with
  | Some value -> Some (Read value)
  | None -> if Slot_map.mem slot producers then Some (Slot slot) else None

let single_output binding =
  match Kernel_region.binding_outputs binding with
  | [ output ] -> Some output
  | _ -> None

let externals input_values slots =
  let rec go acc = function
    | [] -> Some (List.rev acc)
    | slot :: rest -> (
        match Slot_map.find_opt slot input_values with
        | Some value -> go (value :: acc) rest
        | None -> None)
  in
  go [] slots

(* Add/Mul over one internal operand canonicalise to internal-first order;
   this is commutation only (both are IEEE-commutative), never reassociation. *)
let canonical_binary op left right =
  match (op, left, right) with
  | ( (Applicative_expr.Add | Applicative_expr.Mul),
      (Read _ as external_read),
      (Slot _ as internal_ref) ) ->
      Binop (op, internal_ref, external_read)
  | _ -> Binop (op, left, right)

let scalar_unary input_values producers output operands wrap =
  match (output, operands) with
  | Some output, [ operand ] -> (
      match
        ( operand_expr input_values producers operand,
          Kernel_region.output_value output )
      with
      | Some operand_expr, Some value ->
          Scalar
            {
              slot = Kernel_region.output_slot output;
              expr = wrap operand_expr;
              value;
              shape = Kernel_region.output_shape output;
              dtype = Kernel_region.output_dtype output;
            }
      | _ -> Opaque)
  | _ -> Opaque

let scalar_binary input_values producers output operands op =
  match (output, operands) with
  | Some output, [ left; right ] -> (
      match
        ( operand_expr input_values producers left,
          operand_expr input_values producers right )
      with
      | Some left_expr, Some right_expr -> (
          match Kernel_region.output_value output with
          | Some value ->
              Scalar
                {
                  slot = Kernel_region.output_slot output;
                  expr = canonical_binary op left_expr right_expr;
                  value;
                  shape = Kernel_region.output_shape output;
                  dtype = Kernel_region.output_dtype output;
                }
          | None -> Opaque)
      | _ -> Opaque)
  | _ -> Opaque

let classify input_values producers binding =
  match (Kernel_region.binding_primitive binding, single_output binding) with
  | Kernel_region.Primitive.Linear { m; n; k; bias = false; storage }, output -> (
      match externals input_values (Kernel_region.binding_inputs binding) with
      | Some (input :: weight :: parameters) -> (
          match output with
          | Some output -> (
              match Kernel_region.output_value output with
              | Some value ->
                  Contraction
                    {
                      slot = Kernel_region.output_slot output;
                      m;
                      n;
                      k;
                      storage;
                      input;
                      weight;
                      parameters;
                      value;
                      shape = Kernel_region.output_shape output;
                    }
              | None -> Opaque)
          | None -> Opaque)
      | _ -> Opaque)
  | Kernel_region.Primitive.Linear _, _ -> Opaque
  | Kernel_region.Primitive.Unary op, output -> (
      match unary_of op with
      | Some unop ->
          scalar_unary input_values producers output
            (Kernel_region.binding_inputs binding)
            (fun operand -> Unop (unop, operand))
      | None -> Opaque)
  | Kernel_region.Primitive.Cast dtype, output ->
      scalar_unary input_values producers output
        (Kernel_region.binding_inputs binding)
        (fun operand -> Unop (Applicative_expr.Cast dtype, operand))
  | Kernel_region.Primitive.Binary op, output -> (
      match binary_of op with
      | Some binop ->
          scalar_binary input_values producers output
            (Kernel_region.binding_inputs binding)
            binop
      | None -> Opaque)
  | _ -> Opaque

(* ------------------------------------------------------------------------ *)
(* Dependency closure and assembly                                           *)
(* ------------------------------------------------------------------------ *)

(* Returns the set of slots that feed the given region result slots, walking
   binding producers backwards.  Region inputs have no producer and terminate
   the walk. *)
let dependency_closure bindings result_slots =
  let by_slot =
    List.fold_left
      (fun map binding ->
        List.fold_left
          (fun map output ->
            Slot_map.add (Kernel_region.output_slot output) binding map)
          map (Kernel_region.binding_outputs binding))
      Slot_map.empty bindings
  in
  let visited = ref Slot_map.empty in
  let pending = ref result_slots in
  while !pending <> [] do
    match !pending with
    | [] -> ()
    | slot :: rest ->
        pending := rest;
        if not (Slot_map.mem slot !visited) then
          match Slot_map.find_opt slot by_slot with
          | None -> ()
          | Some binding ->
              visited := Slot_map.add slot () !visited;
              (* Every output of a binding denotes the same SSA value. *)
              List.iter
                (fun output ->
                  visited :=
                    Slot_map.add (Kernel_region.output_slot output) () !visited)
                (Kernel_region.binding_outputs binding);
              pending := Kernel_region.binding_inputs binding @ !pending
  done;
  !visited

let rec reads_of expr acc =
  match expr with
  | Read value -> value :: acc
  | Slot _ | Const _ -> acc
  | Unop (_, inner) -> reads_of inner acc
  | Binop (_, left, right) -> reads_of right (reads_of left acc)

(* Every member step must live in the group domain and so must every external
   tensor it reads back per coordinate. *)
let step_in_domain domain (step : step) =
  Tensor_shape.equal step.shape domain
  && List.for_all
       (fun value ->
         Tensor_shape.equal (Ir.Value.logical_shape value) domain)
       (reads_of step.expr [])

let binding_in_slots slots binding =
  List.exists
    (fun output -> Slot_map.mem (Kernel_region.output_slot output) slots)
    (Kernel_region.binding_outputs binding)

let of_region (region : Kernel_region.region) =
  let inputs = Kernel_region.inputs region in
  let bindings = Kernel_region.bindings region in
  let input_values =
    List.fold_left
      (fun map ({ Kernel_region.slot; value } : Kernel_region.input) ->
        Slot_map.add slot value map)
      Slot_map.empty inputs
  in
  let producers =
    List.fold_left
      (fun map binding ->
        List.fold_left
          (fun map output ->
            Slot_map.add (Kernel_region.output_slot output) binding map)
          map (Kernel_region.binding_outputs binding))
      Slot_map.empty bindings
  in
  let result_slots =
    List.map
      (fun ({ Kernel_region.slot; _ } : Kernel_region.result) -> slot)
      (Kernel_region.results region)
  in
  let closure = dependency_closure bindings result_slots in
  (* Every binding must feed a region result (one component per region). *)
  if not (List.for_all (binding_in_slots closure) bindings) then None
  else
    let zipped = List.combine bindings (List.map (classify input_values producers) bindings) in
    let member_classifieds =
      List.filter_map
        (fun (binding, classified) ->
          if binding_in_slots closure binding then Some classified else None)
        zipped
    in
    if List.exists (function Opaque -> true | _ -> false) member_classifieds
    then None
    else
      let contractions, steps =
        List.fold_left
          (fun (contractions, steps) classified ->
            match classified with
            | Contraction contraction -> (contraction :: contractions, steps)
            | Scalar step -> (contractions, step :: steps)
            | Opaque -> (contractions, steps))
          ([], []) member_classifieds
      in
      let contractions = List.rev contractions in
      let steps = List.rev steps in
      match contractions with
      | [] -> None
      | first :: _ ->
          let domain = first.shape in
          if
            not
              (List.for_all
                 (fun (contraction : contraction) ->
                   Tensor_shape.equal contraction.shape domain)
                 contractions)
          then None
          else if not (List.for_all (step_in_domain domain) steps) then None
          else
            (* Region results become group exports; every result must be a
               member binding output. *)
            let exports =
              List.filter_map
                (fun ({ Kernel_region.slot; value } : Kernel_region.result) ->
                  if Slot_map.mem slot closure then Some { slot; value }
                  else None)
                (Kernel_region.results region)
            in
            if List.length exports <> List.length (Kernel_region.results region)
            then None
            else
              Some
                {
                  name = Kernel_region.name region;
                  domain;
                  contractions;
                  steps;
                  exports;
                }

(* ------------------------------------------------------------------------ *)
(* Queries                                                                   *)
(* ------------------------------------------------------------------------ *)

let rec referenced_slots expr =
  match expr with
  | Slot slot -> [ slot ]
  | Const _ | Read _ -> []
  | Unop (_, inner) -> referenced_slots inner
  | Binop (_, left, right) -> referenced_slots left @ referenced_slots right

let use_count group slot =
  List.fold_left
    (fun count step ->
      count
      + List.length
          (List.filter (Kernel_region.Slot.equal slot) (referenced_slots step.expr)))
    0 group.steps

let is_exported group slot =
  List.exists (fun export -> Kernel_region.Slot.equal export.slot slot) group.exports

let slot_expr group slot =
  let rec go = function
    | [] -> None
    | (step : step) :: rest ->
        if Kernel_region.Slot.equal step.slot slot then Some step.expr else go rest
  in
  go group.steps

(* ------------------------------------------------------------------------ *)
(* Single-accumulator reconstruction                                         *)
(* ------------------------------------------------------------------------ *)

(* Inline every member step into one monoid expression whose single variable
   is the contraction slot ([Epilogue.Acc]). *)
let rec inline env expr =
  match expr with
  | Slot slot -> (
      match Slot_map.find_opt slot env with
      | Some value -> value
      | None -> raise Not_found)
  | Const c -> Applicative_expr.Const c
  | Read value ->
      Applicative_expr.Read { value; coord = Coordinate_map.identity }
  | Unop (op, inner) -> Applicative_expr.Unop (op, inline env inner)
  | Binop (op, left, right) ->
      Applicative_expr.Binop (op, inline env left, inline env right)

(* Tags are canonical normal forms recognised on the step list: a group whose
   single step is a bare Silu/Gelu over the accumulator, or a flat external
   Add, carries the legacy tag; every other chain is the pure monoid [Expr]. *)
let reconstruct group expr =
  if group.steps = [] then Epilogue.none
  else
    match (group.contractions, group.steps) with
    | [ contraction ], [ step ] -> (
        match step.expr with
        | Unop (Applicative_expr.Silu, Slot slot)
          when Kernel_region.Slot.equal slot contraction.slot ->
            Epilogue.silu
        | Unop (Applicative_expr.Gelu, Slot slot)
          when Kernel_region.Slot.equal slot contraction.slot ->
            Epilogue.gelu
        | Binop (Applicative_expr.Add, Slot slot, Read residual)
          when Kernel_region.Slot.equal slot contraction.slot ->
            Epilogue.residual_add ~residual
        | Binop (Applicative_expr.Add, Read residual, Slot slot)
          when Kernel_region.Slot.equal slot contraction.slot ->
            Epilogue.residual_add ~residual
        | _ -> Epilogue.of_expr expr)
    | _ -> Epilogue.of_expr expr

let single_acc_epilogue group =
  match (group.contractions, group.exports) with
  | [ contraction ], [ export ] ->
      let env = ref Slot_map.empty in
      let ok =
        try
          env :=
            Slot_map.add contraction.slot
              (Applicative_expr.Slot Epilogue.Acc)
              !env;
          List.iter
            (fun step ->
              let value = inline !env step.expr in
              env := Slot_map.add step.slot value !env)
            group.steps;
          true
        with Not_found -> false
      in
      if not ok then None
      else
        (match Slot_map.find_opt export.slot !env with
        | None -> None
        | Some expr -> Some (reconstruct group expr))
  | _ -> None

(* ------------------------------------------------------------------------ *)
(* G4P12-02 widening: select/fold routed through the elaboration            *)
(* ------------------------------------------------------------------------ *)

(* The legacy [Kernel_ir.Fused_tactic] recognizers stay the pinned oracle;
   [Kernel_ir] cannot consume this module (OCaml self-type compile cycle), so
   the elaboration lives here and consumers cascade to it only when the
   legacy recognizer admits nothing.  The widening claims a whole region as
   one [Linear_epilogue] exactly when its elaborated group is a single-
   accumulator chain following the legacy foldable op set -- Silu/Gelu
   unaries and Add/Mul binaries whose internal leaves all sit on the single
   accumulator link, including same-slot self branches ([y*y] consumes the
   link twice) -- with the single export writing the final step slot.  Bare
   identity regions stay unclaimed (legacy select depth requirement), and
   Cast/Sigmoid cuts keep failing exactly as they do under the legacy
   recognizer. *)

let grammar_step expr =
  match expr with
  | Unop ((Applicative_expr.Silu | Applicative_expr.Gelu), _) -> true
  | Binop ((Applicative_expr.Add | Applicative_expr.Mul), _, _) -> true
  | _ -> false

let single_acc_chain_epilogue (group : t) =
  match (group.contractions, group.exports) with
  | [ contraction ], [ export ] ->
      let rec chain_ok (expected : Kernel_region.Slot.t) (steps : step list) =
        match steps with
        | [] -> true
        | (step : step) :: rest ->
            let leaves = referenced_slots step.expr in
            grammar_step step.expr
            && (match leaves with
               | [ _ ] -> true
               | [ left; right ] -> Kernel_region.Slot.equal left right
               | _ -> false)
            && List.for_all (Kernel_region.Slot.equal expected) leaves
            && chain_ok step.slot rest
      in
      let covers_export (steps : step list) =
        match List.rev steps with
        | step :: _ -> Kernel_region.Slot.equal export.slot step.slot
        | [] -> Kernel_region.Slot.equal export.slot contraction.slot
      in
      if covers_export group.steps && chain_ok contraction.slot group.steps
      then single_acc_epilogue group
      else None
  | _ -> None

let linear_of_contraction contraction =
  {
    Kernel_ir.Fused_tactic.m = contraction.m;
    n = contraction.n;
    k = contraction.k;
    storage = contraction.storage;
    input = contraction.input;
    weight = contraction.weight;
    parameters = contraction.parameters;
  }

let select_linear_epilogue region =
  match of_region region with
  | None -> None
  | Some group -> (
      match group.steps, single_acc_chain_epilogue group with
      | _ :: _, Some epilogue -> (
          match (group.contractions, group.exports) with
          | [ contraction ], [ export ] ->
              Some
                (Kernel_ir.Fused_tactic.Linear_epilogue
                   {
                     linear = linear_of_contraction contraction;
                     epilogue;
                     output = export.value;
                   })
          | _ -> None)
      | _ -> None)

let fold_linear_epilogue region =
  match of_region region with
  | Some group -> single_acc_chain_epilogue group
  | None -> None

let select region =
  match Kernel_ir.Fused_tactic.select region with
  | Some tactic -> Some tactic
  | None -> select_linear_epilogue region
