(** Backend tactic selection from region primitive dataflow: fused linear
    epilogues, gated products, RMS normalization/RoPE, short-conv-SILU,
    and gated delta.  Depends on [Kernel_region] queries only. *)

open Kernel_region

  type linear = {
    m : int;
    n : int;
    k : int;
    storage : Ir.Linear_storage.layout;
    input : Ir.Value.t;
    weight : Ir.Value.t;
    parameters : Ir.Value.t list;
  }

  type t =
    | Linear_epilogue of {
        linear : linear;
        epilogue : Epilogue.t;
        output : Ir.Value.t;
      }
    | Gated_product of {
        gate : linear;
        up : linear;
        activation : Primitive.unary;
        output : Ir.Value.t;
      }
    | Rms_normalize of {
        input : Ir.Value.t;
        weight : Ir.Value.t option;
        residual : Ir.Value.t option;
        layer_scalar : Ir.Value.t option;
        output : Ir.Value.t;
        epsilon : float;
      }
    | Rms_rope of {
        config : Ir.Rms_rope.t;
        input : Ir.Value.t;
        weight : Ir.Value.t;
        cosine : Ir.Value.t;
        sine : Ir.Value.t;
        output : Ir.Value.t;
      }
    | Short_conv_silu of {
        config : Ir.Short_conv_silu.t;
        input : Ir.Value.t;
        weight : Ir.Value.t;
        output : Ir.Value.t;
      }
    | Gated_delta of {
        query : Ir.Value.t;
        key : Ir.Value.t;
        value : Ir.Value.t;
        gate : Ir.Value.t;
        beta : Ir.Value.t;
        output : Ir.Value.t;
      }

  let input_map region =
    List.fold_left
      (fun values ({ slot; value } : input) -> Slot_map.add slot value values)
      Slot_map.empty region.inputs

  let value_map region =
    List.fold_left
      (fun values binding ->
        List.fold_left
          (fun values ({ slot; value; _ } : output_spec) ->
            match value with
            | Some value -> Slot_map.add slot value values
            | None -> values)
          values binding.outputs)
      (input_map region) region.bindings

  let input_value values slot = Slot_map.find_opt slot values

  let external_values values slots =
    let rec collect reversed = function
      | [] -> Some (List.rev reversed)
      | slot :: rest ->
          (match input_value values slot with
          | None -> None
          | Some value -> collect (value :: reversed) rest)
    in
    collect [] slots

  let output_slot binding =
    match binding.outputs with [ output ] -> Some output.slot | _ -> None

  let linear values binding =
    match binding.primitive, external_values values binding.inputs with
    | Primitive.Linear { m; n; k; bias = false; storage },
      Some (input :: weight :: parameters) ->
        Some { m; n; k; storage; input; weight; parameters }
    | _ -> None

  let result_value region slot =
    region.results
    |> List.find_opt (fun result -> Slot.equal result.slot slot)
    |> Option.map (fun result -> result.value)

  let binding_output binding =
    match binding.outputs with [ output ] -> Some output.slot | _ -> None

  let producer region slot =
    List.find_opt
      (fun binding ->
        Option.exists (fun output -> Slot.equal output slot)
          (binding_output binding))
      region.bindings

  let value_origin region values slot =
    match producer region slot with
    | Some { primitive = Primitive.Cast _; inputs = [ source ]; _ } ->
        Slot_map.find_opt source values
    | _ -> Slot_map.find_opt slot values

  let scalar_float = function
    | Ir.Scalar.Bool _ -> None
    | Ir.Scalar.Int value -> Some (Float.of_int value)
    | Ir.Scalar.Float value -> Some value

  let square = function
    | Primitive.Pow (Ir.Scalar.Int 2)
    | Primitive.Pow (Ir.Scalar.Float 2.0) -> true
    | _ -> false

  let inverse_square_root = function
    | Primitive.Rsqrt
    | Primitive.Pow (Ir.Scalar.Float (-0.5)) -> true
    | _ -> false

  let normalized_origin region slot =
    match producer region slot with
    | Some { primitive = Primitive.Cast _; inputs = [ source ]; _ } -> source
    | _ -> slot

  let rms_core region values normalized_slot =
    match producer region normalized_slot with
    | Some
        { primitive = Primitive.Binary Primitive.Mul;
          inputs = [ left; right ]; _ } ->
        let try_orientation input_slot inverse_slot =
          match producer region inverse_slot with
          | Some
              { primitive = Primitive.Unary inverse;
                inputs = [ stabilized ]; _ }
            when inverse_square_root inverse -> (
              match producer region stabilized with
              | Some
                  { primitive =
                      Primitive.Scalar_binary
                        { operator = Primitive.Add; scalar; _ };
                    inputs = [ mean ]; _ } -> (
                  match scalar_float scalar, producer region mean with
                  | Some epsilon,
                    Some
                      { primitive =
                          Primitive.Reduce
                            { Ir.Reduction.operator = Ir.Reduction.Mean;
                              keepdim = true; _ };
                        inputs = [ squared ]; _ }
                    when Float.is_finite epsilon && epsilon > 0.0 -> (
                      match producer region squared with
                      | Some
                          { primitive = Primitive.Unary power;
                            inputs = [ source ]; _ }
                        when square power && Slot.equal source input_slot ->
                          Option.map
                            (fun input -> input, epsilon)
                            (value_origin region values input_slot)
                      | _ -> None)
                  | _ -> None)
              | _ -> None)
          | _ -> None
        in
        (match try_orientation left right with
        | Some matched -> Some matched
        | None -> try_orientation right left)
    | _ -> None

  let rms_scale region values (scale : binding) =
    match scale.inputs, binding_output scale with
    | [ left; right ], Some scaled ->
        let try_orientation normalized weight_slot =
          match
            rms_core region values (normalized_origin region normalized),
            value_origin region values weight_slot
          with
          | Some (input, epsilon), Some weight ->
              Some (input, weight, epsilon, scaled)
          | _ -> None
        in
        (match try_orientation left right with
        | Some matched -> Some matched
        | None -> try_orientation right left)
    | _ -> None

  let scale_binding region slot =
    match producer region slot with
    | Some ({ primitive = Primitive.Binary Primitive.Mul; _ } as scale) ->
        Some scale
    | Some { primitive = Primitive.Cast _; inputs = [ scaled ]; _ } -> (
        match producer region scaled with
        | Some ({ primitive = Primitive.Binary Primitive.Mul; _ } as scale) ->
            Some scale
        | _ -> None)
    | _ -> None

  let rms_parameters region values slot =
    match producer region slot with
    | Some
        { primitive = Primitive.Rms_norm { epsilon };
          inputs = [ input; weight ]; _ } -> (
        match value_origin region values input,
              value_origin region values weight
        with
        | Some input, Some weight -> Some (input, weight, epsilon)
        | _ -> None)
    | _ -> (
        match scale_binding region slot with
        | Some scale ->
            rms_scale region values scale
            |> Option.map (fun (input, weight, epsilon, _) ->
                   input, weight, epsilon)
        | None -> None)

  let rms_normalize region values =
    let result =
      match region.results with [ result ] -> Some result | _ -> None
    in
    let final_scale result_slot =
      let direct =
        region.bindings
        |> List.find_opt (fun binding ->
               Option.exists (Slot.equal result_slot) (binding_output binding))
      in
      match direct with
      | Some ({ primitive = Primitive.Binary Primitive.Mul; _ } as scale) ->
          Some scale
      | Some { primitive = Primitive.Cast _; inputs = [ scale_slot ]; _ } -> (
          match producer region scale_slot with
          | Some ({ primitive = Primitive.Binary Primitive.Mul; _ } as scale) ->
              Some scale
          | _ -> None)
      | _ -> None
    in
    let try_normalized normalized_slot =
      match producer region normalized_slot with
      | Some
          { primitive = Primitive.Binary Primitive.Mul;
            inputs = [ left; right ]; _ } ->
          let try_orientation input_slot inverse_slot =
            match producer region inverse_slot with
            | Some
                { primitive = Primitive.Unary inverse;
                  inputs = [ stabilized ]; _ }
              when inverse_square_root inverse -> (
                match producer region stabilized with
                | Some
                    { primitive =
                        Primitive.Scalar_binary
                          { operator = Primitive.Add; scalar; _ };
                      inputs = [ mean ]; _ } -> (
                    match scalar_float scalar, producer region mean with
                    | Some epsilon,
                      Some
                        { primitive =
                            Primitive.Reduce
                              { Ir.Reduction.operator = Ir.Reduction.Mean;
                                keepdim = true; _ };
                          inputs = [ squared ]; _ }
                      when Float.is_finite epsilon && epsilon > 0.0 -> (
                        match producer region squared with
                        | Some
                            { primitive = Primitive.Unary power;
                              inputs = [ source ]; _ }
                          when square power && Slot.equal source input_slot ->
                            Option.map
                              (fun input -> input, epsilon)
                              (value_origin region values input_slot)
                        | _ -> None)
                    | _ -> None)
                | _ -> None)
            | _ -> None
          in
          (match try_orientation left right with
          | Some matched -> Some matched
          | None -> try_orientation right left)
      | _ -> None
    in
    match result with
    | None -> None
    | Some result -> (
        let match_rms slot =
          let try_unweighted () =
            let normalized = normalized_origin region slot in
            match try_normalized normalized with
            | Some (input, epsilon) -> Some (input, None, epsilon)
            | None -> None
          in
          match final_scale slot with
          | Some { inputs = [ left; right ]; _ } ->
              let try_orientation normalized weight_slot =
                let normalized = normalized_origin region normalized in
                match try_normalized normalized,
                      value_origin region values weight_slot
                with
                | Some (input, epsilon), Some weight ->
                    Some (input, Some weight, epsilon)
                | _ -> None
              in
              (match try_orientation left right with
              | Some tactic -> Some tactic
              | None -> (
                  match try_orientation right left with
                  | Some tactic -> Some tactic
                  | None -> try_unweighted ()))
          | _ -> try_unweighted ()
        in
        let match_residual_slot slot =
          match producer region slot with
          | Some
              { primitive = Primitive.Binary Primitive.Add;
                inputs = [ left; right ]; _ } ->
              let try_residual norm_slot residual_slot =
                match match_rms norm_slot, value_origin region values residual_slot with
                | Some (input, weight, epsilon), Some residual ->
                    Some (input, weight, Some residual, epsilon)
                | _ -> None
              in
              (match try_residual left right with
              | Some tactic -> Some tactic
              | None -> try_residual right left)
          | _ -> None
        in
        let match_rms_or_residual slot =
          match match_residual_slot slot with
          | Some tactic -> Some tactic
          | None -> (
              match match_rms slot with
              | Some (input, weight, epsilon) ->
                  Some (input, weight, None, epsilon)
              | None -> None)
        in
        let match_scale () =
          match producer region result.slot with
          | Some
              { primitive = Primitive.Binary Primitive.Mul;
                inputs = [ left; right ]; _ } ->
              let try_scale pre_slot scale_slot =
                match value_origin region values scale_slot with
                | Some scale
                  when Tensor_shape.numel (Ir.Value.logical_shape scale) = 1 -> (
                    match match_rms_or_residual (normalized_origin region pre_slot) with
                    | Some (input, weight, residual, epsilon) ->
                        Some
                          (Rms_normalize
                             { input; weight; residual; layer_scalar = Some scale;
                               output = result.value; epsilon })
                    | None -> None)
                | _ -> None
              in
              (match try_scale left right with
              | Some tactic -> Some tactic
              | None -> try_scale right left)
          | _ -> None
        in
        match match_scale () with
        | Some tactic -> Some tactic
        | None -> (
            match match_rms_or_residual result.slot with
            | Some (input, weight, residual, epsilon) ->
                Some
                  (Rms_normalize
                     { input; weight; residual; layer_scalar = None;
                       output = result.value; epsilon })
            | None -> None))

  let rms_rope region values =
    let external_inputs = input_map region in
    let tensor_input slot = input_value external_inputs slot in
    let multiply slot =
      match producer region slot with
      | Some
          { primitive = Primitive.Binary Primitive.Mul;
            inputs = [ left; right ]; _ } -> Some (left, right)
      | _ -> None
    in
    let orient_external left right =
      match tensor_input left, tensor_input right with
      | None, Some external_value -> Some (left, external_value)
      | Some external_value, None -> Some (right, external_value)
      | _ -> None
    in
    match region.results with
    | [ result ] -> (
        let sum_slot, token_major_layout =
          match producer region result.slot with
          | Some
              { primitive =
                  Primitive.Movement
                    (Ir.Movement.Transpose { axis0 = 1; axis1 = 2 });
                inputs = [ summed ]; _ } -> summed, true
          | _ -> result.slot, false
        in
        match producer region sum_slot with
        | Some
            { primitive = Primitive.Binary Primitive.Add;
              inputs = [ direct_term; rotated_term ]; _ } ->
            let try_terms direct_term rotated_term =
              match multiply direct_term, multiply rotated_term with
              | Some direct_inputs, Some rotated_inputs -> (
                  match
                    orient_external (fst direct_inputs) (snd direct_inputs),
                    orient_external (fst rotated_inputs) (snd rotated_inputs)
                  with
                  | Some (transposed, cosine), Some (rotated, sine) -> (
                      match producer region rotated with
                      | Some
                          { primitive =
                              Primitive.Movement
                                (Ir.Movement.Concat { axis = 3 });
                            inputs = [ negated; low ]; _ } -> (
                          match producer region negated, producer region low with
                          | ( Some
                                { primitive = Primitive.Unary Primitive.Neg;
                                  inputs = [ high ]; _ },
                              Some
                                { primitive =
                                    Primitive.Movement (Ir.Movement.Index low_index);
                                  inputs = [ low_source ]; _ } ) -> (
                              match producer region high with
                              | Some
                                  { primitive =
                                      Primitive.Movement
                                        (Ir.Movement.Index high_index);
                                    inputs = [ high_source ]; _ }
                                when Slot.equal low_source transposed
                                     && Slot.equal high_source transposed ->
                                  let half_dimension =
                                    match
                                      Tensor_shape.Index.selectors low_index,
                                      Tensor_shape.Index.selectors high_index
                                    with
                                    | ( [ _; _; _;
                                          Tensor_shape.Index.Slice
                                            { start = 0; step = 1; length } ],
                                        [ _; _; _;
                                          Tensor_shape.Index.Slice
                                            { start; step = 1;
                                              length = high_length } ] )
                                      when start = length
                                           && high_length = length -> Some length
                                    | _ -> None
                                  in
                                  let scaled =
                                    if token_major_layout then Some transposed
                                    else
                                      match producer region transposed with
                                      | Some
                                          { primitive =
                                              Primitive.Movement
                                                (Ir.Movement.Transpose
                                                  { axis0 = 1; axis1 = 2 });
                                            inputs = [ scaled ]; _ } -> Some scaled
                                      | _ -> None
                                  in
                                  (match half_dimension, scaled with
                                  | Some half_dimension, Some scaled -> (
                                      match rms_parameters region values scaled with
                                      | Some (input, weight, epsilon) -> (
                                              match
                                                Ir.Rms_rope.create ~epsilon
                                                  ~half_dimension
                                              with
                                              | Ok config ->
                                                  Some
                                                    (Rms_rope
                                                       { config; input; weight;
                                                         cosine; sine;
                                                         output = result.value })
                                              | Error _ -> None)
                                      | None -> None)
                                  | _ -> None)
                              | _ -> None)
                          | _ -> None)
                      | _ -> None)
                  | _ -> None)
              | _ -> None
            in
            (match try_terms direct_term rotated_term with
            | Some tactic -> Some tactic
            | None -> try_terms rotated_term direct_term)
        | _ -> None)
    | _ -> None

  let short_conv_silu region values =
    let shapes =
      let inputs =
        List.fold_left
          (fun shapes ({ slot; value } : input) ->
            Slot_map.add slot (Ir.Value.logical_shape value) shapes)
          Slot_map.empty region.inputs
      in
      List.fold_left
        (fun shapes binding ->
          List.fold_left
            (fun shapes ({ slot; shape; _ } : output_spec) ->
              Slot_map.add slot shape shapes)
            shapes binding.outputs)
        inputs region.bindings
    in
    let dimensions slot =
      Slot_map.find_opt slot shapes |> Option.map Tensor_shape.dimensions
    in
    let transpose slot =
      match producer region slot with
      | Some
          { primitive =
              Primitive.Movement
                (Ir.Movement.Transpose { axis0 = 1; axis1 = 2 });
            inputs = [ input ]; _ } -> Some input
      | _ -> None
    in
    match region.results with
    | [ result ] -> (
        let ( let* ) = Option.bind in
        let* silu = transpose result.slot in
        let* trimmed =
          match producer region silu with
          | Some { primitive = Primitive.Unary Primitive.Silu; inputs = [ input ]; _ } ->
              Some input
          | _ -> None
        in
        let* convolution_output, output_start, length =
          match producer region trimmed with
          | Some
              { primitive = Primitive.Movement (Ir.Movement.Index index);
                inputs = [ input ]; _ } -> (
              match Tensor_shape.Index.selectors index with
              | [ _; _;
                  Tensor_shape.Index.Slice { start; step = 1; length } ] ->
                  Some (input, start, length)
              | _ -> None)
          | _ -> None
        in
        let* convolution, channel_major_input, weight_slot =
          match producer region convolution_output with
          | Some
              { primitive = Primitive.Short_conv convolution;
                inputs = [ input; weight ]; _ } ->
              Some (convolution, input, weight)
          | _ -> None
        in
        let* input_slot = transpose channel_major_input in
        let* input = Slot_map.find_opt input_slot values in
        let* weight = Slot_map.find_opt weight_slot values in
        let* input_dimensions = dimensions input_slot in
        let* channel_dimensions = dimensions channel_major_input in
        let* weight_dimensions = dimensions weight_slot in
        let* convolution_dimensions = dimensions convolution_output in
        let* output_dimensions = dimensions result.slot in
        match
          input_dimensions, channel_dimensions, weight_dimensions,
          convolution_dimensions, output_dimensions
        with
        | ( [ batches; tokens; channels ],
            [ input_batches; input_channels; input_tokens ],
            [ weight_channels; 1; _ ],
            [ conv_batches; conv_channels; conv_width ],
            [ output_batches; output_tokens; output_channels ] )
          when batches > 0 && tokens > 0 && channels > 0
               && input_batches = batches && input_tokens = tokens
               && input_channels = channels && weight_channels = channels
               && conv_batches = batches && conv_channels = channels
               && output_batches = batches && output_tokens = tokens
               && output_channels = channels && length = tokens
               && Ir.Short_conv.groups convolution = channels
               && output_start >= 0 && output_start + tokens <= conv_width ->
            Ir.Short_conv_silu.create ~convolution ~output_start
            |> Result.to_option
            |> Option.map (fun config ->
                   Short_conv_silu { config; input; weight; output = result.value })
        | _ -> None)
    | _ -> None

  type padded_input = {
    source_slot : Slot.t;
    padded_slot : Slot.t;
    source_value : Ir.Value.t;
  }

  let gated_delta region values =
    let ( let* ) = Option.bind in
    let shapes =
      let inputs =
        List.fold_left
          (fun shapes ({ slot; value } : input) ->
            Slot_map.add slot (Ir.Value.logical_shape value) shapes)
          Slot_map.empty region.inputs
      in
      List.fold_left
        (fun shapes binding ->
          List.fold_left
            (fun shapes ({ slot; shape; _ } : output_spec) ->
              Slot_map.add slot shape shapes)
            shapes binding.outputs)
        inputs region.bindings
    in
    let dimensions slot =
      Slot_map.find_opt slot shapes |> Option.map Tensor_shape.dimensions
    in
    let consumers =
      List.fold_left
        (fun consumers (binding : binding) ->
          List.fold_left
            (fun consumers slot ->
              let current =
                Slot_map.find_opt slot consumers |> Option.value ~default:[]
              in
              Slot_map.add slot (binding :: current) consumers)
            consumers binding.inputs)
        Slot_map.empty region.bindings
    in
    let consumers_of slot =
      Slot_map.find_opt slot consumers |> Option.value ~default:[]
    in
    let rec collect slot visited bindings pads =
      if List.exists (Slot.equal slot) visited then bindings, pads
      else
        let visited = slot :: visited in
        match producer region slot with
        | None -> bindings, pads
        | Some binding -> (
            match binding.primitive, binding.inputs, binding_output binding with
            | Primitive.Pad_right_zero { axis = 2 }, [ source ], Some padded ->
                (match input_value values source with
                | Some source_value ->
                    binding :: bindings,
                    { source_slot = source; padded_slot = padded; source_value }
                    :: pads
                | None -> binding :: bindings, pads)
            | _ ->
                List.fold_left
                  (fun (bindings, pads) input ->
                    collect input visited bindings pads)
                  (binding :: bindings, pads) binding.inputs)
    in
    let exactly_one predicate candidates =
      match List.filter predicate candidates with [ candidate ] -> Some candidate | _ -> None
    in
    let result = match region.results with [ result ] -> Some result | _ -> None in
    let* result = result in
    let* final_input =
      match producer region result.slot with
      | Some
          { primitive = Primitive.Cast Ir.Dtype.Float16;
            inputs = [ input ]; _ }
        when Ir.Value.dtype result.value = Ir.Dtype.Float16 ->
          Some input
      | _ -> None
    in
    let bindings, pads = collect final_input [] [] [] in
    let bindings =
      bindings
      |> List.fold_left
           (fun unique binding ->
             match binding_output binding with
             | None -> unique
             | Some slot -> Slot_map.add slot binding unique)
           Slot_map.empty
      |> Slot_map.bindings |> List.map snd
    in
    let pads =
      pads
      |> List.fold_left
           (fun unique pad -> Slot_map.add pad.source_slot pad unique)
           Slot_map.empty
      |> Slot_map.bindings |> List.map snd
    in
    let* batch, tokens, heads, width =
      match Tensor_shape.dimensions (Ir.Value.logical_shape result.value) with
      | [ batch; tokens; heads; width ]
        when batch > 0 && tokens > 0 && heads > 0 && width > 0
             && width mod 32 = 0 ->
          Some (batch, tokens, heads, width)
      | _ -> None
    in
    let vector_pads, scalar_pads =
      List.partition
        (fun pad ->
          Ir.Value.dtype pad.source_value = Ir.Dtype.Float32
          && dimensions pad.source_slot = Some [ batch; heads; tokens; width ])
        pads
    in
    let scalar_pads =
      List.filter
        (fun pad ->
          Ir.Value.dtype pad.source_value = Ir.Dtype.Float32
          && dimensions pad.source_slot = Some [ batch; heads; tokens ])
        scalar_pads
    in
    if List.length vector_pads <> 3 || List.length scalar_pads <> 2 then None
    else
      let q_pad =
        exactly_one
          (fun pad ->
            consumers_of pad.padded_slot
            |> List.exists (fun binding ->
                   match binding.primitive, binding.inputs with
                   | Primitive.Scalar_binary { operator = Primitive.Mul; _ },
                     [ input ] ->
                       Slot.equal input pad.padded_slot
                   | _ -> false))
          vector_pads
      in
      let k_pad =
        exactly_one
          (fun pad ->
            consumers_of pad.padded_slot
            |> List.exists (fun reshape ->
                   match reshape.primitive, reshape.inputs,
                         binding_output reshape with
                   | Primitive.Movement Ir.Movement.Reshape, [ input ],
                     Some reshaped
                     when Slot.equal input pad.padded_slot ->
                       consumers_of reshaped
                       |> List.exists (fun transpose ->
                              match transpose.primitive, transpose.inputs with
                              | ( Primitive.Movement
                                    (Ir.Movement.Transpose { axis0; axis1 }),
                                  [ actual ] ) ->
                                  Slot.equal actual reshaped
                                  && ((axis0 = 4 && axis1 = 3)
                                     || (axis0 = 3 && axis1 = 4))
                              | _ -> false)
                   | _ -> false))
          vector_pads
      in
      let beta_pad =
        exactly_one
          (fun pad ->
            consumers_of pad.padded_slot
            |> List.exists (fun binding ->
                   match binding.primitive, binding.inputs with
                   | Primitive.Movement (Ir.Movement.Unsqueeze 3), [ input ] ->
                       Slot.equal input pad.padded_slot
                   | _ -> false))
          scalar_pads
      in
      let* q_pad = q_pad in
      let* k_pad = k_pad in
      let* beta_pad = beta_pad in
      if Slot.equal q_pad.source_slot k_pad.source_slot then None
      else
        let* value_pad =
          exactly_one
            (fun pad ->
              not (Slot.equal pad.source_slot q_pad.source_slot)
              && not (Slot.equal pad.source_slot k_pad.source_slot))
            vector_pads
        in
        let* gate_pad =
          exactly_one
            (fun pad -> not (Slot.equal pad.source_slot beta_pad.source_slot))
            scalar_pads
        in
        let triangular, batched_matmul, update_slice =
          List.fold_left
            (fun (triangular, batched_matmul, update_slice) binding ->
              match binding.primitive with
              | Primitive.Triangular_recurrence _ ->
                  triangular + 1, batched_matmul, update_slice
              | Primitive.Batched_matmul ->
                  triangular, batched_matmul + 1, update_slice
              | Primitive.Update_slice _ ->
                  triangular, batched_matmul, update_slice + 1
              | _ -> triangular, batched_matmul, update_slice)
            (0, 0, 0) bindings
        in
        if triangular <> 1 || batched_matmul < 4 || update_slice < 1 then None
        else
          Some
            (Gated_delta
               { query = q_pad.source_value;
                 key = k_pad.source_value;
                 value = value_pad.source_value;
                 gate = gate_pad.source_value;
                 beta = beta_pad.source_value;
                 output = result.value })

  (* Generalized epilogue folding for single-linear regions.

     [chain_of region values linear_output slot] classifies the value bound
     at [slot] relative to the linear contraction's output slot:

     - [Folded (epilogue, depth)] when [slot] is reachable from the linear
       output through the foldable pointwise grammar (Silu/Gelu unaries and
       Add/Mul binaries whose non-chain operand is an external region input),
       carrying the [Epilogue.t] that rewrites [slot] as a function of the
       contraction accumulator together with the number of chain bindings;
     - [External value] when [slot] is a region input leaf;
     - [Nonfoldable] otherwise (multiple linears, internal fan-out to a
       non-chain consumer, unary operators outside the Applicative IR, ...).

     Composition runs outward from the accumulator with [Epilogue.compose],
     so the returned expression is the full associative pointwise readback. *)

  type chain =
    | Folded of Epilogue.t * int
    | External of Ir.Value.t
    | Nonfoldable

  let rec chain_of region values linear_output slot =
    if Slot.equal slot linear_output then Folded (Epilogue.none, 0)
    else
      match producer region slot with
      | None -> (
          match Slot_map.find_opt slot values with
          | Some value -> External value
          | None -> Nonfoldable)
      | Some binding -> (
          match binding.primitive, binding.inputs with
          | Primitive.Unary activation, [ operand ]
            when
              (match activation with
              | Primitive.Silu | Primitive.Gelu -> true
              | Primitive.Sigmoid | Primitive.Neg | Primitive.Rsqrt
              | Primitive.Cos | Primitive.Sin | Primitive.Tanh
              | Primitive.Exp | Primitive.Softplus | Primitive.Pow _ ->
                  false) -> (
              match chain_of region values linear_output operand with
              | Folded (producer, depth) ->
                  let consumer =
                    match activation with
                    | Primitive.Silu -> Epilogue.silu
                    | _ -> Epilogue.gelu
                  in
                  Folded (Epilogue.compose ~producer ~consumer, depth + 1)
              | External _ | Nonfoldable -> Nonfoldable)
          | Primitive.Binary Primitive.Add, [ left; right ]
          | Primitive.Binary Primitive.Mul, [ left; right ] -> (
              match
                (chain_of region values linear_output left,
                 chain_of region values linear_output right)
              with
              | Folded (producer, depth), External value
              | External value, Folded (producer, depth) ->
                  let consumer =
                    match binding.primitive with
                    | Primitive.Binary Primitive.Add ->
                        Epilogue.residual_add ~residual:value
                    | _ -> Epilogue.mul ~factor:value
                  in
                  Folded (Epilogue.compose ~producer ~consumer, depth + 1)
              | _ -> Nonfoldable)
          | _ -> Nonfoldable)

  let linear_epilogue region values =
    let linears =
      region.bindings
      |> List.filter_map (fun binding ->
             match linear values binding, output_slot binding with
             | Some linear, Some output -> Some (linear, output)
             | _ -> None)
    in
    match linears with
    | [ (linear, linear_output) ] ->
        (* Admission: select every region whose entire binding set lies on the
           foldable pointwise chain.  Flat residual-add regions keep routing
           through the legacy _add kernels (byte-identical), and richer chains
           are served by digest-keyed kernels whose header carries the
           canonical epilogue buffers; an admitted-but-unserved epilogue is
           never silent because Metal_runtime fails loudly.  A bare single
           linear (identity epilogue, depth 0) is not a fused tactic and stays
           unclaimed. *)
        let candidates =
          region.results
          |> List.filter_map (fun result ->
                 match chain_of region values linear_output result.slot with
                 | Folded (epilogue, depth) when depth >= 1 ->
                     Some (result, epilogue, depth)
                 | Folded (_, _) | External _ | Nonfoldable -> None)
        in
        (match candidates with
        | [ (result, epilogue, depth) ]
          when depth + 1 = List.length region.bindings ->
            Some
              (Linear_epilogue
                 { linear; epilogue; output = result.value })
        | _ -> None)
    | _ -> None

  let fold_linear_epilogue region =
    let values = value_map region in
    let linears =
      region.bindings
      |> List.filter_map (fun binding ->
             match linear values binding, output_slot binding with
             | Some linear, Some output -> Some (linear, output)
             | _ -> None)
    in
    match linears with
    | [ (_, linear_output) ] ->
        let folded =
          region.results
          |> List.filter_map (fun result ->
                 match chain_of region values linear_output result.slot with
                 | Folded (epilogue, depth) -> Some (epilogue, depth)
                 | External _ | Nonfoldable -> None)
        in
        (match folded with
        | [ (epilogue, depth) ]
          when depth + 1 = List.length region.bindings ->
            Some epilogue
        | _ -> None)
    | _ -> None

  let gated_product region values =
    let linears =
      region.bindings
      |> List.filter_map (fun binding ->
             match linear values binding, output_slot binding with
             | Some linear, Some output -> Some (linear, output)
             | _ -> None)
    in
    let unaries =
      region.bindings
      |> List.filter_map (fun binding ->
             match binding.primitive, binding.inputs, output_slot binding with
             | Primitive.Unary activation, [ input ], Some output ->
                 Some (activation, input, output)
             | _ -> None)
    in
    let products =
      region.bindings
      |> List.filter_map (fun binding ->
             match binding.primitive, binding.inputs, output_slot binding with
             | Primitive.Binary Primitive.Mul, [ left; right ], Some output ->
                 Some (left, right, output)
             | _ -> None)
    in
    match linears, unaries, products with
    | [ first; second ], [ (activation, activated_input, activated_output) ],
      [ (left, right, product_output) ] ->
        let oriented =
          if Slot.equal (snd first) activated_input then
            Some (fst first, fst second, snd second)
          else if Slot.equal (snd second) activated_input then
            Some (fst second, fst first, snd first)
          else None
        in
        (match oriented, result_value region product_output with
        | Some (gate, up, up_output), Some output
          when
            (match activation with
            | Primitive.Silu | Primitive.Gelu | Primitive.Sigmoid -> true
            | Primitive.Neg | Primitive.Rsqrt | Primitive.Cos | Primitive.Sin
            | Primitive.Tanh | Primitive.Exp | Primitive.Softplus
            | Primitive.Pow _ -> false)
            && ((Slot.equal left activated_output && Slot.equal right up_output)
            || (Slot.equal right activated_output && Slot.equal left up_output))
            && gate.m = up.m && gate.n = up.n && gate.k = up.k
            && gate.storage = up.storage
            && gate.parameters = [] && up.parameters = []
            && Ir.Value.equal gate.input up.input ->
              Some (Gated_product { gate; up; activation; output })
        | _ -> None)
    | _ -> None

  let select region =
    let values = value_map region in
    match gated_delta region values with
    | Some tactic -> Some tactic
    | None -> (
    match short_conv_silu region values with
    | Some tactic -> Some tactic
    | None -> (
    match linear_epilogue region values with
    | Some tactic -> Some tactic
    | None -> (
        match gated_product region values with
        | Some tactic -> Some tactic
        | None -> (
            match rms_rope region values with
            | Some tactic -> Some tactic
            | None -> rms_normalize region values))))

  let select_all region =
    let original_results = region.results in
    let candidate_results =
      region.bindings
      |> List.rev
      |> List.concat_map (fun binding ->
             List.filter_map
               (fun ({ slot; value; _ } : output_spec) ->
                 Option.map
                   (fun value -> { slot; value; storage = Fresh })
                   value)
               binding.outputs)
    in
    let candidates = original_results @ candidate_results in
    let rec collect seen reversed = function
      | [] -> List.rev reversed
      | result :: rest ->
          let id = Ir.Value.id result.value in
          if List.mem id seen then collect seen reversed rest
          else
            let seen = id :: seen in
            let candidate = { region with results = [ result ] } in
            (match select candidate with
            | Some tactic -> collect seen (tactic :: reversed) rest
            | None -> collect seen reversed rest)
    in
    collect [] [] candidates
