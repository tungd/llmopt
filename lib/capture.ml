type state = {
  graph : Ir.Graph.t;
  mutable next_barrier : int;
}

let fresh_tensor_value graph ~shape ~logical_shape ~dtype =
  let value = Ir.Graph.fresh_tensor_value graph ~shape:logical_shape ~dtype in
  if not (Shape.equal (Ir.Value.shape value) shape) then
    invalid_arg "effect logical shape does not match tile shape";
  value

let run thunk =
  let state = { graph = Ir.Graph.create (); next_barrier = 0 } in
  Effect.Deep.match_with thunk ()
    {
      retc = (fun value -> Ok (value, state.graph));
      exnc = (fun exception_value -> Error exception_value);
      effc =
        (fun (type a) (eff : a Effect.t) ->
          match eff with
          | Tile_effect.Input { name; source; shape; logical_shape; dtype } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let value =
                    Ir.Graph.tensor_input state.graph ~name ~source
                      ~shape:logical_shape ~dtype
                  in
                  if not (Shape.equal (Ir.Value.shape value) shape) then
                    invalid_arg "effect logical shape does not match tile shape";
                  Effect.Deep.continue continuation value)
          | Tile_effect.Alloc { shape; logical_shape; dtype; space; layout } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let value =
                    fresh_tensor_value state.graph ~shape ~logical_shape ~dtype
                  in
                  Ir.Graph.append state.graph ~op:(Ir.Op.Alloc { space; layout })
                    ~inputs:[] ~output:(Some value);
                  Effect.Deep.continue continuation value)
          | Tile_effect.Copy { src; dst } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  Ir.Graph.append state.graph
                    ~op:(Ir.Op.Copy { asynchronous = false; barrier = None })
                    ~inputs:[ src; dst ] ~output:None;
                  Effect.Deep.continue continuation ())
          | Tile_effect.Async_copy { src; dst; barrier } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  Ir.Graph.append state.graph
                    ~op:
                      (Ir.Op.Copy
                         {
                           asynchronous = true;
                           barrier = Some (Tile_effect.Barrier.id barrier);
                         })
                    ~inputs:[ src; dst ] ~output:None;
                  Effect.Deep.continue continuation ())
          | Tile_effect.Matmul { lhs; rhs; shape; logical_shape } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let lhs_shape = Ir.Value.shape lhs in
                  let rhs_shape = Ir.Value.shape rhs in
                  let output =
                    fresh_tensor_value state.graph ~shape ~logical_shape
                      ~dtype:(Ir.Value.dtype lhs)
                  in
                  Ir.Graph.append state.graph
                    ~op:
                      (Ir.Op.Matmul
                         {
                           m = Shape.rows lhs_shape;
                           n = Shape.cols rhs_shape;
                           k = Shape.cols lhs_shape;
                         })
                    ~inputs:[ lhs; rhs ] ~output:(Some output);
                  Effect.Deep.continue continuation output)
          | Tile_effect.Linear { input; weight; bias; shape; logical_shape } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let input_shape = Ir.Value.shape input in
                  let weight_shape = Ir.Value.shape weight in
                  let output =
                    fresh_tensor_value state.graph ~shape ~logical_shape
                      ~dtype:(Ir.Value.dtype input)
                  in
                  Ir.Graph.append state.graph
                    ~op:
                      (Ir.Op.Linear
                         {
                           m = Shape.rows input_shape;
                           n = Shape.rows weight_shape;
                           k = Shape.cols input_shape;
                           bias = Option.is_some bias;
                         })
                    ~inputs:(input :: weight :: Option.to_list bias)
                    ~output:(Some output);
                  Effect.Deep.continue continuation output)
          | Tile_effect.W4a16_linear
              { input; weight; scale; bias; shape; logical_shape } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let input_shape = Ir.Value.shape input in
                  let weight_shape = Ir.Value.shape weight in
                  let output =
                    fresh_tensor_value state.graph ~shape ~logical_shape
                      ~dtype:(Ir.Value.dtype input)
                  in
                  Ir.Graph.append state.graph
                    ~op:
                      (Ir.Op.W4a16_linear
                         {
                           m = Shape.rows input_shape;
                           n = Shape.rows weight_shape;
                           k = Shape.cols input_shape;
                           bias = Option.is_some bias;
                         })
                    ~inputs:(input :: weight :: scale :: Option.to_list bias)
                    ~output:(Some output);
                  Effect.Deep.continue continuation output)
          | Tile_effect.Add { lhs; rhs; shape; logical_shape; broadcast } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let output =
                    fresh_tensor_value state.graph ~shape ~logical_shape
                      ~dtype:(Ir.Value.dtype lhs)
                  in
                  Ir.Graph.append state.graph ~op:(Ir.Op.Add { broadcast })
                    ~inputs:[ lhs; rhs ] ~output:(Some output);
                  Effect.Deep.continue continuation output)
          | Tile_effect.Gelu { input; shape; logical_shape } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let output =
                    fresh_tensor_value state.graph ~shape ~logical_shape
                      ~dtype:(Ir.Value.dtype input)
                  in
                  Ir.Graph.append state.graph ~op:Ir.Op.Gelu ~inputs:[ input ]
                    ~output:(Some output);
                  Effect.Deep.continue continuation output)
          | Tile_effect.Relu { input; shape; logical_shape } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let output =
                    fresh_tensor_value state.graph ~shape ~logical_shape
                      ~dtype:(Ir.Value.dtype input)
                  in
                  Ir.Graph.append state.graph ~op:Ir.Op.Relu ~inputs:[ input ]
                    ~output:(Some output);
                  Effect.Deep.continue continuation output)
          | Tile_effect.Primitive
              { operation; inputs; shape; logical_shape; dtype } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let output =
                    fresh_tensor_value state.graph ~shape ~logical_shape ~dtype
                  in
                  Ir.Graph.append state.graph ~op:(Ir.Op.Primitive operation)
                    ~inputs ~output:(Some output);
                  Effect.Deep.continue continuation output)
          | Tile_effect.Opaque
              {
                op;
                target;
                arguments;
                keyword_arguments;
                inputs;
                shape;
                logical_shape;
                dtype;
              } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let output =
                    fresh_tensor_value state.graph ~shape ~logical_shape ~dtype
                  in
                  Ir.Graph.append state.graph
                    ~op:
                      (Ir.Op.Opaque
                         { op; target; arguments; keyword_arguments })
                    ~inputs
                    ~output:(Some output);
                  Effect.Deep.continue continuation output)
          | Tile_effect.Output { name; value } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  Ir.Graph.add_output state.graph ~name value;
                  Effect.Deep.continue continuation ())
          | Tile_effect.Barrier_create name ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let id = state.next_barrier in
                  state.next_barrier <- state.next_barrier + 1;
                  Ir.Graph.append state.graph
                    ~op:(Ir.Op.Barrier_create { id; name }) ~inputs:[] ~output:None;
                  Effect.Deep.continue continuation (Tile_effect.Barrier.of_id id))
          | Tile_effect.Barrier_arrive barrier ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  Ir.Graph.append state.graph
                    ~op:(Ir.Op.Barrier_arrive (Tile_effect.Barrier.id barrier))
                    ~inputs:[] ~output:None;
                  Effect.Deep.continue continuation ())
          | Tile_effect.Barrier_wait barrier ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  Ir.Graph.append state.graph
                    ~op:(Ir.Op.Barrier_wait (Tile_effect.Barrier.id barrier))
                    ~inputs:[] ~output:None;
                  Effect.Deep.continue continuation ())
          | _ -> None);
    }
