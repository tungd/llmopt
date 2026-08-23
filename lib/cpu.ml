open Bigarray

module Tensor = struct
  type t = (float, float32_elt, c_layout) Array2.t

  let create shape = Array2.create float32 c_layout (Shape.rows shape) (Shape.cols shape)

  let of_rows rows =
    let row_count = Array.length rows in
    if row_count = 0 then invalid_arg "Tensor.of_rows: empty input";
    let col_count = Array.length rows.(0) in
    if col_count = 0 then invalid_arg "Tensor.of_rows: empty row";
    Array.iter
      (fun row ->
        if Array.length row <> col_count then invalid_arg "Tensor.of_rows: ragged input")
      rows;
    let tensor = create (Shape.of_ints_exn ~rows:row_count ~cols:col_count) in
    Array.iteri
      (fun row values -> Array.iteri (fun col value -> Array2.set tensor row col value) values)
      rows;
    tensor

  let get tensor row col = Array2.get tensor row col
  let set tensor row col value = Array2.set tensor row col value
  let fill tensor value = Array2.fill tensor value
  let shape tensor =
    Shape.of_ints_exn ~rows:(Array2.dim1 tensor) ~cols:(Array2.dim2 tensor)

  let to_rows tensor =
    Array.init (Array2.dim1 tensor) (fun row ->
        Array.init (Array2.dim2 tensor) (fun col -> get tensor row col))
end

type execution = { outputs : (string, Tensor.t) Hashtbl.t }

let output execution name = Hashtbl.find_opt execution.outputs name

let outputs execution =
  Hashtbl.to_seq execution.outputs |> List.of_seq

type state = {
  input_values : (string, Tensor.t) Hashtbl.t;
  values : (int, Tensor.t) Hashtbl.t;
  execution : execution;
  mutable next_value : int;
}

let failf format = Printf.ksprintf (fun message -> raise (Failure message)) format

let fresh_value state ~logical_shape ~shape ~dtype =
  let value =
    Ir.Value.make_tensor ~id:state.next_value ~shape:logical_shape ~dtype
  in
  if not (Shape.equal (Ir.Value.shape value) shape) then
    invalid_arg "effect logical shape does not match CPU tile shape";
  state.next_value <- state.next_value + 1;
  value

let bind state value tensor =
  let expected = Ir.Value.shape value in
  let actual = Tensor.shape tensor in
  if not (Shape.equal expected actual) then
    failf "CPU shape mismatch for value %d: expected %s, got %s"
      (Ir.Value_id.to_int (Ir.Value.id value)) (Shape.to_string expected)
      (Shape.to_string actual);
  Hashtbl.replace state.values (Ir.Value_id.to_int (Ir.Value.id value)) tensor

let find state value =
  match Hashtbl.find_opt state.values (Ir.Value_id.to_int (Ir.Value.id value)) with
  | Some tensor -> tensor
  | None ->
      failf "CPU value %d has no storage"
        (Ir.Value_id.to_int (Ir.Value.id value))

let copy_into source destination =
  let source_shape = Tensor.shape source in
  let destination_shape = Tensor.shape destination in
  if not (Shape.equal source_shape destination_shape) then
    failf "CPU copy shape mismatch: %s -> %s" (Shape.to_string source_shape)
      (Shape.to_string destination_shape);
  for row = 0 to Shape.rows source_shape - 1 do
    for col = 0 to Shape.cols source_shape - 1 do
      Tensor.set destination row col (Tensor.get source row col)
    done
  done

let matmul left right output =
  let left_shape = Tensor.shape left in
  let output_shape = Tensor.shape output in
  for row = 0 to Shape.rows output_shape - 1 do
    for col = 0 to Shape.cols output_shape - 1 do
      let accumulator = ref 0.0 in
      for inner = 0 to Shape.cols left_shape - 1 do
        accumulator :=
          !accumulator
          +. (Tensor.get left row inner *. Tensor.get right inner col)
      done;
      Tensor.set output row col !accumulator
    done
  done

let linear input weight output bias =
  let input_shape = Tensor.shape input in
  let output_shape = Tensor.shape output in
  for row = 0 to Shape.rows output_shape - 1 do
    for col = 0 to Shape.cols output_shape - 1 do
      let accumulator = ref 0.0 in
      for inner = 0 to Shape.cols input_shape - 1 do
        accumulator :=
          !accumulator
          +. (Tensor.get input row inner *. Tensor.get weight col inner)
      done;
      let value =
        match bias with
        | None -> !accumulator
        | Some bias -> !accumulator +. Tensor.get bias 0 col
      in
      Tensor.set output row col value
  done
  done

let q8_linear input weight scale output bias =
  let input_shape = Tensor.shape input in
  let output_shape = Tensor.shape output in
  for row = 0 to Shape.rows output_shape - 1 do
    for col = 0 to Shape.cols output_shape - 1 do
      let accumulator = ref 0.0 in
      let scale_value = Tensor.get scale 0 col in
      for inner = 0 to Shape.cols input_shape - 1 do
        accumulator :=
          !accumulator
          +. (Tensor.get input row inner
             *. (Tensor.get weight col inner *. scale_value))
      done;
      let value =
        match bias with
        | None -> !accumulator
        | Some bias -> !accumulator +. Tensor.get bias 0 col
      in
      Tensor.set output row col value
    done
  done

let add ~left ~right ~output ~broadcast =
  let output_shape = Tensor.shape output in
  let left_shape = Tensor.shape left in
  let right_shape = Tensor.shape right in
  for row = 0 to Shape.rows output_shape - 1 do
    for col = 0 to Shape.cols output_shape - 1 do
      let left_row, right_row =
        match broadcast with
        | Shape.Same -> row, row
        | Shape.Row ->
            ((if Shape.rows left_shape = 1 then 0 else row),
             (if Shape.rows right_shape = 1 then 0 else row))
      in
      Tensor.set output row col
        (Tensor.get left left_row col +. Tensor.get right right_row col)
    done
  done

let gelu input output =
  let shape = Tensor.shape input in
  for row = 0 to Shape.rows shape - 1 do
    for col = 0 to Shape.cols shape - 1 do
      let value = Tensor.get input row col in
      let scaled = sqrt 2.0 *. (value +. 0.044715 *. value *. value *. value) in
      Tensor.set output row col (0.5 *. value *. (1.0 +. tanh (scaled *. 0.5)))
    done
  done

let relu input output =
  let shape = Tensor.shape input in
  for row = 0 to Shape.rows shape - 1 do
    for col = 0 to Shape.cols shape - 1 do
      Tensor.set output row col (Float.max 0.0 (Tensor.get input row col))
    done
  done

let run ~inputs thunk =
  let input_values = Hashtbl.create (List.length inputs) in
  List.iter (fun (name, tensor) -> Hashtbl.replace input_values name tensor) inputs;
  let state =
    {
      input_values;
      values = Hashtbl.create 32;
      execution = { outputs = Hashtbl.create 8 };
      next_value = 0;
    }
  in
  Effect.Deep.match_with thunk ()
    {
      retc = (fun value -> Ok (value, state.execution));
      exnc = (fun exception_value -> Error exception_value);
      effc =
        (fun (type a) (eff : a Effect.t) ->
          match eff with
          | Tile_effect.Input
              { name; source = _; shape; logical_shape; dtype } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let source =
                    match Hashtbl.find_opt state.input_values name with
                    | Some tensor -> tensor
                    | None -> failf "CPU input %s is not bound" name
                  in
                  let value = fresh_value state ~logical_shape ~shape ~dtype in
                  bind state value source;
                  Effect.Deep.continue continuation value)
          | Tile_effect.Alloc { shape; logical_shape; dtype; _ } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let value = fresh_value state ~logical_shape ~shape ~dtype in
                  let tensor = Tensor.create shape in
                  Tensor.fill tensor 0.0;
                  bind state value tensor;
                  Effect.Deep.continue continuation value)
          | Tile_effect.Copy { src; dst } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  copy_into (find state src) (find state dst);
                  Effect.Deep.continue continuation ())
          | Tile_effect.Async_copy { src; dst; _ } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  copy_into (find state src) (find state dst);
                  Effect.Deep.continue continuation ())
          | Tile_effect.Matmul { lhs; rhs; shape; logical_shape } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let value =
                    fresh_value state ~logical_shape ~shape
                      ~dtype:(Ir.Value.dtype lhs)
                  in
                  let tensor = Tensor.create shape in
                  matmul (find state lhs) (find state rhs) tensor;
                  bind state value tensor;
                  Effect.Deep.continue continuation value)
          | Tile_effect.Linear { input; weight; bias; shape; logical_shape } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let value =
                    fresh_value state ~logical_shape ~shape
                      ~dtype:(Ir.Value.dtype input)
                  in
                  let tensor = Tensor.create shape in
                  linear (find state input) (find state weight) tensor
                    (Option.map (find state) bias);
                  bind state value tensor;
                  Effect.Deep.continue continuation value)
          | Tile_effect.Q8_linear
              { input; weight; scale; bias; shape; logical_shape } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let value =
                    fresh_value state ~logical_shape ~shape
                      ~dtype:(Ir.Value.dtype input)
                  in
                  let tensor = Tensor.create shape in
                  q8_linear (find state input) (find state weight) (find state scale)
                    tensor (Option.map (find state) bias);
                  bind state value tensor;
                  Effect.Deep.continue continuation value)
          | Tile_effect.Add { lhs; rhs; shape; logical_shape; broadcast } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let value =
                    fresh_value state ~logical_shape ~shape
                      ~dtype:(Ir.Value.dtype lhs)
                  in
                  let tensor = Tensor.create shape in
                  add ~left:(find state lhs) ~right:(find state rhs) ~output:tensor ~broadcast;
                  bind state value tensor;
                  Effect.Deep.continue continuation value)
          | Tile_effect.Gelu { input; shape; logical_shape } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let value =
                    fresh_value state ~logical_shape ~shape
                      ~dtype:(Ir.Value.dtype input)
                  in
                  let tensor = Tensor.create shape in
                  gelu (find state input) tensor;
                  bind state value tensor;
                  Effect.Deep.continue continuation value)
          | Tile_effect.Relu { input; shape; logical_shape } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  let value =
                    fresh_value state ~logical_shape ~shape
                      ~dtype:(Ir.Value.dtype input)
                  in
                  let tensor = Tensor.create shape in
                  relu (find state input) tensor;
                  bind state value tensor;
                  Effect.Deep.continue continuation value)
          | Tile_effect.Output { name; value } ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  Hashtbl.replace state.execution.outputs name (find state value);
                  Effect.Deep.continue continuation ())
          | Tile_effect.Barrier_create _ ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  Effect.Deep.continue continuation (Tile_effect.Barrier.of_id 0))
          | Tile_effect.Barrier_arrive _ ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  Effect.Deep.continue continuation ())
          | Tile_effect.Barrier_wait _ ->
              Some
                (fun (continuation : (a, _) Effect.Deep.continuation) ->
                  Effect.Deep.continue continuation ())
          | _ -> None);
    }
