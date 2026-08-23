let dtype_matches fx_dtype archive_dtype =
  match fx_dtype, archive_dtype with
  | Ir.Dtype.Float32, Weight_archive.Dtype.F32
  | Ir.Dtype.Float16, Weight_archive.Dtype.F16
  | Ir.Dtype.Bfloat16, Weight_archive.Dtype.BF16
  | Ir.Dtype.Int64, Weight_archive.Dtype.I64
  | Ir.Dtype.Int32, Weight_archive.Dtype.I32
  | Ir.Dtype.Int8, Weight_archive.Dtype.I8
  | Ir.Dtype.Bool, Weight_archive.Dtype.Bool -> true
  | _ -> false

let validate_tensor archive input =
  let key = Serving_schedule.Tensor_input.key input in
  let value = Serving_schedule.Tensor_input.value input in
  match Weight_archive.find archive key with
  | None -> Error ("serving tensor store is missing schedule binding: " ^ key)
  | Some tensor ->
      if
        not
          (dtype_matches (Ir.Value.dtype value)
             (Weight_archive.Tensor.dtype tensor))
      then
        Error
          (Printf.sprintf
             "serving tensor %s dtype does not match the binary schedule" key)
      else
        let shape =
          Ir.Value.logical_shape value |> Tensor_shape.dimensions
        in
        if shape <> Weight_archive.Tensor.shape tensor then
          Error
            (Printf.sprintf
               "serving tensor %s shape does not match the binary schedule" key)
        else Ok ()

let validate ~package ~archive =
  let rec inputs = function
    | [] -> Ok ()
    | input :: rest ->
        (match archive with
        | None ->
            if
              Serving_package.stage package
              = Serving_package.Stage.Compiled_graph
            then inputs rest
            else Error "serving package has tensor bindings but no archive"
        | Some archive ->
            Result.bind (validate_tensor archive input) (fun () -> inputs rest))
  in
  match Serving_package.stage package, archive with
  | Serving_package.Stage.Serving, None ->
      Error "serving package tensor archive was not loaded"
  | Serving_package.Stage.Compiled_graph, Some _ ->
      Error "compiled-graph package unexpectedly loaded a tensor archive"
  | _ -> inputs (Serving_package.schedule package |> Serving_schedule.tensor_inputs)
