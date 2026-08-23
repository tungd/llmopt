let dtype_matches fx_dtype archive_dtype =
  match fx_dtype, archive_dtype with
  | Ir.Dtype.Float32, Safetensors.Dtype.F32
  | Ir.Dtype.Float16, Safetensors.Dtype.F16
  | Ir.Dtype.Bfloat16, Safetensors.Dtype.BF16
  | Ir.Dtype.Int64, Safetensors.Dtype.I64
  | Ir.Dtype.Int32, Safetensors.Dtype.I32
  | Ir.Dtype.Int8, Safetensors.Dtype.I8
  | Ir.Dtype.Bool, Safetensors.Dtype.Bool -> true
  | _ -> false

let validate_tensor archive node key =
  match Safetensors.find archive key with
  | None -> Error ("serving tensor store is missing FX binding: " ^ key)
  | Some tensor ->
      if not (dtype_matches (Fx.Node.dtype node) (Safetensors.Tensor.dtype tensor))
      then
        Error
          (Printf.sprintf "serving tensor %s dtype does not match FX node %s" key
             (Fx.Node.name node))
      else
        (match Fx.Node.shape node with
        | None ->
            Error ("serving tensor binding has no static FX shape: " ^ key)
        | Some shape when shape <> Safetensors.Tensor.shape tensor ->
            Error
              (Printf.sprintf "serving tensor %s shape does not match FX node %s"
                 key (Fx.Node.name node))
        | Some _ -> Ok ())

let validate ~package ~fx ~archive =
  let rec nodes = function
    | [] -> Ok ()
    | node :: rest ->
        (match Fx.Node.binding node with
        | Fx.Binding.Computed | Fx.Binding.Runtime -> nodes rest
        | Fx.Binding.Tensor_store { key } ->
            if Fx.Node.op node <> "placeholder" && Fx.Node.op node <> "get_attr"
            then
              Error
                ("only FX placeholders and get_attr nodes may bind tensors: "
                ^ Fx.Node.name node)
            else
              (match archive with
              | None ->
                  if Serving_package.stage package
                     = Serving_package.Stage.Compiled_graph
                  then nodes rest
                  else Error "serving package has tensor bindings but no archive"
              | Some archive ->
                  Result.bind (validate_tensor archive node key) (fun () ->
                      nodes rest)))
  in
  match Serving_package.stage package, archive with
  | Serving_package.Stage.Serving, None ->
      Error "serving package tensor archive was not loaded"
  | Serving_package.Stage.Compiled_graph, Some _ ->
      Error "compiled-graph package unexpectedly loaded a tensor archive"
  | _ -> nodes (Fx.nodes fx)
