(* One projection interpreter serves ordinary and expert-indexed contractions. *)
let ( let* ) = Result.bind
module State = Metal_execution_state
module Batch = State.Execution_batch
module Buffer = Metal_native.Buffer

let scratch (context : State.context) name bytes =
  let owner = context.execution_batch in
  match List.assoc_opt name owner.exl3_scratch with
  | Some buffer when Buffer.byte_length buffer >= bytes -> Ok buffer
  | _ ->
      let* buffer = Buffer.create ~runtime:context.runtime ~bytes in
      owner.exl3_scratch <- (name, buffer) :: List.remove_assoc name owner.exl3_scratch;
      owner.resources <- buffer :: owner.resources;
      Ok buffer

let dispatch (context : State.context) name buffers parameters grid =
  let* entry = Metal_native.kernel_entry context.runtime ~name
    ~operation:Kernel_abi.Operation.Linear ~input_dtype:Ir.Dtype.Float16
    ~output_dtype:Ir.Dtype.Float16 in
  Metal_native.Batch.mark_barrier context.batch;
  Metal_native.dispatch ~batch:context.batch context.runtime entry ~buffers ~parameters ~grid

module Routes = struct
  type t = { rows : Buffer.t; tiles : Buffer.t; count : Buffer.t; capacity : int }
end

let projection ?routes (context : State.context) state ~m ~n ~k ~experts ~indices
    ~repeat_input ~top_k input weight output =
  let* _ = Exl3.create ~bits:4 ~k ~n in
  let* key = match State.Value_map.find_opt (Ir.Value.id weight) state.State.tensor_keys with
    | Some key -> Ok key | None -> Error "EXL3 projection requires a named archive weight" in
  let* packed = State.find_value state weight in
  let expert_axis = List.length (Tensor_shape.dimensions (Ir.Value.logical_shape weight)) = 3 in
  let* () = if Ir.Value.dtype weight = Ir.Dtype.Quant Ir.Dtype.Exl3_4
    && Tensor_shape.dimensions (Ir.Value.logical_shape weight) =
       (if expert_axis then [experts; n; k] else [n; k]) then Ok ()
    else Error ("EXL3 weight metadata mismatch: " ^ key) in
  let scale suffix width =
    let* buffer, tensor = Metal_native.tensor context.runtime ~name:(key ^ suffix) in
    let shape = if expert_axis then [experts; width] else [width] in
    if Weight_archive.Tensor.dtype tensor = Weight_archive.Dtype.F16
       && Weight_archive.Tensor.shape tensor = shape
       && Weight_archive.Tensor.byte_length tensor = experts * width * 2
    then Ok buffer else Error ("EXL3 scale metadata mismatch: " ^ key ^ suffix) in
  let* suh = scale ".suh" k in
  let* svh = scale ".svh" n in
  let indexed = Option.is_some indices in
  let indices = Option.value indices ~default:packed in
  let* x = scratch context "projection.input" (m * k * 2) in
  let* y = scratch context "projection.accumulator" (m * n * 4) in
  let* parameters = Metal_parameters.u32s
    [m; n; k; 4; Bool.to_int indexed; Bool.to_int repeat_input; top_k; 0] in
  let* a = dispatch context "exl3_input" [input; suh; x; indices] parameters (k, m, 1) in
  let* b = match routes with
    | Some (routes : Routes.t) ->
        dispatch context "exl3_contract_grouped"
          [x; packed; y; routes.rows; routes.tiles; routes.count] parameters
          ((n / 32) * 256, routes.capacity, 1)
    | None ->
        let name, rows, threads, columns = if indexed || m = 1 then "exl3_contract_vector", 1, 128, 16
          else if m < 8 then "exl3_contract_vector4", 4, 128, 16
          else "exl3_contract_matrix", 16, 256, 32 in
        dispatch context name [x; packed; y; indices] parameters
          ((n / columns) * threads, (m + rows - 1) / rows, 1)
  in
  let* c = dispatch context "exl3_output" [y; svh; output; indices] parameters (n, m, 1) in
  Ok [a; b; c]

let linear context state ~m ~n ~k input weight output =
  let* input = State.find_value state input in
  let* buffer = State.workspace_buffer state output in
  let* kernels = projection context state ~m ~n ~k ~experts:1 ~indices:None
    ~repeat_input:false ~top_k:1 input weight buffer in
  Ok (State.bind_value state output buffer, kernels)

let moe (context : State.context) state ~m ~n_experts ~top_k ~hidden_dim
    ~intermediate_dim ~norm_topk_prob values output =
  match values with
  | [hidden; router; gate; up; down] ->
      let* hidden_buffer = State.find_value state hidden in
      let* router_buffer = State.find_value state router in
      let* indices = scratch context "moe.indices" (m * top_k * 4) in
      let* weights = scratch context "moe.weights" (m * top_k * 2) in
      let* gate_buffer = scratch context "moe.gate" (m * top_k * intermediate_dim * 2) in
      let* up_buffer = scratch context "moe.up" (m * top_k * intermediate_dim * 2) in
      let* product = scratch context "moe.product" (m * top_k * intermediate_dim * 2) in
      let* down_buffer = scratch context "moe.down" (m * top_k * hidden_dim * 2) in
      let* result = State.workspace_buffer state output in
      let name = match Ir.Value.dtype router with
        | Ir.Dtype.Float32 -> "llmopt_moe_topk_softmax_f32"
        | _ -> "llmopt_moe_topk_softmax_f16" in
      let* entry = Metal_native.kernel_entry context.runtime ~name
        ~operation:Kernel_abi.Operation.Moe_swiglu_ffn ~input_dtype:(Ir.Value.dtype router)
        ~output_dtype:Ir.Dtype.Float16 in
      let* parameters = Metal_parameters.moe_topk ~tokens:m ~num_experts:n_experts
        ~top_k ~norm_topk_prob () in
      Metal_native.Batch.mark_barrier context.batch;
      let* topk = Metal_native.dispatch ~batch:context.batch context.runtime entry
        ~buffers:[router_buffer; indices; weights] ~parameters ~grid:(m * 32, 1, 1) in
      let* routes, sorting =
        if m < 8 || n_experts > 64 then Ok (None, []) else
        let pairs = m * top_k in
        let capacity = min pairs ((pairs + 15) / 16 + n_experts) in
        let* tokens = scratch context "moe.sorted_tokens" (pairs * 4) in
        let* rows = scratch context "moe.sorted_rows" (pairs * 4) in
        let* inverse = scratch context "moe.inverse" (pairs * 4) in
        let* tiles = scratch context "moe.tiles" (capacity * 16) in
        let* count = scratch context "moe.tile_count" 4 in
        let* entry = Metal_native.kernel_entry context.runtime ~name:"llmopt_moe_sort_and_bin"
          ~operation:Kernel_abi.Operation.Moe_swiglu_ffn ~input_dtype:Ir.Dtype.Float16
          ~output_dtype:Ir.Dtype.Float16 in
        let* parameters = Metal_parameters.moe_sort ~tokens:m ~num_experts:n_experts ~top_k () in
        Metal_native.Batch.mark_barrier context.batch;
        let* kernel = Metal_native.dispatch ~batch:context.batch context.runtime entry
          ~buffers:[indices; tokens; rows; inverse; tiles; count] ~parameters ~grid:(64,1,1) in
        Ok (Some Routes.{rows; tiles; count; capacity}, [kernel])
      in
      let project = projection ?routes context state ~m:(m * top_k) ~experts:n_experts
        ~indices:(Some indices) ~top_k in
      let* gate_kernels = project ~n:intermediate_dim ~k:hidden_dim ~repeat_input:true
        hidden_buffer gate gate_buffer in
      let* up_kernels = project ~n:intermediate_dim ~k:hidden_dim ~repeat_input:true
        hidden_buffer up up_buffer in
      let count = m * top_k * intermediate_dim in
      let* parameters = Metal_parameters.u32s [count] in
      let* activation = dispatch context "exl3_swiglu" [gate_buffer; up_buffer; product]
        parameters (((count + 255) / 256) * 256, 1, 1) in
      let* down_kernels = project ~n:hidden_dim ~k:intermediate_dim ~repeat_input:false
        product down down_buffer in
      let* parameters = Metal_parameters.u32s [m; hidden_dim; top_k; 0] in
      let* reduce = dispatch context "exl3_moe_reduce" [down_buffer; weights; result]
        parameters (((m * hidden_dim + 255) / 256) * 256, 1, 1) in
      Ok (State.bind_value state output result,
          [topk] @ sorting @ gate_kernels @ up_kernels @ [activation] @ down_kernels @ [reduce])
  | _ -> Error "EXL3 MoE expects hidden, router, gate, up, down"
