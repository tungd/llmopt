module Linear_bias = Pass_fuse_linear_bias
module Short_conv_step_fused = Pass_fuse_short_conv_step
module Lm_head_argmax = Pass_fuse_lm_head_argmax
module Co_schedule = Pass_co_schedule
module Epilogues = Pass_fuse_epilogues

let ( let* ) = Result.bind

let fuse_linear_bias = Pass_fuse_linear_bias.run
let discover_swiglu_ffn _ = Ok []
let recover_scans = Kernel_ir.Scan.recover
let fuse_swiglu_ffn graph = Ok (Pass_fuse_epilogues.run graph)
let fuse_short_conv_step = Pass_fuse_short_conv_step.run
let fuse_lm_head_argmax = Pass_fuse_lm_head_argmax.run

let parameterize_token_id_output graph =
  let next_node =
    Ir.Graph.nodes graph
    |> List.fold_left (fun next node -> max next (Ir.node_id node + 1)) 0
    |> ref
  in
  let token_output = ref None in
  let rec rewrite acc = function
    | [] -> Ok (List.rev acc)
    | node :: rest ->
        (match Ir.node_op node, Ir.node_inputs node with
        | Ir.Op.Output { name = "token_id" }, [ input ]
          when Ir.Value.dtype input <> Ir.Dtype.Int32 ->
            let rank =
              Tensor_shape.dimensions (Ir.Value.logical_shape input)
              |> List.length
            in
            if rank = 0 then Error "token_id source must have a vocabulary axis"
            else if
              Ir.Value.dtype input <> Ir.Dtype.Float16
              && Ir.Value.dtype input <> Ir.Dtype.Float32
            then
              Error
                (Printf.sprintf "token_id source must be f16, f32, or i32; got %s"
                   (Ir.Dtype.to_string (Ir.Value.dtype input)))
            else
              let axis = rank - 1 in
              let* shape =
                Tensor_shape.reduce (Ir.Value.logical_shape input) ~axes:[ axis ]
                  ~keepdim:false
                |> Result.map_error Tensor_shape.error_to_string
              in
              let token =
                Ir.Graph.fresh_tensor_value graph ~shape ~dtype:Ir.Dtype.Int32
              in
              let argmax =
                Ir.node_create ~id:(Ir.node_id node)
                  ~op:
                    (Ir.Op.Primitive
                       (Ir.Primitive.Reduce
                          { Ir.Reduction.operator = Argmax;
                            axes = [ axis ];
                            keepdim = false }))
                  ~inputs:[ input ] ~output:(Some token)
              in
              let output =
                Ir.node_create ~id:!next_node
                  ~op:(Ir.Op.Output { name = "token_id" }) ~inputs:[ token ]
                  ~output:None
              in
              incr next_node;
              token_output := Some token;
              rewrite (output :: argmax :: acc) rest
        | _ -> rewrite (node :: acc) rest)
  in
  let* nodes = rewrite [] (Ir.Graph.nodes graph) in
  let outputs =
    Ir.Graph.outputs graph
    |> List.map (fun (name, value) ->
           match name, !token_output with
           | "token_id", Some token -> name, token
           | _ -> name, value)
  in
  Ok (Ir.Graph.with_nodes_and_outputs graph nodes outputs)
let fuse_epilogues = Pass_fuse_epilogues.run
let co_schedule = Pass_co_schedule.run
let co_schedule_plan = Pass_co_schedule.run_plan

let semantic_passes =
  [
    Pass_fuse_epilogues.pass;
    Pass_fuse_linear_bias.pass;
  ]

let default_pipeline = Pass.Pipeline.of_list semantic_passes

module Queue_stage = struct
  type kind = Sequential | Concurrent

  type t = {
    index : int;
    kind : kind;
    requests : Compute_plan.Request.t list;
  }

  let index stage = stage.index
  let kind stage = stage.kind
  let requests stage = stage.requests

  let of_compute_stage stage =
    let requests = Compute_plan.Stage.requests stage in
    let kind = if List.length requests <= 1 then Sequential else Concurrent in
    { index = Compute_plan.Stage.index stage; kind; requests }
end

module Receipt = struct
  module Legacy_matcher = struct
    type t = Quant_linear_add | W4a16_linear_add | Quant_gated_linear
  end

  module Dependency = struct
    type relation = Compute_plan.Edge.relation

    type t = {
      source : Compute_plan.Node_id.t;
      destination : Compute_plan.Node_id.t;
      relation : relation;
    }

    let source dependency = dependency.source
    let destination dependency = dependency.destination
    let relation dependency = dependency.relation
  end

  module Compiler_barrier = struct
    type kind = Create | Arrive | Wait

    type t = {
      node_id : int;
      id : int;
      kind : kind;
    }

    let node_id barrier = barrier.node_id
    let id barrier = barrier.id
    let kind barrier = barrier.kind
  end

  module Linear_selection = struct
    type t = {
      node_id : Compute_plan.Node_id.t;
      operation : Ir.Op.t;
      chosen : Kernel_abi.Linear_tactic.Candidate.t;
      alternatives : Kernel_abi.Linear_tactic.Candidate.t list;
    }

    let node_id selection = selection.node_id
    let operation selection = selection.operation
    let chosen selection = selection.chosen
    let alternatives selection = selection.alternatives
  end

  type t = {
    target : Target_hardware.t;
    decisions : Constructive_lowering.Decision.t list;
    events : Constructive_lowering.Lowering.Event.t list;
    uncovered : Constructive_lowering.Lowering.Uncovered.t list;
    unified_count : int;
    accepted_count : int;
    cut_count : int;
    legacy_invocation_count : int;
    legacy_rewrite_count : int;
    legacy_counts : (Legacy_matcher.t * int * int) list;
    linear_selections : Linear_selection.t list;
    kernels : Kernel_abi.Entry.t list;
    queue_stages : Queue_stage.t list;
    dependencies : Dependency.t list;
    barriers : Compiler_barrier.t list;
  }

  let target receipt = receipt.target
  let decisions receipt = receipt.decisions
  let events receipt = receipt.events
  let uncovered receipt = receipt.uncovered
  let unified_count receipt = receipt.unified_count
  let accepted_count receipt = receipt.accepted_count
  let cut_count receipt = receipt.cut_count
  let legacy_count receipt = receipt.legacy_rewrite_count
  let legacy_invocation_count receipt = receipt.legacy_invocation_count
  let legacy_rewrite_count receipt = receipt.legacy_rewrite_count
  let legacy_counts receipt = receipt.legacy_counts
  let linear_selections receipt = receipt.linear_selections
  let kernels receipt = receipt.kernels
  let queue_stages receipt = receipt.queue_stages
  let dependencies receipt = receipt.dependencies
  let barriers receipt = receipt.barriers

  let node_id_to_json node_id = `Int (Compute_plan.Node_id.to_int node_id)

  let value_to_json value =
    `Assoc
      [ ("id", `Int (Ir.Value_id.to_int (Ir.Value.id value)));
        ( "shape",
          `List
            (Tensor_shape.dimensions (Ir.Value.logical_shape value)
            |> List.map (fun dimension -> `Int dimension)) );
        ("dtype", `String (Ir.Dtype.to_string (Ir.Value.dtype value))) ]

  let relation_to_string = function
    | Compute_plan.Edge.Data -> "data"
    | Compute_plan.Edge.State -> "state"
    | Compute_plan.Edge.Data_and_state -> "data-and-state"

  let effect_to_string = function
    | Compute_plan.Effect.Pure -> "pure"
    | Compute_plan.Effect.Stateful -> "stateful"
    | Compute_plan.Effect.Synchronizing -> "synchronizing"
    | Compute_plan.Effect.Opaque -> "opaque"

  let numerical_to_json = function
    | Constructive_lowering.Boundary.Numerical.Shape_mismatch node ->
        `Assoc [ ("kind", `String "shape-mismatch"); ("node", node_id_to_json node) ]
    | Constructive_lowering.Boundary.Numerical.Dtype_mismatch node ->
        `Assoc [ ("kind", `String "dtype-mismatch"); ("node", node_id_to_json node) ]
    | Constructive_lowering.Boundary.Numerical.Dimension_mismatch node ->
        `Assoc
          [ ("kind", `String "dimension-mismatch");
            ("node", node_id_to_json node) ]

  let resource_to_json = function
    | Constructive_lowering.Boundary.Resource.Threadgroup_size
        { required; available } ->
        `Assoc
          [ ("kind", `String "threadgroup-size");
            ("required", `Int required);
            ("available", `Int available) ]
    | Constructive_lowering.Boundary.Resource.Sram_capacity
        { required; available } ->
        `Assoc
          [ ("kind", `String "sram-capacity");
            ("required", `Int required);
            ("available", `Int available) ]

  let boundary_to_json = function
    | Constructive_lowering.Boundary.Graph_output value ->
        `Assoc
          [ ("kind", `String "graph-output"); ("value", value_to_json value) ]
    | Constructive_lowering.Boundary.External_use { value; consumers } ->
        `Assoc
          [ ("kind", `String "external-use");
            ("value", value_to_json value);
            ( "consumers",
              `List (List.map node_id_to_json consumers) ) ]
    | Constructive_lowering.Boundary.Effect_state
        { relation; source_effect; destination_effect } ->
        `Assoc
          [ ("kind", `String "effect-state");
            ("relation", `String (relation_to_string relation));
            ("source-effect", `String (effect_to_string source_effect));
            ( "destination-effect",
              `String (effect_to_string destination_effect) ) ]
    | Constructive_lowering.Boundary.Unsupported_operation { node; operation } ->
        `Assoc
          [ ("kind", `String "unsupported-operation");
            ("node", node_id_to_json node);
            ("operation", `String (Ir.Op.to_string operation)) ]
    | Constructive_lowering.Boundary.Unsupported_index_map node ->
        `Assoc
          [ ("kind", `String "unsupported-index-map");
            ("node", node_id_to_json node) ]
    | Constructive_lowering.Boundary.Layout_boundary node ->
        `Assoc
          [ ("kind", `String "layout-boundary");
            ("node", node_id_to_json node) ]
    | Constructive_lowering.Boundary.Missing_normalization
        { source; destination } ->
        `Assoc
          [ ("kind", `String "missing-normalization");
            ("source", node_id_to_json source);
            ("destination", node_id_to_json destination) ]
    | Constructive_lowering.Boundary.Numerical_incompatibility numerical ->
        let json = numerical_to_json numerical in
        `Assoc [ ("kind", `String "numerical-incompatibility"); ("detail", json) ]
    | Constructive_lowering.Boundary.Target_resource resource ->
        `Assoc
          [ ("kind", `String "target-resource");
            ("detail", resource_to_json resource) ]

  let decision_to_json decision =
    let candidate = Constructive_lowering.Decision.candidate decision in
    let boundaries =
      Constructive_lowering.Decision.legality decision
      |> Constructive_lowering.Legality.boundaries
      |> List.map boundary_to_json
    in
    `Assoc
      [ ( "status",
          `String
            (match decision with
            | Constructive_lowering.Decision.Accepted _ -> "accepted"
            | Constructive_lowering.Decision.Cut _ -> "cut") );
        ("source", node_id_to_json (Constructive_lowering.Decision.source candidate));
        ( "destination",
          node_id_to_json (Constructive_lowering.Decision.destination candidate) );
        ( "relation",
          `String
            (relation_to_string (Constructive_lowering.Decision.relation candidate)) );
        ("boundaries", `List boundaries) ]

  let kind_to_string = function
    | Constructive_lowering.Lowering.Kind.Linear_add -> "linear-add"
    | Constructive_lowering.Lowering.Kind.W4a16_linear_add -> "w4a16-linear-add"
    | Constructive_lowering.Lowering.Kind.Gated_linear -> "gated-linear"
    | Constructive_lowering.Lowering.Kind.Rms_norm -> "rms-normalize"
    | Constructive_lowering.Lowering.Kind.Rms_rope -> "rms-rope"
    | Constructive_lowering.Lowering.Kind.Short_conv_silu -> "short-conv-silu"
    | Constructive_lowering.Lowering.Kind.Gated_delta -> "gated-delta"

  let event_to_json event =
    `Assoc
      [ ("kind", `String (kind_to_string (Constructive_lowering.Lowering.Event.kind event)));
        ( "source-member-node-ids",
          `List
            (Constructive_lowering.Lowering.Event.source_member_node_ids event
            |> List.map node_id_to_json) );
        ( "replacement-node-id",
          node_id_to_json
            (Constructive_lowering.Lowering.Event.replacement_node_id event) );
        ( "replacement-operation",
          `String
            (Ir.Op.to_string
               (Ir.node_op
                  (Constructive_lowering.Lowering.Event.replacement_node event))) ) ]

  let uncovered_reason_to_json = function
    | Constructive_lowering.Lowering.Uncovered.Cut legality ->
        `Assoc
          [ ("kind", `String "cut");
            ( "boundaries",
              `List
                (Constructive_lowering.Legality.boundaries legality
                |> List.map boundary_to_json) ) ]
    | Constructive_lowering.Lowering.Uncovered.Unsupported_pattern ->
        `Assoc [ ("kind", `String "unsupported-pattern") ]
    | Constructive_lowering.Lowering.Uncovered.Overlap node ->
        `Assoc [ ("kind", `String "overlap"); ("node", node_id_to_json node) ]

  let uncovered_to_json uncovered =
    `Assoc
      [ ("kind", `String (kind_to_string (Constructive_lowering.Lowering.Uncovered.kind uncovered)));
        ("source", node_id_to_json (Constructive_lowering.Lowering.Uncovered.source uncovered));
        ( "destination",
          node_id_to_json
            (Constructive_lowering.Lowering.Uncovered.destination uncovered) );
        ( "source-member-node-ids",
          `List
            (Constructive_lowering.Lowering.Uncovered.source_member_node_ids uncovered
            |> List.map node_id_to_json) );
        ( "reason",
          uncovered_reason_to_json
            (Constructive_lowering.Lowering.Uncovered.reason uncovered) ) ]

  let legacy_matcher_to_string = function
    | Legacy_matcher.Quant_linear_add -> "fuse_quant_linear_add"
    | Legacy_matcher.W4a16_linear_add -> "fuse_w4a16_linear_add"
    | Legacy_matcher.Quant_gated_linear -> "fuse_quant_gated_linear"

  let queue_kind_to_string = function
    | Queue_stage.Sequential -> "sequential"
    | Queue_stage.Concurrent -> "concurrent"

  let queue_stage_to_json stage =
    `Assoc
      [ ("index", `Int (Queue_stage.index stage));
        ("kind", `String (queue_kind_to_string (Queue_stage.kind stage)));
        ( "commands",
          `List
            (Queue_stage.requests stage
            |> List.map (fun request ->
                   `Assoc
                     [ ("node-id", node_id_to_json (Compute_plan.Request.id request));
                       ( "operation",
                         `String
                           (Ir.Op.to_string (Compute_plan.Request.op request)) ) ])) ) ]

  let dependency_to_json dependency =
    `Assoc
      [ ("source", node_id_to_json (Dependency.source dependency));
        ( "destination",
          node_id_to_json (Dependency.destination dependency) );
        ( "relation",
          `String (relation_to_string (Dependency.relation dependency)) ) ]

  let barrier_kind_to_string = function
    | Compiler_barrier.Create -> "create"
    | Compiler_barrier.Arrive -> "arrive"
    | Compiler_barrier.Wait -> "wait"

  let barrier_to_json barrier =
    `Assoc
      [ ("node-id", `Int (Compiler_barrier.node_id barrier));
        ("id", `Int (Compiler_barrier.id barrier));
        ( "kind",
          `String (barrier_kind_to_string (Compiler_barrier.kind barrier)) ) ]

  let storage_to_string = function
    | Ir.Linear_storage.Dense dtype ->
        "dense:" ^ Ir.Dtype.to_string dtype
    | Ir.Linear_storage.Block_quantized quant ->
        "block-quantized:" ^ Ir.Dtype.quant_to_string quant
    | Ir.Linear_storage.Groupwise_packed { bits; group_elements } ->
        Printf.sprintf "groupwise-packed:%d-bit-x%d" bits group_elements

  let backend_to_string = function
    | Block_schema.Target_tactic.Portable -> "portable"
    | Block_schema.Target_tactic.Metal -> "metal"
    | Block_schema.Target_tactic.Cuda -> "cuda"
    | Block_schema.Target_tactic.Hip -> "hip"

  let execution_to_string = function
    | Block_schema.Target_tactic.Scalar -> "scalar"
    | Block_schema.Target_tactic.Simd width -> Printf.sprintf "simd%d" width
    | Block_schema.Target_tactic.Warp width -> Printf.sprintf "warp%d" width
    | Block_schema.Target_tactic.Wave width -> Printf.sprintf "wave%d" width

  let decode_to_string = function
    | Block_schema.Target_tactic.Direct_integer -> "direct-integer"
    | Block_schema.Target_tactic.Packed_nibble -> "packed-nibble"
    | Block_schema.Target_tactic.Packed_bitfield -> "packed-bitfield"
    | Block_schema.Target_tactic.Codebook -> "codebook"

  let target_tactic_to_json target_tactic =
    `Assoc
      [ ("name", `String (Block_schema.Target_tactic.name target_tactic));
        ( "backend",
          `String
            (backend_to_string
               (Block_schema.Target_tactic.backend target_tactic)) );
        ( "execution",
          `String
            (execution_to_string
               (Block_schema.Target_tactic.execution target_tactic)) );
        ( "vector-width",
          `Int (Block_schema.Target_tactic.vector_width target_tactic) );
        ( "decode",
          `String
            (decode_to_string (Block_schema.Target_tactic.decode target_tactic)) );
        ( "primitives",
          `List
            (Block_schema.Target_tactic.primitives target_tactic
            |> List.map (fun primitive ->
                   `String
                     (Block_schema.Target_tactic.Primitive.to_string primitive))) ) ]

  let resource_to_json resource =
    `Assoc
      [ ( "threadgroup-threads",
          `Int
            (Kernel_abi.Linear_tactic.Candidate.Resource.threadgroup_threads
               resource) );
        ( "simdgroups-per-threadgroup",
          `Int
            (Kernel_abi.Linear_tactic.Candidate.Resource.simdgroups_per_threadgroup
               resource) );
        ( "output-values-per-threadgroup",
          `Int
            (Kernel_abi.Linear_tactic.Candidate.Resource.output_values_per_threadgroup
               resource) );
        ( "decoded-weight-reuse",
          `Int
            (Kernel_abi.Linear_tactic.Candidate.Resource.decoded_weight_reuse
               resource) );
        ( "activation-reuse",
          `Int
            (Kernel_abi.Linear_tactic.Candidate.Resource.activation_reuse resource) );
        ( "split-k-partitions",
          `Int
            (Kernel_abi.Linear_tactic.Candidate.Resource.split_k_partitions resource) );
        ( "threadgroup-sram-bytes",
          `Int
            (Kernel_abi.Linear_tactic.Candidate.Resource.threadgroup_sram_bytes
               resource) );
        ( "dispatches",
          `Int
            (Kernel_abi.Linear_tactic.Candidate.Resource.dispatches resource) ) ]

  let candidate_cost_to_json cost =
    let float name value = name, `Float value in
    `Assoc
      [ float "flops" (Kernel_abi.Linear_tactic.Candidate.Cost.flops cost);
        float "physical-storage-bytes"
          (Kernel_abi.Linear_tactic.Candidate.Cost.physical_storage_bytes cost);
        float "input-bytes"
          (Kernel_abi.Linear_tactic.Candidate.Cost.input_bytes cost);
        float "output-bytes"
          (Kernel_abi.Linear_tactic.Candidate.Cost.output_bytes cost);
        float "weight-traffic-bytes"
          (Kernel_abi.Linear_tactic.Candidate.Cost.weight_traffic_bytes cost);
        float "activation-traffic-bytes"
          (Kernel_abi.Linear_tactic.Candidate.Cost.activation_traffic_bytes cost);
        float "memory-bytes"
          (Kernel_abi.Linear_tactic.Candidate.Cost.memory_bytes cost);
        ( "workgroups",
          `Int (Kernel_abi.Linear_tactic.Candidate.Cost.workgroups cost) );
        float "core-utilization"
          (Kernel_abi.Linear_tactic.Candidate.Cost.core_utilization cost);
        ( "dispatches",
          `Int (Kernel_abi.Linear_tactic.Candidate.Cost.dispatches cost) );
        float "compute-seconds"
          (Kernel_abi.Linear_tactic.Candidate.Cost.compute_seconds cost);
        float "memory-seconds"
          (Kernel_abi.Linear_tactic.Candidate.Cost.memory_seconds cost);
        float "roofline-seconds"
          (Kernel_abi.Linear_tactic.Candidate.Cost.roofline_seconds cost);
        float "dispatch-seconds"
          (Kernel_abi.Linear_tactic.Candidate.Cost.dispatch_seconds cost);
        float "total-seconds"
          (Kernel_abi.Linear_tactic.Candidate.Cost.total_seconds cost) ]

  let candidate_to_json candidate =
    let tactic = Kernel_abi.Linear_tactic.Candidate.tactic candidate in
    let schema = Kernel_abi.Linear_tactic.Candidate.schema candidate in
    let target_tactic =
      Kernel_abi.Linear_tactic.Candidate.target_tactic candidate
    in
    let legality = Kernel_abi.Linear_tactic.Candidate.legality candidate in
    let resource = Kernel_abi.Linear_tactic.Candidate.Legality.resource legality in
    `Assoc
      [ ( "name",
          `String (Kernel_abi.Linear_tactic.name tactic) );
        ( "threadgroup",
          `List
            (let x, y, z = Kernel_abi.Linear_tactic.threadgroup tactic in
             [ `Int x; `Int y; `Int z ]) );
        ("schema", `String (Block_schema.to_string schema));
        ("storage", `String (storage_to_string (Block_schema.storage schema)));
        ("target-tactic", target_tactic_to_json target_tactic);
        ( "legality",
          `Assoc
            [ ("simd-width", `Int (Kernel_abi.Linear_tactic.Candidate.Legality.simd_width legality));
              ( "target-simd-width",
                `Int
                  (Kernel_abi.Linear_tactic.Candidate.Legality.target_simd_width
                     legality) );
              ( "target-max-threads-per-threadgroup",
                `Int
                  (Kernel_abi.Linear_tactic.Candidate.Legality.target_max_threads_per_threadgroup
                     legality) );
              ( "target-sram-capacity-bytes",
                `Int
                  (Kernel_abi.Linear_tactic.Candidate.Legality.target_sram_capacity_bytes
                     legality) );
              ("resource", resource_to_json resource) ] );
        ( "cost",
          candidate_cost_to_json
            (Kernel_abi.Linear_tactic.Candidate.cost candidate) ) ]

  let linear_selection_to_json selection =
    `Assoc
      [ ( "node-id",
          node_id_to_json (Linear_selection.node_id selection) );
        ( "operation",
          `String (Ir.Op.to_string (Linear_selection.operation selection)) );
        ( "chosen",
          candidate_to_json (Linear_selection.chosen selection) );
        ( "alternatives",
          `List
            (Linear_selection.alternatives selection
            |> List.map candidate_to_json) ) ]

  let kernel_to_json entry =
    let x, y, z = Kernel_abi.Entry.threadgroup entry in
    `Assoc
      [ ("name", `String (Kernel_abi.Entry.name entry));
        ( "operation",
          `String
            (Kernel_abi.Operation.to_string (Kernel_abi.Entry.operation entry)) );
        ( "input-dtype",
          `String (Ir.Dtype.to_string (Kernel_abi.Entry.input_dtype entry)) );
        ( "output-dtype",
          `String (Ir.Dtype.to_string (Kernel_abi.Entry.output_dtype entry)) );
        ("threadgroup", `List [ `Int x; `Int y; `Int z ]);
        ( "tile",
          match Kernel_abi.Entry.tile entry with
          | None -> `Null
          | Some (m, n, k) -> `List [ `Int m; `Int n; `Int k ] ) ]

  let to_yojson receipt =
    let legacy_counts =
      receipt.legacy_counts
      |> List.map (fun (matcher, invocations, rewrites) ->
             `Assoc
               [ ("matcher", `String (legacy_matcher_to_string matcher));
                 ("invocations", `Int invocations);
                 ("rewrites", `Int rewrites) ])
    in
    `Assoc
      [ ("schema-version", `Int 2);
        ("target", Target_hardware.to_json receipt.target);
        ( "fusion",
          `Assoc
            [ ( "decisions",
                `List (List.map decision_to_json receipt.decisions) );
              ( "events",
                `List (List.map event_to_json receipt.events) );
              ( "uncovered",
                `List (List.map uncovered_to_json receipt.uncovered) ) ] );
        ( "lowering",
          `Assoc
            [ ("unified-count", `Int receipt.unified_count);
              ("accepted-count", `Int receipt.accepted_count);
              ("cut-count", `Int receipt.cut_count);
              ("legacy-count", `Int receipt.legacy_rewrite_count);
              ( "legacy-invocation-count",
                `Int receipt.legacy_invocation_count );
              ("legacy-rewrite-count", `Int receipt.legacy_rewrite_count);
              ("legacy-counts", `List legacy_counts) ] );
        ( "linear-tactics",
          `List (List.map linear_selection_to_json receipt.linear_selections) );
        ( "kernels",
          `List (List.map kernel_to_json receipt.kernels) );
        ( "aot-queue",
          `Assoc
            [ ( "stages",
                `List (List.map queue_stage_to_json receipt.queue_stages) );
              ( "dependencies",
                `List (List.map dependency_to_json receipt.dependencies) );
              ( "barriers",
                `List (List.map barrier_to_json receipt.barriers) ) ] ) ]

  let to_json_string receipt = Yojson.Basic.to_string (to_yojson receipt) ^ "\n"

  let write_file path receipt =
    try
      let channel = open_out_bin path in
      Fun.protect
        ~finally:(fun () -> close_out_noerr channel)
        (fun () -> output_string channel (to_json_string receipt));
      Ok ()
    with Sys_error message -> Error ("could not write optimization receipt: " ^ message)

  let dependencies_of_plan plan =
    let module Pair_set = Set.Make (struct
      type t = int * int

      let compare = compare
    end) in
    let add_request (seen, dependencies) request =
      let destination = Compute_plan.Request.id request in
      List.fold_left
        (fun (seen, dependencies) dependency ->
          let source = Compute_plan.Dependency.node dependency in
          let key =
            Compute_plan.Node_id.to_int source,
            Compute_plan.Node_id.to_int destination
          in
          if Pair_set.mem key seen then seen, dependencies
          else
            match
              Compute_plan.edge plan ~source ~destination
            with
            | None -> seen, dependencies
            | Some edge ->
                ( Pair_set.add key seen,
                  { Dependency.source = source;
                    destination;
                    relation = Compute_plan.Edge.relation edge }
                  :: dependencies ))
        (seen, dependencies)
        (Compute_plan.Request.dependencies request)
    in
    let _, dependencies =
      List.fold_left add_request (Pair_set.empty, [])
        (Compute_plan.requests plan)
    in
    List.sort
      (fun left right ->
        match
          Compute_plan.Node_id.compare (Dependency.source left)
            (Dependency.source right)
        with
        | 0 ->
            Compute_plan.Node_id.compare (Dependency.destination left)
              (Dependency.destination right)
        | order -> order)
      dependencies

  let barriers_of_graph graph =
    Ir.Graph.nodes graph
    |> List.filter_map (fun node ->
           let node_id = Ir.node_id node in
           match Ir.node_op node with
           | Ir.Op.Barrier_create { id; _ } ->
               Some { Compiler_barrier.node_id; id; kind = Create }
           | Ir.Op.Barrier_arrive id ->
               Some { Compiler_barrier.node_id; id; kind = Arrive }
           | Ir.Op.Barrier_wait id ->
               Some { Compiler_barrier.node_id; id; kind = Wait }
           | _ -> None)

  let selection_for_request target request =
    let inputs = Compute_plan.Request.inputs request in
    let output =
      match Compute_plan.Request.outputs request with
      | output :: _ -> Some output
      | [] -> None
    in
    let select ~m ~n ~k ~input ~storage ~output =
      match
        Kernel_abi.Linear_tactic.select_costed ~target ~m ~n ~k
          ~input_dtype:(Ir.Value.dtype input) ~storage
          ~output_dtype:(Ir.Value.dtype output)
      with
      | None -> None
      | Some selection ->
          Some
            { Linear_selection.node_id = Compute_plan.Request.id request;
              operation = Compute_plan.Request.op request;
              chosen = Kernel_abi.Linear_tactic.Selection.chosen selection;
              alternatives =
                Kernel_abi.Linear_tactic.Selection.alternatives selection }
    in
    match Compute_plan.Request.op request, inputs, output with
    | Ir.Op.Linear { m; n; k; bias }, input :: weight :: parameters, Some output ->
        (match
           Ir.Linear_storage.classify ~has_bias:bias ~weight ~parameters
         with
        | Ok storage ->
            select ~m ~n ~k ~input ~storage:storage.layout ~output
        | Error _ -> None)
    | Ir.Op.Linear_add { m; n; k }, input :: weight :: _, Some output ->
        (match Ir.Value.dtype weight with
        | Ir.Dtype.Quant quant ->
            select ~m ~n ~k ~input
              ~storage:(Ir.Linear_storage.Block_quantized quant) ~output
        | _ -> None)
    | _ -> None

  let linear_selections_of_plan target plan =
    Compute_plan.requests plan
    |> List.filter_map (selection_for_request target)

  let make ~target ~analysis ~lowering ~legacy_counts ~execution_plan
      ~execution_graph ~kernels =
    let events = Constructive_lowering.Lowering.events lowering in
    let decisions = Constructive_lowering.Analysis.decisions analysis in
    let accepted_count, cut_count =
      List.fold_left
        (fun (accepted, cut) decision ->
          match decision with
          | Constructive_lowering.Decision.Accepted _ -> accepted + 1, cut
          | Constructive_lowering.Decision.Cut _ -> accepted, cut + 1)
        (0, 0) decisions
    in
    let all_legacy_matchers =
      [ Legacy_matcher.Quant_linear_add;
        Legacy_matcher.W4a16_linear_add;
        Legacy_matcher.Quant_gated_linear ]
    in
    let legacy_counts =
      List.map
        (fun matcher ->
          let invocations, rewrites =
            legacy_counts
            |> List.find_opt (fun (candidate, _, _) -> candidate = matcher)
            |> function
            | Some (_, invocations, rewrites) -> invocations, rewrites
            | None -> 0, 0
          in
          matcher, invocations, rewrites)
        all_legacy_matchers
    in
    let legacy_invocation_count =
      List.fold_left
        (fun count (_, invocations, _) -> count + invocations)
        0 legacy_counts
    in
    let legacy_rewrite_count =
      List.fold_left (fun count (_, _, rewrites) -> count + rewrites) 0
        legacy_counts
    in
    { target;
      decisions;
      events;
      uncovered = Constructive_lowering.Lowering.uncovered lowering;
      unified_count = List.length events;
      accepted_count;
      cut_count;
      legacy_invocation_count;
      legacy_rewrite_count;
      legacy_counts;
      linear_selections = linear_selections_of_plan target execution_plan;
      kernels;
      queue_stages =
        Compute_plan.stages execution_plan
        |> List.map Queue_stage.of_compute_stage;
      dependencies = dependencies_of_plan execution_plan;
      barriers = barriers_of_graph execution_graph }
end

(* [default_pipeline] remains the public convenience pipeline.  Optimization
   needs an analysis snapshot before the covered Linear/epilogue matchers, so
   it uses this private semantic prefix and lowers those families from the
   resulting immutable plan. *)
let optimize_pre_analysis_pipeline =
  Pass.Pipeline.of_list [ Pass_fuse_linear_bias.pass ]

let execution_passes =
  [
    Pass_fuse_short_conv_step.pass;
    Pass_fuse_lm_head_argmax.pass;
  ]

let execution_pipeline = Pass.Pipeline.of_list execution_passes

module Optimization = struct
  module Legacy_fallback = struct
    type matcher = Quant_linear_add | W4a16_linear_add | Quant_gated_linear

    type t = { matcher : matcher; invocations : int; rewrites : int }

    let matcher fallback = fallback.matcher
    let invocations fallback = fallback.invocations
    let rewrites fallback = fallback.rewrites

    let name = function
      | Quant_linear_add -> "fuse_quant_linear_add"
      | W4a16_linear_add -> "fuse_w4a16_linear_add"
      | Quant_gated_linear -> "fuse_quant_gated_linear"
  end

  type t = {
    target : Target_hardware.t;
    semantic_graph : Ir.Graph.t;
    plan : Compute_plan.t;
    fusion_regions : Kernel_ir.t list;
    region_analysis : Constructive_lowering.Analysis.t;
    lowering : Constructive_lowering.Lowering.t;
    legacy_fallbacks : Legacy_fallback.t list;
    scan_regions : Kernel_ir.Scan.t list;
    execution_plan : Compute_plan.t;
    execution_graph : Ir.Graph.t;
  }

  let target optimization = optimization.target
  let semantic_graph optimization = optimization.semantic_graph
  let plan optimization = optimization.plan
  let fusion_regions optimization = optimization.fusion_regions
  let region_analysis optimization = optimization.region_analysis
  let lowering optimization = optimization.lowering
  let legacy_fallbacks optimization = optimization.legacy_fallbacks
  let scan_regions optimization = optimization.scan_regions
  let execution_plan optimization = optimization.execution_plan
  let execution_graph optimization = optimization.execution_graph

  let receipt_legacy_matcher = function
    | Legacy_fallback.Quant_linear_add -> Receipt.Legacy_matcher.Quant_linear_add
    | Legacy_fallback.W4a16_linear_add ->
        Receipt.Legacy_matcher.W4a16_linear_add
    | Legacy_fallback.Quant_gated_linear ->
        Receipt.Legacy_matcher.Quant_gated_linear

  let receipt ~kernels optimization =
    let legacy_counts =
      optimization.legacy_fallbacks
      |> List.map (fun fallback ->
             ( receipt_legacy_matcher
                 (Legacy_fallback.matcher fallback),
               Legacy_fallback.invocations fallback,
               Legacy_fallback.rewrites fallback ))
    in
    Receipt.make ~target:optimization.target
      ~analysis:optimization.region_analysis ~lowering:optimization.lowering
      ~legacy_counts ~execution_plan:optimization.execution_plan
      ~execution_graph:optimization.execution_graph ~kernels
end

module Optimization_node_id_set = Set.Make (Int)
module Optimization_node_id_map = Map.Make (Int)

let fallback_matcher = function
  | Constructive_lowering.Lowering.Kind.Linear_add ->
      Some Optimization.Legacy_fallback.Quant_linear_add
  | Constructive_lowering.Lowering.Kind.W4a16_linear_add ->
      Some Optimization.Legacy_fallback.W4a16_linear_add
  | Constructive_lowering.Lowering.Kind.Gated_linear ->
      Some Optimization.Legacy_fallback.Quant_gated_linear
  | Constructive_lowering.Lowering.Kind.Rms_norm
  | Constructive_lowering.Lowering.Kind.Rms_rope
  | Constructive_lowering.Lowering.Kind.Short_conv_silu
  | Constructive_lowering.Lowering.Kind.Gated_delta -> None

let fallback_matcher_equal left right =
  match left, right with
  | ( Optimization.Legacy_fallback.Quant_linear_add,
      Optimization.Legacy_fallback.Quant_linear_add )
  | ( Optimization.Legacy_fallback.W4a16_linear_add,
      Optimization.Legacy_fallback.W4a16_linear_add )
  | ( Optimization.Legacy_fallback.Quant_gated_linear,
      Optimization.Legacy_fallback.Quant_gated_linear ) -> true
  | _ -> false

let legacy_fallback_kinds lowering =
  Constructive_lowering.Lowering.uncovered lowering
  |> List.fold_left
       (fun matchers uncovered ->
         let matcher =
           fallback_matcher
             (Constructive_lowering.Lowering.Uncovered.kind uncovered)
         in
         match matcher with
         | None -> matchers
         | Some matcher ->
             if List.exists (fallback_matcher_equal matcher) matchers then matchers
             else matchers @ [ matcher ])
       []

let invoke_legacy_fallback matcher graph =
  match matcher with
  | Optimization.Legacy_fallback.Quant_linear_add ->
      Pass_fuse_linear_bias.fuse_quant_linear_add graph
  | Optimization.Legacy_fallback.W4a16_linear_add ->
      Pass_fuse_linear_bias.fuse_w4a16_linear_add graph
  | Optimization.Legacy_fallback.Quant_gated_linear ->
      Pass_fuse_linear_bias.fuse_quant_gated_linear graph

let legacy_rewrite_count matcher graph =
  Ir.Graph.nodes graph
  |> List.fold_left
       (fun count node ->
         let matched =
           match matcher, Ir.node_op node with
           | Optimization.Legacy_fallback.Quant_linear_add,
             Ir.Op.Linear_add _ -> true
           | Optimization.Legacy_fallback.W4a16_linear_add,
             Ir.Op.W4a16_linear { bias = true; _ } -> true
           | Optimization.Legacy_fallback.Quant_gated_linear,
             Ir.Op.Gated_linear _ -> true
           | _ -> false
         in
         if matched then count + 1 else count)
       0

let apply_legacy_fallbacks graph lowering =
  legacy_fallback_kinds lowering
  |> List.fold_left
       (fun (graph, fallbacks) matcher ->
         let before = legacy_rewrite_count matcher graph in
         let rewritten = invoke_legacy_fallback matcher graph in
         let after = legacy_rewrite_count matcher rewritten in
         ( rewritten,
           { Optimization.Legacy_fallback.matcher;
             invocations = 1;
             rewrites = max 0 (after - before) }
           :: fallbacks ))
       (graph, [])
  |> fun (graph, fallbacks) -> graph, List.rev fallbacks

let protected_node_ids lowering =
  let add ids node_id = Optimization_node_id_set.add node_id ids in
  let event_ids =
    Constructive_lowering.Lowering.events lowering
    |> List.fold_left
         (fun ids event ->
           List.fold_left add ids
             (Constructive_lowering.Lowering.Event.source_member_node_ids event))
         Optimization_node_id_set.empty
  in
  Constructive_lowering.Lowering.uncovered lowering
  |> List.fold_left
       (fun ids uncovered ->
         let ids =
           add ids (Constructive_lowering.Lowering.Uncovered.source uncovered)
         in
         add ids (Constructive_lowering.Lowering.Uncovered.destination uncovered))
       event_ids

(* The generic epilogue pass still owns unrelated Matmul and dense epilogue
   families.  It receives a view with covered endpoints hidden, so it cannot
   rediscover a covered region or rewrite an explicitly uncovered candidate. *)
let run_uncovered_epilogues graph lowering =
  let protected = protected_node_ids lowering in
  let visible_nodes =
    Ir.Graph.nodes graph
    |> List.filter (fun node ->
           not (Optimization_node_id_set.mem (Ir.node_id node) protected))
  in
  let transformed = Pass_fuse_epilogues.run (Ir.Graph.with_nodes graph visible_nodes) in
  let transformed_by_id =
    Ir.Graph.nodes transformed
    |> List.fold_left
         (fun nodes node ->
           Optimization_node_id_map.add (Ir.node_id node) node nodes)
         Optimization_node_id_map.empty
  in
  Ir.Graph.with_nodes graph
    (Ir.Graph.nodes graph
    |> List.filter_map (fun original ->
           if Optimization_node_id_set.mem (Ir.node_id original) protected then
             Some original
           else
             Optimization_node_id_map.find_opt (Ir.node_id original) transformed_by_id))

let optimize ?(target = Target_hardware.default) graph =
  let analysis_graph = Pass.Pipeline.run optimize_pre_analysis_pipeline graph in
  let* analysis_plan = Compute_plan.of_graph analysis_graph in
  let region_analysis = Constructive_lowering.analyze ~target analysis_plan in
  let lowering = Constructive_lowering.lower region_analysis in
  let* fusion_regions = Constructive_lowering.executable_regions lowering in
  let lowered_graph = Constructive_lowering.Lowering.graph lowering in
  let lowered_graph, legacy_fallbacks =
    apply_legacy_fallbacks lowered_graph lowering
  in
  let scan_regions = Kernel_ir.Scan.recover analysis_graph in
  let max_scan_width =
    let capacity = target.memory.sram_capacity_bytes / 4 in
    let rec fit width =
      let next = width + 1 in
      if
        next <= 64
        && next <= target.execution.max_threads_per_threadgroup
        && (next * next) + next <= capacity
      then fit next
      else width
    in
    fit 0
  in
  let scan_lowered_graph =
    Kernel_ir.Scan.fuse_triangular_recurrences ~max_width:max_scan_width
      lowered_graph
  in
  let fused_graph = run_uncovered_epilogues scan_lowered_graph lowering in
  let final_semantic_graph =
    Pass_fuse_linear_bias.canonicalize_layouts fused_graph
    |> Pass_fuse_linear_bias.fuse_w4a16_qkv
    |> Pass_fuse_linear_bias.fuse_quant_qkv
  in
  let* semantic_plan = Compute_plan.of_graph final_semantic_graph in
  let execution_input = Pass.Pipeline.run execution_pipeline final_semantic_graph in
  let* execution_plan = Compute_plan.of_graph execution_input in
  let execution_graph = Pass_co_schedule.run_plan execution_plan in
  Ok
    {
      Optimization.target = target;
      Optimization.semantic_graph = final_semantic_graph;
      plan = semantic_plan;
      fusion_regions;
      region_analysis;
      lowering;
      legacy_fallbacks;
      scan_regions;
      execution_plan;
      execution_graph;
    }
