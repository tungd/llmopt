(** Metal kernel-tactic selection: the [Tactic] record with its linear and
   attention selection registries.  Extracted from metal.ml (LOOP items 5-6
   prerequisite); the Metal facade aliases it. *)

module Tactic = struct
  type t = { name : string; threadgroup : int * int * int }

  type attention_problem = {
    target : Target_hardware.t;
    head_dimension : int;
    input_dtype : Ir.Dtype.t;
    output_dtype : Ir.Dtype.t;
  }

  type attention_registration = {
    build : attention_problem -> t;
    accepts : attention_problem -> bool;
  }

  let name tactic = tactic.name
  let threadgroup tactic = tactic.threadgroup

  let supports_simd target ~threads =
    target.Target_hardware.memory.simd_lanes = 32
    && target.execution.max_threads_per_threadgroup >= threads

  let f16_io input_dtype output_dtype =
    input_dtype = Ir.Dtype.Float16 && output_dtype = Ir.Dtype.Float16

  let select_linear ~target ~m ~n ~k ~input_dtype ~storage ~output_dtype =
    Kernel_abi.Linear_tactic.select_costed ~target ~m ~n ~k ~input_dtype
      ~storage ~output_dtype
    |> Option.map (fun selection ->
           let candidate =
             Kernel_abi.Linear_tactic.Selection.chosen selection
           in
           let tactic =
             Kernel_abi.Linear_tactic.Candidate.tactic candidate
           in
           { name = Kernel_abi.Linear_tactic.name tactic;
             threadgroup = Kernel_abi.Linear_tactic.threadgroup tactic })

  let attention_registry : attention_registration list =
    [ { build =
          (fun (problem : attention_problem) ->
            { name =
                Printf.sprintf "llmopt_attention_f16_simd_h%d"
                  problem.head_dimension;
              threadgroup = (256, 1, 1) });
        accepts =
          (fun (problem : attention_problem) ->
            problem.head_dimension > 0 && problem.head_dimension mod 32 = 0
            && f16_io problem.input_dtype problem.output_dtype
            && supports_simd problem.target ~threads:256) };
      { build =
          (fun _ ->
            { name = "llmopt_attention_f16"; threadgroup = (64, 1, 1) });
        accepts =
          (fun (problem : attention_problem) ->
            problem.head_dimension > 0
            && f16_io problem.input_dtype problem.output_dtype
            && problem.target.execution.max_threads_per_threadgroup >= 64) } ]

  let select_attention ~target ~head_dimension ~input_dtype ~output_dtype =
    let problem = { target; head_dimension; input_dtype; output_dtype } in
    attention_registry
    |> List.find_map (fun (registration : attention_registration) ->
           if registration.accepts problem then Some (registration.build problem)
           else None)
end
