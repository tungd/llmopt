let ( let* ) = Result.bind

module Speculative_acceptance = struct
  type t = {
    emitted_tokens : int array;
    accepted_draft_tokens : int;
  }

  let greedy ~draft_tokens ~target_predictions =
    let draft_count = Array.length draft_tokens in
    if Array.length target_predictions <> draft_count + 1 then
      Error
        (Printf.sprintf
           "speculative verification requires %d target predictions for %d draft tokens"
           (draft_count + 1) draft_count)
    else
      let rec first_mismatch index =
        if index = draft_count then index
        else if draft_tokens.(index) = target_predictions.(index) then
          first_mismatch (index + 1)
        else index
      in
      let accepted = first_mismatch 0 in
      let emitted = Array.make (accepted + 1) target_predictions.(accepted) in
      if accepted > 0 then Array.blit draft_tokens 0 emitted 0 accepted;
      Ok { emitted_tokens = emitted; accepted_draft_tokens = accepted }

  let emitted_tokens result = Array.copy result.emitted_tokens
  let accepted_draft_tokens result = result.accepted_draft_tokens
end

module Target_coupled_mtp = struct
  type t = {
    proposed_tokens : int array;
    emitted_tokens : int array;
    accepted_draft_tokens : int;
  }

  let run ~max_draft_tokens ~state ~propose ~verify ~commit =
    if max_draft_tokens <= 0 then
      Error "target-coupled MTP max_draft_tokens must be positive"
    else
      let* proposed_tokens = propose state in
      let proposal_count = Array.length proposed_tokens in
      if proposal_count = 0 then
        Error "target-coupled MTP assistant produced no draft tokens"
      else if proposal_count > max_draft_tokens then
        Error
          (Printf.sprintf
             "target-coupled MTP assistant produced %d tokens; maximum is %d"
             proposal_count max_draft_tokens)
      else
        let* target_predictions, verified_state = verify state proposed_tokens in
        let* acceptance =
          Speculative_acceptance.greedy ~draft_tokens:proposed_tokens
            ~target_predictions
        in
        let accepted_draft_tokens =
          Speculative_acceptance.accepted_draft_tokens acceptance
        in
        let* state =
          commit verified_state ~accepted_draft_tokens
        in
        Ok
          ( {
              proposed_tokens = Array.copy proposed_tokens;
              emitted_tokens =
                Speculative_acceptance.emitted_tokens acceptance;
              accepted_draft_tokens;
            },
            state )

  let proposed_tokens result = Array.copy result.proposed_tokens
  let emitted_tokens result = Array.copy result.emitted_tokens
  let accepted_draft_tokens result = result.accepted_draft_tokens
end

module Medusa = struct
  type tree = {
    tokens : int array;
    masks : int array;
    depths : int array;
    parents : int array;
  }

  type t = {
    tree_tokens : int array;
    emitted_tokens : int array;
    accepted_depth : int;
  }

  let create_linear_tree ~head_predictions =
    let count = Array.length head_predictions in
    let tokens = Array.copy head_predictions in
    let depths = Array.init count (fun i -> i + 1) in
    let parents = Array.init count (fun i -> if i = 0 then -1 else i - 1) in
    let masks =
      Array.init 8 (fun i ->
          if i < count then
            (1 lsl (i + 1)) - 1
          else 0)
    in
    { tokens; masks; depths; parents }

  let create_tree ~nodes ~parents ~depths =
    let count = Array.length nodes in
    let tokens = Array.copy nodes in
    let masks =
      Array.init 8 (fun i ->
          if i >= count then 0
          else
            let rec ancestors node_idx acc =
              if node_idx < 0 || node_idx >= count then acc
              else
                let acc' = acc lor (1 lsl node_idx) in
                let parent = parents.(node_idx) in
                if parent >= 0 && parent < node_idx then
                  ancestors parent acc'
                else acc'
            in
            ancestors i (1 lsl i))
    in
    { tokens; masks; depths; parents }

  let tree_masks tree = Array.copy tree.masks
  let tree_tokens_array tree = Array.copy tree.tokens

  let accept_tree ~tree ~target_predictions =
    let count = Array.length tree.tokens in
    if Array.length target_predictions < count + 1 then
      Error
        (Printf.sprintf
           "Medusa verification requires at least %d target predictions for tree of size %d"
           (count + 1) count)
    else
      let rec match_linear idx =
        if idx = count then count
        else if tree.tokens.(idx) = target_predictions.(idx) then
          match_linear (idx + 1)
        else idx
      in
      let accepted_count = match_linear 0 in
      let emitted = Array.make (accepted_count + 1) target_predictions.(accepted_count) in
      if accepted_count > 0 then Array.blit tree.tokens 0 emitted 0 accepted_count;
      Ok
        {
          tree_tokens = Array.copy tree.tokens;
          emitted_tokens = emitted;
          accepted_depth = accepted_count;
        }

  let run ~tree ~state ~verify ~commit =
    let* target_predictions, verified_state = verify state tree in
    let* acceptance = accept_tree ~tree ~target_predictions in
    let* state = commit verified_state ~accepted_depth:acceptance.accepted_depth in
    Ok (acceptance, state)

  let tree_tokens result = Array.copy result.tree_tokens
  let emitted_tokens result = Array.copy result.emitted_tokens
  let accepted_depth result = result.accepted_depth
end

module Prompt_lookup = struct
  let find_candidate_ngram ?(min_ngram = 1) ?(max_ngram = 3) ~max_draft_tokens ~tokens () =
    let total = Array.length tokens in
    if total <= min_ngram || max_draft_tokens <= 0 then [||]
    else
      let max_n = min max_ngram (total - 1) in
      let rec try_ngram_len n =
        if n < min_ngram then [||]
        else
          let suffix_start = total - n in
          let rec scan_history pos best =
            if pos < 0 then
              if Array.length best = 0 then try_ngram_len (n - 1) else best
            else
              let rec check_match i =
                if i = n then true
                else if tokens.(pos + i) = tokens.(suffix_start + i) then
                  check_match (i + 1)
                else false
              in
              if check_match 0 then
                let candidate_start = pos + n in
                let available = total - candidate_start in
                let draft_count = min max_draft_tokens available in
                let best =
                  if draft_count > Array.length best then
                    Array.sub tokens candidate_start draft_count
                  else best
                in
                scan_history (pos - 1) best
              else scan_history (pos - 1) best
          in
          scan_history (suffix_start - 1) [||]
      in
      try_ngram_len max_n

  let propose_linear ?min_ngram ?max_ngram ~max_draft_tokens ~tokens () =
    find_candidate_ngram ?min_ngram ?max_ngram ~max_draft_tokens ~tokens ()

  let propose_tree ?min_ngram ?max_ngram ~max_draft_tokens ~tokens () =
    let candidates = find_candidate_ngram ?min_ngram ?max_ngram ~max_draft_tokens ~tokens () in
    if Array.length candidates = 0 then None
    else Some (Medusa.create_linear_tree ~head_predictions:candidates)
end
