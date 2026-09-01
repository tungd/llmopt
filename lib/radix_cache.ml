(*
 * Compressed-edge radix mechanics are adapted from SGLang's RadixCache and
 * MambaRadixCache (Apache-2.0), revision
 * d1af3c89233c475fc1bf11939d86787e6cddd58c. See THIRD_PARTY_NOTICES.md.
 *
 * As in SGLang's hybrid cache, per-token KV values may be split with an edge,
 * while a recurrent checkpoint cannot be synthesized at a split point.
 *)

module Key = struct
  type t = { namespace : string option; tokens : int array }

  let create ?namespace tokens = { namespace; tokens = Array.copy tokens }
  let namespace key = key.namespace
  let tokens key = Array.copy key.tokens
  let length key = Array.length key.tokens
end

module Stats = struct
  type t = {
    cached_tokens : int;
    evictable_tokens : int;
    protected_tokens : int;
    checkpoints : int;
    hits : int;
    misses : int;
    evicted_tokens : int;
  }
end

type child_key = string option * int list

type ('slot, 'checkpoint) node = {
  namespace : string option;
  mutable key : int array;
  mutable value : 'slot array;
  children : (child_key, ('slot, 'checkpoint) node) Hashtbl.t;
  mutable parent : ('slot, 'checkpoint) node option;
  mutable resumable : bool;
  mutable checkpoint : 'checkpoint option;
  mutable lock_ref : int;
  mutable last_access : int;
}

type ('slot, 'checkpoint) t = {
  root : ('slot, 'checkpoint) node;
  page_size : int;
  mutable clock : int;
  mutable cached_tokens : int;
  mutable hits : int;
  mutable misses : int;
  mutable evicted_tokens : int;
}

type ('slot, 'checkpoint) lease = {
  owner : ('slot, 'checkpoint) t;
  node : ('slot, 'checkpoint) node;
  values : 'slot array;
  checkpoint : 'checkpoint option;
  mutable released : bool;
}

type ('slot, 'checkpoint) insert_result = {
  prefix_tokens : int;
  redundant_values : 'slot array;
  redundant_checkpoint : 'checkpoint option;
  canonical_values : 'slot array;
  retained_checkpoint : 'checkpoint option;
}

type ('slot, 'checkpoint) eviction = {
  values : 'slot array;
  checkpoints : 'checkpoint list;
}

let make_node ?parent ?checkpoint ?(resumable = false) ?(namespace = None)
    ?(lock_ref = 0) ~key ~value ~last_access () =
  {
    namespace;
    key;
    value;
    children = Hashtbl.create 4;
    parent;
    resumable;
    checkpoint;
    lock_ref;
    last_access;
  }

let create ~page_size =
  if page_size <= 0 then Error "radix page_size must be positive"
  else
    let root =
      make_node ~key:[||] ~value:[||] ~lock_ref:1 ~last_access:0 ()
    in
    Ok
      {
        root;
        page_size;
        clock = 0;
        cached_tokens = 0;
        hits = 0;
        misses = 0;
        evicted_tokens = 0;
      }

let tick cache =
  cache.clock <- cache.clock + 1;
  cache.clock

let touch cache node = node.last_access <- tick cache

let child_key cache namespace tokens =
  if Array.length tokens = 0 then invalid_arg "empty radix edge has no child key";
  let length = min cache.page_size (Array.length tokens) in
  (namespace, Array.to_list (Array.sub tokens 0 length))

let child_key_at cache namespace tokens offset limit =
  if offset < 0 || offset >= limit || limit > Array.length tokens then
    invalid_arg "empty radix edge has no child key";
  let length = min cache.page_size (limit - offset) in
  let rec collect index acc =
    if index < offset then acc
    else collect (index - 1) (tokens.(index) :: acc)
  in
  (namespace, collect (offset + length - 1) [])

let common_prefix left right =
  let limit = min (Array.length left) (Array.length right) in
  let rec loop index =
    if index = limit || left.(index) <> right.(index) then index
    else loop (index + 1)
  in
  loop 0

let common_prefix_at edge tokens offset limit =
  let length = min (Array.length edge) (limit - offset) in
  let rec loop index =
    if index = length || edge.(index) <> tokens.(offset + index) then index
    else loop (index + 1)
  in
  loop 0

let sub array offset length =
  if length = 0 then [||] else Array.sub array offset length

let aligned cache length = length / cache.page_size * cache.page_size

let split_node cache namespace child split_length =
  if split_length <= 0 || split_length >= Array.length child.key then
    invalid_arg "radix split must be inside an edge";
  let parent =
    match child.parent with
    | Some parent -> parent
    | None -> invalid_arg "cannot split the radix root"
  in
  let original_key = child.key in
  let original_value = child.value in
  let prefix_key = sub original_key 0 split_length in
  let prefix_value = sub original_value 0 split_length in
  let suffix_length = Array.length original_key - split_length in
  child.key <- sub original_key split_length suffix_length;
  child.value <- sub original_value split_length suffix_length;
  let intermediate =
    make_node ~parent ~namespace ~key:prefix_key ~value:prefix_value
      ~lock_ref:child.lock_ref ~last_access:child.last_access ()
  in
  child.parent <- Some intermediate;
  Hashtbl.replace intermediate.children (child_key cache namespace child.key) child;
  Hashtbl.replace parent.children (child_key cache namespace original_key) intermediate;
  touch cache intermediate;
  intermediate

let rec path_values node segments =
  match node.parent with
  | None -> segments
  | Some parent -> path_values parent (node.value :: segments)

let concatenate arrays =
  let length =
    List.fold_left (fun total values -> total + Array.length values) 0 arrays
  in
  let output = Array.make length None in
  let offset = ref 0 in
  List.iter
    (fun values ->
      Array.iter
        (fun value ->
          output.(!offset) <- Some value;
          incr offset)
        values)
    arrays;
  Array.map
    (function Some value -> value | None -> assert false)
    output

let concatenate_path node =
  let rec total_length total current =
    match current.parent with
    | None -> total
    | Some parent ->
        total_length (total + Array.length current.value) parent
  in
  let length = total_length 0 node in
  if length = 0 then [||]
  else
    let output = Array.make length node.value.(0) in
    let rec fill current offset =
      match current.parent with
      | None -> if offset <> 0 then invalid_arg "radix path length mismatch"
      | Some parent ->
          let segment_length = Array.length current.value in
          let segment_offset = offset - segment_length in
          Array.blit current.value 0 output segment_offset segment_length;
          fill parent segment_offset
    in
    fill node length;
    output

let rec adjust_lock delta cache node =
  if node != cache.root then (
    node.lock_ref <- node.lock_ref + delta;
    if node.lock_ref < 0 then invalid_arg "negative radix lock reference";
    match node.parent with
    | None -> invalid_arg "radix node belongs to another tree"
    | Some parent -> adjust_lock delta cache parent)

let match_prefix cache key =
  let tokens = key.Key.tokens in
  let usable_length = aligned cache (Array.length tokens) in
  let namespace = key.Key.namespace in
  let offset_walk =
    Sys.getenv_opt "LLMOPT_EXPERIMENTAL_RADIX_OFFSET_WALK" = Some "1"
  in
  let direct_values =
    Sys.getenv_opt "LLMOPT_EXPERIMENTAL_RADIX_DIRECT_VALUES" = Some "1"
  in
  let access = tick cache in
  cache.root.last_access <- access;
  let rec walk node offset best =
    if offset = usable_length then (node, best)
    else
      let follow child prefix =
          child.last_access <- access;
          if prefix = 0 then (node, best)
          else if prefix < Array.length child.key then
            let intermediate = split_node cache namespace child prefix in
            let best =
              if intermediate.resumable then intermediate else best
            in
            (intermediate, best)
          else
            let best =
              if child.resumable then child else best
            in
            walk child (offset + prefix) best
      in
      if offset_walk then
        match
          Hashtbl.find_opt node.children
            (child_key_at cache namespace tokens offset usable_length)
        with
        | None -> (node, best)
        | Some child ->
            follow child
              (aligned cache
                 (common_prefix_at child.key tokens offset usable_length))
      else
        let remaining = sub tokens offset (usable_length - offset) in
        match
          Hashtbl.find_opt node.children (child_key cache namespace remaining)
        with
        | None -> (node, best)
        | Some child ->
            follow child (aligned cache (common_prefix child.key remaining))
  in
  let _, best = walk cache.root 0 cache.root in
  let values =
    if best == cache.root then [||]
    else if direct_values then concatenate_path best
    else concatenate (path_values best [])
  in
  let checkpoint = best.checkpoint in
  if Array.length values = 0 then cache.misses <- cache.misses + 1
  else cache.hits <- cache.hits + 1;
  adjust_lock 1 cache best;
  { owner = cache; node = best; values; checkpoint; released = false }

let matched_tokens (lease : (_, _) lease) = Array.length lease.values
let matched_values (lease : (_, _) lease) = Array.copy lease.values
let matched_values_view (lease : (_, _) lease) = lease.values
let checkpoint (lease : (_, _) lease) = lease.checkpoint

let release cache lease =
  if lease.owner != cache then Error "radix lease belongs to another cache"
  else if lease.released then Error "radix lease was already released"
  else (
    adjust_lock (-1) cache lease.node;
    lease.released <- true;
    Ok ())

let insert_internal cache ~key ~values ?checkpoint () =
  let tokens = key.Key.tokens in
  let namespace = key.Key.namespace in
  let length = Array.length tokens in
  let offset_insert =
    Sys.getenv_opt "LLMOPT_EXPERIMENTAL_RADIX_OFFSET_INSERT" = Some "1"
  in
  if length = 0 then Error "cannot insert an empty radix key"
  else if length <> Array.length values then
    Error "radix key and value arrays must have equal length"
  else if length mod cache.page_size <> 0 then
    Error "radix insertion must end at a page boundary"
  else
    let access = tick cache in
    let rec walk node offset =
      node.last_access <- access;
      if offset = length then (`Existing node, offset)
      else
        let insert_leaf () =
          let remaining = sub tokens offset (length - offset) in
          let suffix_values = sub values offset (length - offset) in
          let leaf =
            make_node ~parent:node ~namespace ?checkpoint ~resumable:true
              ~key:remaining ~value:suffix_values ~last_access:access ()
          in
          Hashtbl.add node.children (child_key cache namespace remaining) leaf;
          cache.cached_tokens <- cache.cached_tokens + Array.length remaining;
          (`Inserted leaf, offset)
        in
        let follow child prefix =
          child.last_access <- access;
          if prefix = 0 then invalid_arg "radix child index invariant failed"
          else if prefix < Array.length child.key then
            let intermediate = split_node cache namespace child prefix in
            let next_offset = offset + prefix in
            if next_offset = length then (`Existing intermediate, next_offset)
            else walk intermediate next_offset
          else walk child (offset + prefix)
        in
        if offset_insert then
          match
            Hashtbl.find_opt node.children
              (child_key_at cache namespace tokens offset length)
          with
          | None -> insert_leaf ()
          | Some child ->
              follow child
                (aligned cache (common_prefix_at child.key tokens offset length))
        else
          let remaining = sub tokens offset (length - offset) in
          match
            Hashtbl.find_opt node.children (child_key cache namespace remaining)
          with
          | None -> insert_leaf ()
          | Some child ->
              follow child (aligned cache (common_prefix child.key remaining))
    in
    let outcome, prefix_tokens = walk cache.root 0 in
    let retained_node =
      match outcome with `Inserted node | `Existing node -> node
    in
    let redundant_checkpoint =
      match outcome with
      | `Inserted _ -> None
      | `Existing node ->
          let was_resumable = node.resumable in
          node.resumable <- true;
          (match checkpoint, node.checkpoint with
          | None, _ -> None
          | Some checkpoint, None ->
              node.checkpoint <- Some checkpoint;
              None
          | Some checkpoint, Some _ when was_resumable -> Some checkpoint
          | Some _, Some _ -> None)
    in
    let canonical_values =
      if prefix_tokens = 0 then [||]
      else
        path_values retained_node [] |> concatenate
        |> fun values -> sub values 0 prefix_tokens
    in
    Ok
      ( {
          prefix_tokens;
          redundant_values = sub values 0 prefix_tokens;
          redundant_checkpoint;
          canonical_values;
          retained_checkpoint = retained_node.checkpoint;
        },
        retained_node )

let insert cache ~key ~values ?checkpoint () =
  insert_internal cache ~key ~values ?checkpoint () |> Result.map fst

let insert_and_match cache ~key ~values ?checkpoint () =
  match insert_internal cache ~key ~values ?checkpoint () with
  | Error _ as error -> error
  | Ok (result, node) ->
      adjust_lock 1 cache node;
      let lease =
        {
          owner = cache;
          node;
          values = concatenate_path node;
          checkpoint = result.retained_checkpoint;
          released = false;
        }
      in
      Ok (result, lease)

let extend cache lease ~token ~value ?checkpoint () =
  if lease.owner != cache then Error "radix lease belongs to another cache"
  else if lease.released then Error "cannot extend a released radix lease"
  else if cache.page_size <> 1 then
    Error "radix lease extension currently requires page size one"
  else if not lease.node.resumable then
    Error "radix lease extension requires a resumable prefix"
  else
    let parent = lease.node in
    let namespace = parent.namespace in
    let edge = [| token |] in
    let access = tick cache in
    parent.last_access <- access;
    let child, redundant_checkpoint =
      match Hashtbl.find_opt parent.children (child_key cache namespace edge) with
      | None ->
          let child =
            make_node ~parent ~namespace ?checkpoint ~resumable:true ~key:edge
              ~value:[| value |] ~last_access:access ()
          in
          Hashtbl.add parent.children (child_key cache namespace edge) child;
          cache.cached_tokens <- cache.cached_tokens + 1;
          child, None
      | Some child ->
          if Array.length child.key <> 1 || Array.length child.value <> 1 then
            invalid_arg "page-size-one radix child has invalid edge geometry";
          child.last_access <- access;
          let was_resumable = child.resumable in
          child.resumable <- true;
          let redundant_checkpoint =
            match checkpoint, child.checkpoint with
            | None, _ -> None
            | Some checkpoint, None ->
                child.checkpoint <- Some checkpoint;
                None
            | Some checkpoint, Some _ when was_resumable -> Some checkpoint
            | Some _, Some _ -> None
          in
          child, redundant_checkpoint
    in
    adjust_lock 1 cache child;
    let values = Array.make (Array.length lease.values + 1) child.value.(0) in
    Array.blit lease.values 0 values 0 (Array.length lease.values);
    let next =
      {
        owner = cache;
        node = child;
        values;
        checkpoint = child.checkpoint;
        released = false;
      }
    in
    Ok (next, child.value.(0), redundant_checkpoint)

let rec collect_leaves root =
  Hashtbl.fold
    (fun _ child leaves ->
      if Hashtbl.length child.children = 0 && child.lock_ref = 0 then
        child :: leaves
      else collect_leaves child @ leaves)
    root.children []

let least_recent = function
  | [] -> None
  | first :: rest ->
      Some
        (List.fold_left
           (fun oldest node ->
             if node.last_access < oldest.last_access then node else oldest)
           first rest)

let delete_leaf cache node =
  let parent =
    match node.parent with Some parent -> parent | None -> assert false
  in
  Hashtbl.remove parent.children (child_key cache node.namespace node.key);
  cache.cached_tokens <- cache.cached_tokens - Array.length node.value;
  node.parent <- None;
  (node.value, node.checkpoint)

let evict cache ~target_tokens =
  let target = max 0 target_tokens in
  let rec loop removed values checkpoints =
    if removed >= target then (values, checkpoints, removed)
    else
      match least_recent (collect_leaves cache.root) with
      | None -> (values, checkpoints, removed)
      | Some leaf ->
          let leaf_values, leaf_checkpoint = delete_leaf cache leaf in
          let checkpoints =
            match leaf_checkpoint with
            | None -> checkpoints
            | Some checkpoint -> checkpoint :: checkpoints
          in
          loop (removed + Array.length leaf_values)
            (leaf_values :: values) checkpoints
  in
  let values, checkpoints, removed = loop 0 [] [] in
  cache.evicted_tokens <- cache.evicted_tokens + removed;
  { values = concatenate (List.rev values); checkpoints = List.rev checkpoints }

let fold_nodes root f initial =
  let rec visit state node =
    Hashtbl.fold (fun _ child state -> visit (f state child) child)
      node.children state
  in
  visit initial root

let stats cache =
  let evictable_tokens, protected_tokens, checkpoints =
    fold_nodes cache.root
      (fun (evictable, protected, checkpoints) node ->
        let length = Array.length node.value in
        let evictable, protected =
          if node.lock_ref = 0 then (evictable + length, protected)
          else (evictable, protected + length)
        in
        let checkpoints =
          checkpoints + (if Option.is_some node.checkpoint then 1 else 0)
        in
        (evictable, protected, checkpoints))
      (0, 0, 0)
  in
  {
    Stats.cached_tokens = cache.cached_tokens;
    evictable_tokens;
    protected_tokens;
    checkpoints;
    hits = cache.hits;
    misses = cache.misses;
    evicted_tokens = cache.evicted_tokens;
  }

let validate cache =
  let error = ref None in
  let tokens = ref 0 in
  let rec visit namespace parent node =
    if node != cache.root then (
      tokens := !tokens + Array.length node.key;
      if Array.length node.key = 0 then error := Some "empty non-root radix edge"
      else if Array.length node.key <> Array.length node.value then
        error := Some "radix edge key/value length mismatch"
      else if Array.length node.key mod cache.page_size <> 0 then
        error := Some "radix edge is not page aligned"
      else if node.lock_ref < 0 then error := Some "negative radix lock reference";
      match node.parent with
      | Some actual when actual == parent -> ()
      | _ -> error := Some "radix parent link mismatch");
    Hashtbl.iter
      (fun ((child_namespace, page) as index) child ->
        let expected_namespace = if node == cache.root then child_namespace else namespace in
        if child_namespace <> expected_namespace then
          error := Some "radix namespace changed below root";
        if child.namespace <> expected_namespace then
          error := Some "radix node namespace mismatch";
        if Array.length child.key = 0 ||
           page <> Array.to_list (Array.sub child.key 0 (min cache.page_size (Array.length child.key))) then
          error := Some "radix child index mismatch";
        if index <> child_key cache child_namespace child.key then
          error := Some "radix child key mismatch";
        visit expected_namespace node child)
      node.children
  in
  if cache.root.parent <> None then error := Some "radix root has a parent";
  if cache.root.lock_ref <> 1 then error := Some "radix root lock must remain one";
  visit None cache.root cache.root;
  if !tokens <> cache.cached_tokens then
    error := Some "radix cached token accounting mismatch";
  match !error with None -> Ok () | Some message -> Error message

module Speculative = struct
  type ('slot, 'checkpoint) reservation = {
    key : Key.t;
    slots : 'slot array;
    checkpoint : 'checkpoint option;
  }

  let create ?checkpoint ~key ~slots () =
    { key; slots; checkpoint }

  let key res = res.key
  let slots res = Array.copy res.slots
  let checkpoint res = res.checkpoint
end

let commit_speculative cache ~reservation ~accepted_count =
  if accepted_count < 0 || accepted_count > Array.length reservation.Speculative.slots then
    Error "accepted_count out of speculative reservation bounds"
  else
    let total = Array.length reservation.Speculative.slots in
    let rejected_slots =
      sub reservation.Speculative.slots accepted_count (total - accepted_count)
    in
    if accepted_count = 0 then
      Ok (None, rejected_slots)
    else
      let accepted_tokens =
        sub (Key.tokens reservation.Speculative.key) 0 accepted_count
      in
      let accepted_slots =
        sub reservation.Speculative.slots 0 accepted_count
      in
      let insertable_length = aligned cache accepted_count in
      if insertable_length = 0 then
        Ok (None, rejected_slots)
      else
        let page_tokens = sub accepted_tokens 0 insertable_length in
        let page_slots = sub accepted_slots 0 insertable_length in
        let key =
          Key.create ?namespace:(Key.namespace reservation.Speculative.key)
            page_tokens
        in
        match reservation.Speculative.checkpoint with
        | Some cp -> (
            match insert cache ~key ~values:page_slots ~checkpoint:cp () with
            | Ok result -> Ok (Some result, rejected_slots)
            | Error err -> Error err)
        | None -> (
            match insert cache ~key ~values:page_slots () with
            | Ok result -> Ok (Some result, rejected_slots)
            | Error err -> Error err)
