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

let make_node ?parent ?checkpoint ?(namespace = None) ?(lock_ref = 0) ~key ~value
    ~last_access () =
  {
    namespace;
    key;
    value;
    children = Hashtbl.create 4;
    parent;
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

let common_prefix left right =
  let limit = min (Array.length left) (Array.length right) in
  let rec loop index =
    if index = limit || left.(index) <> right.(index) then index
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
  let access = tick cache in
  cache.root.last_access <- access;
  let rec walk node offset best =
    if offset = usable_length then (node, best)
    else
      let remaining = sub tokens offset (usable_length - offset) in
      match Hashtbl.find_opt node.children (child_key cache namespace remaining) with
      | None -> (node, best)
      | Some child ->
          child.last_access <- access;
          let prefix = aligned cache (common_prefix child.key remaining) in
          if prefix = 0 then (node, best)
          else if prefix < Array.length child.key then
            let intermediate = split_node cache namespace child prefix in
            let best =
              match intermediate.checkpoint with
              | Some _ -> intermediate
              | None -> best
            in
            (intermediate, best)
          else
            let best =
              match child.checkpoint with Some _ -> child | None -> best
            in
            walk child (offset + prefix) best
  in
  let _, best = walk cache.root 0 cache.root in
  let values =
    if best == cache.root then [||] else concatenate (path_values best [])
  in
  let checkpoint = best.checkpoint in
  if Array.length values = 0 then cache.misses <- cache.misses + 1
  else cache.hits <- cache.hits + 1;
  adjust_lock 1 cache best;
  { owner = cache; node = best; values; checkpoint; released = false }

let matched_tokens (lease : (_, _) lease) = Array.length lease.values
let matched_values (lease : (_, _) lease) = Array.copy lease.values
let checkpoint (lease : (_, _) lease) = lease.checkpoint

let release cache lease =
  if lease.owner != cache then Error "radix lease belongs to another cache"
  else if lease.released then Error "radix lease was already released"
  else (
    adjust_lock (-1) cache lease.node;
    lease.released <- true;
    Ok ())

let insert cache ~key ~values ~checkpoint =
  let tokens = key.Key.tokens in
  let namespace = key.Key.namespace in
  let length = Array.length tokens in
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
        let remaining = sub tokens offset (length - offset) in
        match Hashtbl.find_opt node.children (child_key cache namespace remaining) with
        | None ->
            let suffix_values = sub values offset (length - offset) in
            let leaf =
              make_node ~parent:node ~namespace ~checkpoint ~key:remaining
                ~value:suffix_values ~last_access:access ()
            in
            Hashtbl.add node.children (child_key cache namespace remaining) leaf;
            cache.cached_tokens <- cache.cached_tokens + Array.length remaining;
            (`Inserted leaf, offset)
        | Some child ->
            child.last_access <- access;
            let prefix = aligned cache (common_prefix child.key remaining) in
            if prefix = 0 then invalid_arg "radix child index invariant failed"
            else if prefix < Array.length child.key then
              let intermediate = split_node cache namespace child prefix in
              let next_offset = offset + prefix in
              if next_offset = length then (`Existing intermediate, next_offset)
              else walk intermediate next_offset
            else walk child (offset + prefix)
    in
    let outcome, prefix_tokens = walk cache.root 0 in
    let retained_node =
      match outcome with `Inserted node | `Existing node -> node
    in
    let redundant_checkpoint =
      match outcome with
      | `Inserted _ -> None
      | `Existing node ->
          (match node.checkpoint with
          | None ->
              node.checkpoint <- Some checkpoint;
              None
          | Some _ -> Some checkpoint)
    in
    let canonical_values =
      if prefix_tokens = 0 then [||]
      else
        path_values retained_node [] |> concatenate
        |> fun values -> sub values 0 prefix_tokens
    in
    Ok
      {
        prefix_tokens;
        redundant_values = sub values 0 prefix_tokens;
        redundant_checkpoint;
        canonical_values;
        retained_checkpoint = retained_node.checkpoint;
      }

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
