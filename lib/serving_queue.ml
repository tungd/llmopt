module Request_id = struct
  type t = int

  let counter = ref 0

  let create () =
    incr counter;
    !counter

  let of_int x = x
  let to_int x = x
  let compare = Int.compare
  let equal (a : int) (b : int) = a = b
  let to_string = string_of_int
end

type request_state =
  | Pending_prefill of {
      prompt_tokens : int array;
      cached_tokens : int;
      remaining_prefill : int;
      max_new_tokens : int;
      ignore_eos : bool;
    }
  | Active_decode of {
      prompt_length : int;
      generated_tokens : int list;
      max_new_tokens : int;
      ignore_eos : bool;
    }

type request = {
  id : Request_id.t;
  arrival_time : float;
  state : request_state;
  mutable priority_score : float;
}

module Score = struct
  let default_prefill_rate = 100.0 (* tokens / ms *)
  let default_decode_rate = 10.0  (* tokens / ms *)
  let default_alpha_age = 0.01
  let default_epsilon = 1e-4

  let remaining_processing_time ~prefill_rate ~decode_rate state =
    match state with
    | Pending_prefill { remaining_prefill; max_new_tokens; _ } ->
        let prefill_time =
          Float.of_int remaining_prefill /. max 0.001 prefill_rate
        in
        let decode_time =
          Float.of_int max_new_tokens /. max 0.001 decode_rate
        in
        prefill_time +. decode_time
    | Active_decode { generated_tokens; max_new_tokens; _ } ->
        let generated_count = List.length generated_tokens in
        let remaining_decode = max 0 (max_new_tokens - generated_count) in
        Float.of_int remaining_decode /. max 0.001 decode_rate

  let compute ?(alpha_age = default_alpha_age) ?(epsilon = default_epsilon)
      ~prefill_rate ~decode_rate ~current_time ~arrival_time state =
    let rem_time =
      remaining_processing_time ~prefill_rate ~decode_rate state
    in
    let srpt_term = 1.0 /. (max 1e-6 rem_time +. epsilon) in
    let age = max 0.0 (current_time -. arrival_time) in
    let age_term = alpha_age *. age in
    srpt_term +. age_term
end

module Request_order = struct
  type t = request

  let compare (r1 : request) (r2 : request) =
    (* Higher priority score comes first *)
    let score_cmp = Float.compare r2.priority_score r1.priority_score in
    if score_cmp <> 0 then score_cmp
    else
      (* Earlier arrival time comes first (FIFO tie-breaking) *)
      let arrival_cmp = Float.compare r1.arrival_time r2.arrival_time in
      if arrival_cmp <> 0 then arrival_cmp
      else
        (* Deterministic ID tie-breaking *)
        Request_id.compare r1.id r2.id
end

module Request_set = Set.Make (Request_order)
module Request_map = Map.Make (Request_id)

type t = {
  alpha_age : float;
  prefill_rate : float;
  decode_rate : float;
  mutable queue : Request_set.t;
  mutable by_id : request Request_map.t;
}

let create ?(alpha_age = Score.default_alpha_age)
    ?(prefill_rate = Score.default_prefill_rate)
    ?(decode_rate = Score.default_decode_rate) () =
  {
    alpha_age;
    prefill_rate;
    decode_rate;
    queue = Request_set.empty;
    by_id = Request_map.empty;
  }

let is_empty t = Request_set.is_empty t.queue
let length t = Request_set.cardinal t.queue

let enqueue t req =
  (* If request already exists, remove it first *)
  (match Request_map.find_opt req.id t.by_id with
  | Some existing ->
      t.queue <- Request_set.remove existing t.queue
  | None -> ());
  t.queue <- Request_set.add req t.queue;
  t.by_id <- Request_map.add req.id req t.by_id

let peek_next t =
  Request_set.min_elt_opt t.queue

let pop_next t =
  match Request_set.min_elt_opt t.queue with
  | None -> None
  | Some req ->
      t.queue <- Request_set.remove req t.queue;
      t.by_id <- Request_map.remove req.id t.by_id;
      Some req

let remove t id =
  match Request_map.find_opt id t.by_id with
  | None -> false
  | Some req ->
      t.queue <- Request_set.remove req t.queue;
      t.by_id <- Request_map.remove id t.by_id;
      true

let find t id =
  Request_map.find_opt id t.by_id

let update_scores t ~current_time =
  let updated_requests =
    Request_map.fold
      (fun _id req acc ->
        req.priority_score <-
          Score.compute ~alpha_age:t.alpha_age ~prefill_rate:t.prefill_rate
            ~decode_rate:t.decode_rate ~current_time ~arrival_time:req.arrival_time
            req.state;
        req :: acc)
      t.by_id []
  in
  let new_queue =
    List.fold_left
      (fun q req -> Request_set.add req q)
      Request_set.empty updated_requests
  in
  t.queue <- new_queue

let to_list t =
  Request_set.elements t.queue
