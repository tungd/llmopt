let ( let* ) = Result.bind

module Value_map = Map.Make (struct
  type t = Ir.Value_id.t

  let compare = Ir.Value_id.compare
end)

module Allocation = struct
  type t = { offset : int; bytes : int }

  let offset allocation = allocation.offset
  let bytes allocation = allocation.bytes
end

module Storage = struct
  type t = External | Workspace of Ir.Value_id.t
end

type interval = {
  owner : Ir.Value_id.t;
  first : int;
  last : int;
  bytes : int;
  reserved : int;
}

type block = { offset : int; bytes : int }

type active = {
  last : int;
  block : block;
}

type t = {
  storage_by_value : Storage.t Value_map.t;
  allocation_by_owner : Allocation.t Value_map.t;
  workspace_bytes : int;
  bytes_without_reuse : int;
}

let workspace_bytes plan = plan.workspace_bytes
let bytes_without_reuse plan = plan.bytes_without_reuse
let allocation_count plan = Value_map.cardinal plan.allocation_by_owner

let value_bytes value =
  let shape = Ir.Value.logical_shape value in
  Ir.Tensor_layout.physical_bytes (Ir.Value.layout value) shape

let alignment = 256

let align bytes =
  if bytes > max_int - (alignment - 1) then
    Error "workspace allocation size overflows"
  else Ok (((bytes + alignment - 1) / alignment) * alignment)

let checked_add left right message =
  if right > max_int - left then Error message else Ok (left + right)

let alias_input command =
  match
    Serving_schedule.Command.op command,
    Serving_schedule.Command.inputs command,
    Serving_schedule.Command.output command
  with
  | ( Ir.Op.Primitive
        (Ir.Primitive.Movement
          (Ir.Movement.View | Ir.Movement.Reshape | Ir.Movement.Unsqueeze _
          | Ir.Movement.Contiguous)),
      [ input ],
      Some output ) ->
      Some (input, output)
  | Ir.Op.Primitive (Ir.Primitive.Cast dtype), [ input ], Some output
    when dtype = Ir.Value.dtype input ->
      Some (input, output)
  | _ -> None

let storage_for storage_by_value value =
  match Value_map.find_opt (Ir.Value.id value) storage_by_value with
  | Some storage -> Ok storage
  | None ->
      Error
        (Printf.sprintf "workspace plan reads undefined value %d"
           (Ir.Value.id value |> Ir.Value_id.to_int))

let record_use (intervals : interval Value_map.t) storage index =
  match storage with
  | Storage.External -> Ok intervals
  | Storage.Workspace owner ->
      (match Value_map.find_opt owner intervals with
      | None ->
          Error
            (Printf.sprintf "workspace owner %d has no live interval"
               (Ir.Value_id.to_int owner))
      | Some interval ->
          Ok
            (Value_map.add owner
               { interval with last = max interval.last index }
               intervals))

let record_inputs storage_by_value (intervals : interval Value_map.t) index inputs =
  List.fold_left
    (fun result input ->
      let* intervals = result in
      let* storage = storage_for storage_by_value input in
      record_use intervals storage index)
    (Ok intervals) inputs

let collect_intervals schedule =
  let commands = Serving_schedule.commands schedule in
  let terminal = List.length commands in
  let has_barriers =
    List.exists
      (fun c ->
        match Serving_schedule.Command.op c with
        | Ir.Op.Barrier_wait _ | Ir.Op.Barrier_arrive _ -> true
        | _ -> false)
      commands
  in
  let stage_ends =
    if not has_barriers then Array.init terminal (fun i -> i)
    else
      let arr = Array.make terminal terminal in
      let current_end = ref terminal in
      let commands_rev = List.rev (List.mapi (fun i c -> (i, c)) commands) in
      List.iter
        (fun (i, c) ->
          match Serving_schedule.Command.op c with
          | Ir.Op.Barrier_wait _ | Ir.Op.Barrier_arrive _ ->
              current_end := i;
              arr.(i) <- i
          | _ ->
              arr.(i) <- !current_end)
        commands_rev;
      arr
  in
  let allocate_workspace_output index storage_by_value intervals
      bytes_without_reuse output =
    let owner = Ir.Value.id output in
    let* bytes = value_bytes output in
    let* reserved = align bytes in
    let* bytes_without_reuse =
      checked_add bytes_without_reuse reserved
        "workspace unreused byte count overflows"
    in
    let interval = { owner; first = index; last = index; bytes; reserved } in
    Ok
      ( Value_map.add owner (Storage.Workspace owner) storage_by_value,
        Value_map.add owner interval intervals,
        bytes_without_reuse )
  in
  let allocate_additional_outputs index op storage_by_value intervals
      bytes_without_reuse =
    List.fold_left
      (fun result output ->
        let* storage_by_value, intervals, bytes_without_reuse = result in
        allocate_workspace_output index storage_by_value intervals
          bytes_without_reuse output)
      (Ok (storage_by_value, intervals, bytes_without_reuse))
      (Ir.Op.additional_outputs op)
  in
  let rec collect index storage_by_value
      (intervals : interval Value_map.t) bytes_without_reuse = function
    | [] -> Ok (storage_by_value, intervals, bytes_without_reuse)
    | command :: rest ->
        let op = Serving_schedule.Command.op command in
        let inputs = Serving_schedule.Command.inputs command in
        let output = Serving_schedule.Command.output command in
        let use_index =
          match op with
          | Ir.Op.Output _ -> terminal
          | _ -> stage_ends.(index)
        in
        let* intervals =
          record_inputs storage_by_value intervals use_index inputs
        in
        (match op, output, alias_input command with
        | Ir.Op.Input _, Some output, _ ->
            let storage_by_value =
              Value_map.add (Ir.Value.id output) Storage.External storage_by_value
            in
            let* storage_by_value, intervals, bytes_without_reuse =
              allocate_additional_outputs index op storage_by_value intervals
                bytes_without_reuse
            in
            collect (index + 1) storage_by_value intervals bytes_without_reuse rest
        | _, Some output, Some (input, alias) ->
            let* input_bytes = value_bytes input in
            let* output_bytes = value_bytes alias in
            if input_bytes <> output_bytes then
              Error
                (Printf.sprintf
                   "workspace alias value %d changes storage size from %d to %d"
                   (Ir.Value.id output |> Ir.Value_id.to_int)
                   input_bytes output_bytes)
            else
              let* storage = storage_for storage_by_value input in
              let storage_by_value =
                Value_map.add (Ir.Value.id output) storage storage_by_value
              in
              let* storage_by_value, intervals, bytes_without_reuse =
                allocate_additional_outputs index op storage_by_value intervals
                  bytes_without_reuse
              in
              collect (index + 1) storage_by_value intervals bytes_without_reuse rest
        | _, Some output, None ->
            let* storage_by_value, intervals, bytes_without_reuse =
              allocate_workspace_output index storage_by_value intervals
                bytes_without_reuse output
            in
            let* storage_by_value, intervals, bytes_without_reuse =
              allocate_additional_outputs index op storage_by_value intervals
                bytes_without_reuse
            in
            collect (index + 1) storage_by_value intervals bytes_without_reuse rest
        | _, None, _ ->
            if Ir.Op.additional_outputs op = [] then
              collect (index + 1) storage_by_value intervals bytes_without_reuse rest
            else Error "multi-output schedule command has no primary output")
  in
  collect 0 Value_map.empty Value_map.empty 0 commands

let coalesce blocks =
  let sorted = List.sort (fun left right -> Int.compare left.offset right.offset) blocks in
  let rec merge acc = function
    | [] -> List.rev acc
    | block :: rest ->
        (match acc with
        | previous :: tail when previous.offset + previous.bytes = block.offset ->
            merge ({ previous with bytes = previous.bytes + block.bytes } :: tail) rest
        | _ -> merge (block :: acc) rest)
  in
  merge [] sorted

let release_finished first active free =
  List.fold_left
    (fun (active, free) allocation ->
      if allocation.last < first then active, allocation.block :: free
      else allocation :: active, free)
    ([], free) active
  |> fun (active, free) -> active, coalesce free

let take_best_fit bytes blocks =
  let better candidate best =
    candidate.bytes < best.bytes
    || (candidate.bytes = best.bytes && candidate.offset < best.offset)
  in
  let best =
    List.fold_left
      (fun best block ->
        if block.bytes < bytes then best
        else
          match best with
          | None -> Some block
          | Some current when better block current -> Some block
          | Some _ -> best)
      None blocks
  in
  match best with
  | None -> None
  | Some selected ->
      let rec remove acc = function
        | [] -> List.rev acc
        | block :: rest when block.offset = selected.offset ->
            let remainder =
              if block.bytes = bytes then acc
              else
                { offset = block.offset + bytes; bytes = block.bytes - bytes }
                :: acc
            in
            List.rev_append remainder rest
        | block :: rest -> remove (block :: acc) rest
      in
      Some (selected.offset, remove [] blocks)

let allocate_intervals intervals =
  let sorted =
    intervals |> Value_map.bindings |> List.map snd
    |> List.sort (fun left right ->
           let by_start = Int.compare left.first right.first in
           if by_start <> 0 then by_start
           else Ir.Value_id.compare left.owner right.owner)
  in
  let rec allocate active free high_water allocations = function
    | [] -> Ok (allocations, high_water)
    | interval :: rest ->
        let active, free = release_finished interval.first active free in
        let* offset, free, high_water =
          match take_best_fit interval.reserved free with
          | Some (offset, free) -> Ok (offset, free, high_water)
          | None ->
              let* next =
                checked_add high_water interval.reserved
                  "workspace high-water mark overflows"
              in
              Ok (high_water, free, next)
        in
        let allocation =
          { Allocation.offset = offset; bytes = interval.bytes }
        in
        let active =
          { last = interval.last; block = { offset; bytes = interval.reserved } }
          :: active
        in
        allocate active free high_water
          (Value_map.add interval.owner allocation allocations)
          rest
  in
  allocate [] [] 0 Value_map.empty sorted

let create schedule =
  let* storage_by_value, intervals, bytes_without_reuse =
    collect_intervals schedule
  in
  let* allocation_by_owner, workspace_bytes = allocate_intervals intervals in
  Ok
    {
      storage_by_value;
      allocation_by_owner;
      workspace_bytes;
      bytes_without_reuse;
    }

let plan_concurrent = create

let allocation plan value =
  match Value_map.find_opt (Ir.Value.id value) plan.storage_by_value with
  | Some (Storage.Workspace owner) ->
      Value_map.find_opt owner plan.allocation_by_owner
  | Some Storage.External | None -> None

let check_disjoint plan values =
  let allocs =
    List.filter_map (fun v -> allocation plan v) values
  in
  let arr = Array.of_list allocs in
  let len = Array.length arr in
  let rec check i j =
    if i >= len then true
    else if j >= len then check (i + 1) (i + 2)
    else
      let a = arr.(i) in
      let b = arr.(j) in
      let a_start = a.Allocation.offset in
      let a_end = a_start + a.Allocation.bytes in
      let b_start = b.Allocation.offset in
      let b_end = b_start + b.Allocation.bytes in
      if max a_start b_start < min a_end b_end then false
      else check i (j + 1)
  in
  check 0 1

let to_bytes plan =
  let writer = Binary.Writer.create () in
  Binary.Writer.raw_string writer "LLMOPTMP";
  Binary.Writer.u16 writer 1;
  Binary.Writer.u64 writer plan.workspace_bytes;
  Binary.Writer.u64 writer plan.bytes_without_reuse;
  Binary.Writer.u64 writer (Value_map.cardinal plan.storage_by_value);
  Value_map.iter
    (fun value storage ->
      Binary.Writer.u64 writer (Ir.Value_id.to_int value);
      match storage with
      | Storage.External -> Binary.Writer.u8 writer 0
      | Storage.Workspace owner ->
          Binary.Writer.u8 writer 1;
          Binary.Writer.u64 writer (Ir.Value_id.to_int owner))
    plan.storage_by_value;
  Binary.Writer.u64 writer (Value_map.cardinal plan.allocation_by_owner);
  Value_map.iter
    (fun owner allocation ->
      Binary.Writer.u64 writer (Ir.Value_id.to_int owner);
      Binary.Writer.u64 writer allocation.Allocation.offset;
      Binary.Writer.u64 writer allocation.Allocation.bytes)
    plan.allocation_by_owner;
  Binary.Writer.contents writer

let schedule_values schedule =
  let values = Hashtbl.create 128 in
  let add value =
    Hashtbl.replace values (Ir.Value.id value |> Ir.Value_id.to_int) value
  in
  List.iter
    (fun command ->
      List.iter add (Serving_schedule.Command.inputs command);
      Option.iter add (Serving_schedule.Command.output command);
      List.iter add
        (Ir.Op.additional_outputs (Serving_schedule.Command.op command)))
    (Serving_schedule.commands schedule);
  values

let of_bytes ~schedule bytes =
  let reader = Binary.Reader.create bytes in
  let* magic = Binary.Reader.raw_string reader ~length:8 in
  if magic <> "LLMOPTMP" then Error "invalid serving memory-plan magic"
  else
    let* version = Binary.Reader.u16 reader in
    if version <> 1 then
      Error (Printf.sprintf "unsupported serving memory-plan version: %d" version)
    else
      let* workspace_bytes = Binary.Reader.u64 reader in
      let* bytes_without_reuse = Binary.Reader.u64 reader in
      let values = schedule_values schedule in
      let resolve encoded =
        match Hashtbl.find_opt values encoded with
        | Some value -> Ok (Ir.Value.id value)
        | None ->
            Error
              (Printf.sprintf
                 "serving memory plan references unknown schedule value %d"
                 encoded)
      in
      let* storage_count = Binary.Reader.u64 reader in
      let rec read_storage remaining storage_by_value =
        if remaining = 0 then Ok storage_by_value
        else
          let* encoded_value = Binary.Reader.u64 reader in
          let* value = resolve encoded_value in
          let* tag = Binary.Reader.u8 reader in
          let* storage =
            match tag with
            | 0 -> Ok Storage.External
            | 1 ->
                let* encoded_owner = Binary.Reader.u64 reader in
                let* owner = resolve encoded_owner in
                Ok (Storage.Workspace owner)
            | tag ->
                Error
                  (Printf.sprintf
                     "unknown serving memory-plan storage tag: %d" tag)
          in
          if Value_map.mem value storage_by_value then
            Error
              (Printf.sprintf
                 "duplicate serving memory-plan value %d" encoded_value)
          else
            read_storage (remaining - 1)
              (Value_map.add value storage storage_by_value)
      in
      let* storage_by_value = read_storage storage_count Value_map.empty in
      let* allocation_count = Binary.Reader.u64 reader in
      let rec read_allocations remaining allocation_by_owner =
        if remaining = 0 then Ok allocation_by_owner
        else
          let* encoded_owner = Binary.Reader.u64 reader in
          let* owner = resolve encoded_owner in
          let* offset = Binary.Reader.u64 reader in
          let* bytes = Binary.Reader.u64 reader in
          if bytes <= 0 then Error "serving memory-plan allocation is empty"
          else if offset > workspace_bytes || bytes > workspace_bytes - offset then
            Error "serving memory-plan allocation exceeds its workspace"
          else if Value_map.mem owner allocation_by_owner then
            Error
              (Printf.sprintf
                 "duplicate serving memory-plan owner %d" encoded_owner)
          else
            read_allocations (remaining - 1)
              (Value_map.add owner { Allocation.offset; bytes }
                 allocation_by_owner)
      in
      let* allocation_by_owner =
        read_allocations allocation_count Value_map.empty
      in
      let* () = Binary.Reader.finish reader in
      let* () =
        Value_map.fold
          (fun value storage result ->
            let* () = result in
            match storage with
            | Storage.External -> Ok ()
            | Storage.Workspace owner ->
                if Value_map.mem owner allocation_by_owner then Ok ()
                else
                  Error
                    (Printf.sprintf
                       "serving memory-plan value %d has no owner allocation"
                       (Ir.Value_id.to_int value)))
          storage_by_value (Ok ())
      in
      Ok
        { storage_by_value; allocation_by_owner; workspace_bytes;
          bytes_without_reuse }
