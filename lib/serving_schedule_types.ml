(** Serving schedule shared types and slot helpers: command/stage/tensor-
   input representations plus the integer-set/value-id substrate shared by
   validation, core, and codec.  Split out of Serving_schedule; LOOP. *)

module Command = struct
  type t = {
    node_id : int;
    op : Ir.Op.t;
    inputs : Ir.Value.t list;
    output : Ir.Value.t option;
    fusion_region : Kernel_region.t option;
  }

  let node_id command = command.node_id
  let op command = command.op
  let inputs command = command.inputs
  let output command = command.output
  let fusion_region command = command.fusion_region
end

module Stage = struct
  type t =
    | Sequential of Command.t
    | Concurrent of Command.t list
    | Barrier of int

  let commands = function
    | Sequential c -> [ c ]
    | Concurrent cs -> cs
    | Barrier _ -> []

  let is_barrier = function
    | Barrier _ -> true
    | _ -> false
end

module Tensor_input = struct
  type t = {
    key : string;
    value : Ir.Value.t;
  }

  let key input = input.key
  let value input = input.value
end
module Int_set = Set.Make (Int)
let value_id value = Ir.Value.id value |> Ir.Value_id.to_int
