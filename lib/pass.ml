module type PASS = sig
  val name : string
  val description : string
  val run : Ir.Graph.t -> Ir.Graph.t
end

type t = {
  name : string;
  description : string;
  run : Ir.Graph.t -> Ir.Graph.t;
}

type pass = t

let create ~name ~description ~run = { name; description; run }

let of_module (module M : PASS) =
  { name = M.name; description = M.description; run = M.run }

let name pass = pass.name
let description pass = pass.description
let run pass graph = pass.run graph

module Pipeline = struct
  type pipeline = {
    passes : pass list;
  }
  type t = pipeline

  let empty = { passes = [] }

  let add pass pipeline =
    { passes = pipeline.passes @ [ pass ] }

  let of_list passes = { passes }

  let run pipeline graph =
    List.fold_left (fun g pass -> pass.run g) graph pipeline.passes

  let passes pipeline = pipeline.passes
end
