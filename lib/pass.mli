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

val create :
  name:string ->
  description:string ->
  run:(Ir.Graph.t -> Ir.Graph.t) ->
  t

val of_module : (module PASS) -> t

val name : t -> string
val description : t -> string
val run : t -> Ir.Graph.t -> Ir.Graph.t

module Pipeline : sig
  type pipeline
  type t = pipeline

  val empty : pipeline
  val add : pass -> pipeline -> pipeline
  val of_list : pass list -> pipeline
  val run : pipeline -> Ir.Graph.t -> Ir.Graph.t
  val passes : pipeline -> pass list
end
