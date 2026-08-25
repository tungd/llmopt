let name = "fuse_qkv_linear"
let description =
  "Fuse co-dependent query, key, and value Q8 projections into one QKV kernel"

let value_is left right = Ir.Value.equal left right

type projection = {
  node : Ir.node;
  m : int;
  n : int;
  k : int;
  bias : bool;
  input : Ir.Value.t;
  parameters : Ir.Value.t list;
  output : Ir.Value.t;
  index : int;
}

let q8_linear index node =
  match Ir.node_op node, Ir.node_inputs node, Ir.node_output node with
  | Ir.Op.Q8_linear { m; n; k; bias }, input :: parameters, Some output ->
      Some { node; m; n; k; bias; input; parameters; output; index }
  | _ -> None

let same_projection left right =
  left.m = right.m && left.k = right.k && left.bias = right.bias
  && value_is left.input right.input

let find_qkv nodes =
  let indexed =
    nodes
    |> List.mapi (fun index node -> q8_linear index node)
    |> List.filter_map (fun projection -> projection)
  in
  let rec search = function
    | q :: rest ->
        let key_and_value =
          List.find_map
            (fun k ->
              if k.index <= q.index || not (same_projection q k) then None
              else
                List.find_map
                  (fun v ->
                    if
                      v.index > k.index && same_projection q v && v.n = k.n
                    then Some (k, v)
                    else None)
                  indexed
                )
            indexed
        in
        (match key_and_value with
        | Some (k, v) -> Some (q, k, v)
        | None -> search rest)
    | [] -> None
  in
  search indexed

let fuse q k v =
  Ir.node_create ~id:(Ir.node_id q.node)
    ~op:
      (Ir.Op.Q8_qkv_linear
         {
           m = q.m;
           n_q = q.n;
           n_kv = k.n;
           k = q.k;
           bias = q.bias;
           extra_outputs = [ k.output; v.output ];
         })
    ~inputs:(q.input :: (q.parameters @ k.parameters @ v.parameters))
    ~output:(Some q.output)

let remove_nodes nodes q k v fused =
  List.filter_map
    (fun node ->
      if Ir.node_id node = Ir.node_id q.node then Some fused
      else if
        Ir.node_id node = Ir.node_id k.node || Ir.node_id node = Ir.node_id v.node
      then None
      else Some node)
    nodes

let run graph =
  let rec rewrite nodes =
    match find_qkv nodes with
    | None -> Ir.Graph.with_nodes graph nodes
    | Some (q, k, v) -> rewrite (remove_nodes nodes q k v (fuse q k v))
  in
  rewrite (Ir.Graph.nodes graph)

let pass = Pass.create ~name ~description ~run
