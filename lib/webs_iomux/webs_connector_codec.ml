(* Webs_connector_codec — a vendored, stable copy of the three HTTP/1.1
 * codecs that [Webs_iomux] needs from webs' [Http.Connector.Private]
 * module.
 *
 * webs is ISC-licensed. Rather than reach into the undocumented [Private]
 * surface (which can change between releases), this module re-implements the
 * request-line/headers decoders and the response-head encoder on top of
 * webs' *public* API only:
 *
 *   - Method.decode / Version.decode   (public)
 *   - Headers.decode_http11_header     (public)
 *   - Headers.add_value / add_set_cookie / Name.equal / Name.v / Name.encode
 *     / fold / empty                   (public)
 *
 * The functions keep the same signatures and raising semantics as the
 * originals, so the connector code that calls them is unchanged.
 *)

module H = Webs.Http

let crlf = "\r\n"

(* Decode the request line in [buf] spanning [first .. crlf).
 * [crlf] is the offset of the CRLF that terminates the request line.
 * Returns [(method', target, version)], raising on malformed input —
 * matching the contract of the original [Private.decode_request_line]. *)
let decode_request_line b ~first ~crlf =
  let line = Bytes.sub_string b first (crlf - first) in
  match String.split_on_char ' ' line with
  | [ meth; target; version ] -> (
      let method' =
        match H.Method.decode meth with
        | Ok m -> m
        | Error e -> failwith ("malformed method: " ^ e)
      in
      let version =
        match H.Version.decode version with
        | Ok v -> v
        | Error e -> failwith ("malformed version: " ^ e)
      in
      (method', target, version))
  | _ -> failwith "malformed request line"

(* Decode the header section. [crlfs] is the list of CRLF offsets where the
 * head is [List.hd crlfs] (end of request line) and the tail are the ends of
 * each header field. Raises on a malformed header field. *)
let decode_headers buf ~crlfs =
  let rec loop acc last_crlf = function
    | [] -> acc
    | crlf :: crlfs ->
        let first = last_crlf + 2 in
        let s = Bytes.sub_string buf first (crlf - first) in
        let name, value =
          match H.Headers.decode_http11_header s with
          | Ok v -> v
          | Error e -> failwith ("malformed header: " ^ e)
        in
        let acc =
          if H.Headers.Name.equal name H.Headers.set_cookie then
            H.Headers.add_set_cookie value acc
          else H.Headers.add_value name value acc
        in
        loop acc crlf crlfs
  in
  loop H.Headers.empty (List.hd crlfs) (List.tl crlfs)

(* Encode an HTTP/1.1 response head (status line + headers + trailing CRLF).
 * [status] is the integer status code; [hs] already includes any
 * Content-Length / Transfer-Encoding the caller wants on the wire. *)
let encode_http11_response_head status ~reason hs =
  let status_s = string_of_int status in
  let hs_s =
    H.Headers.fold
      (fun n v acc -> H.Headers.Name.encode n ^ ": " ^ v ^ crlf ^ acc)
      hs ""
  in
  "HTTP/1.1 " ^ status_s ^ " " ^ reason ^ crlf ^ hs_s ^ crlf
