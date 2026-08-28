(* Webs_iomux — an HTTP/1.1 connector for webs driven by an Iomux poll loop.
 *
 * This is an adoption spike: it plugs httpkit's non-blocking event-loop I/O
 * model (Iomux.Poll) into webs' transport-agnostic HTTP model, so that a
 * webs [Http.Request.t -> Http.Response.t] service runs on a single-threaded
 * multiplexing server instead of webs' thread-pool gateway connector.
 *
 * It reuses a vendored, stable copy of webs' HTTP/1.1 codecs
 * (Webs_connector_codec) to decode the request line / headers and to encode
 * the response head, but performs all socket I/O itself with non-blocking
 * Unix.read/Unix.write — it does NOT use Webs_unix's blocking fd helpers.
 * Response bodies are streamed to the socket via webs' Bytes.Reader and
 * chunked transfer-encoding when their length is unknown.
 *
 * Slow-client mitigation (header/body/idle deadlines) is included, which the
 * stock webs gateway connector deliberately omits. *)

module H = Webs.Http
module Bw = Bytesrw.Bytes

let crlf = "\r\n"

type stream = {
  lock : Mutex.t;
  chunks : string Queue.t;
  max_chunks : int;
  not_empty : Condition.t;
  not_full : Condition.t;
  mutable closed : bool;
}

type H.Body.custom_content += Iomux_stream of stream

module Stream = struct
  type t = stream

  let create ?(max_chunks = 256) () =
    if max_chunks < 1 then invalid_arg "Iomux.Stream.create: max_chunks";
    { lock = Mutex.create (); chunks = Queue.create (); max_chunks;
      not_empty = Condition.create (); not_full = Condition.create ();
      closed = false }

  let push t chunk =
    Mutex.lock t.lock;
    Fun.protect
      ~finally:(fun () -> Mutex.unlock t.lock)
      (fun () ->
        if t.closed || Queue.length t.chunks >= t.max_chunks then false
        else (Queue.add chunk t.chunks; Condition.signal t.not_empty; true))

  let push_wait t chunk =
    Mutex.lock t.lock;
    Fun.protect
      ~finally:(fun () -> Mutex.unlock t.lock)
      (fun () ->
        while not t.closed && Queue.length t.chunks >= t.max_chunks do
          Condition.wait t.not_full t.lock
        done;
        if t.closed then false
        else (Queue.add chunk t.chunks; Condition.signal t.not_empty; true))

  let replace t chunk =
    Mutex.lock t.lock;
    Fun.protect
      ~finally:(fun () -> Mutex.unlock t.lock)
      (fun () ->
        if t.closed then false
        else (Queue.clear t.chunks; Queue.add chunk t.chunks;
              Condition.broadcast t.not_full;
              Condition.signal t.not_empty; true))

  let pop t =
    Mutex.lock t.lock;
    Fun.protect
      ~finally:(fun () -> Mutex.unlock t.lock)
      (fun () ->
        if not (Queue.is_empty t.chunks) then (
          let chunk = Queue.take t.chunks in
          Condition.signal t.not_full;
          `Data chunk)
        else if t.closed then `Eof
        else `Wait)

  let close t =
    Mutex.lock t.lock;
    Fun.protect
      ~finally:(fun () -> Mutex.unlock t.lock)
      (fun () ->
        t.closed <- true;
        Condition.broadcast t.not_empty;
        Condition.broadcast t.not_full)

  let closed t =
    Mutex.lock t.lock;
    Fun.protect
      ~finally:(fun () -> Mutex.unlock t.lock)
      (fun () -> t.closed)

  let response t =
    let headers =
      H.Headers.add_value (H.Headers.Name.v "connection") "close"
        H.Headers.empty
    in
    H.Response.make ~headers H.Status.ok_200
      (H.Body.of_custom_content ~content_type:"text/event-stream" (Iomux_stream t))
end

let err_to_response = function
  | `Malformed e ->
      let reason = if e = "" then None else Some e in
      H.Response.empty ?reason H.Status.bad_request_400
  | `Too_large ->
      H.Response.empty H.Status.content_too_large_413
  | `Not_implemented e ->
      let reason = if e = "" then None else Some e in
      H.Response.empty ?reason H.Status.not_implemented_501
  | `Service ->
      H.Response.empty H.Status.server_error_500

(* Find the offset of the end of the header section ("\r\n\r\n").
 * Returns the index just past the second CRLF, or None if incomplete. *)
let find_header_end buf len =
  let rec loop i =
    if i + 3 >= len then None
    else if
      Bytes.get buf i = '\r' && Bytes.get buf (i + 1) = '\n'
      && Bytes.get buf (i + 2) = '\r' && Bytes.get buf (i + 3) = '\n'
    then Some (i + 4)
    else loop (i + 1)
  in
  loop 0

(* Scan CRLF positions in [buf.(0 .. hdr_end-1)]. The first CRLF ends the
 * request line; the rest end the header fields. Mirrors webs' blank-line
 * detection: a CRLF immediately following the previous CRLF marks the end of
 * the header section and is NOT included (it is the leading CRLF of the blank
 * line). *)
let scan_crlfs buf hdr_end =
  let rec loop i acc =
    if i >= hdr_end then List.rev acc
    else if Bytes.get buf i = '\r' && i + 1 < hdr_end && Bytes.get buf (i + 1) = '\n'
    then
      match acc with
      | prev :: _ when prev + 2 = i -> List.rev acc (* blank line, stop *)
      | _ -> loop (i + 2) (i :: acc)
    else loop (i + 1) acc
  in
  loop 0 []

(* Decode a request from a fully-buffered header section.
 * [body_start] is the offset where the body begins; [body_bytes] are the
 * already-available body bytes (may be a prefix). *)
(* Decode a chunked-transfer-encoded request body from [buf] in the range
 * [first .. last). Returns the concatenated body and the offset just past the
 * terminating "0\r\n\r\n", or [Need_more] if the buffer doesn't yet contain
 * the full body, or [Error] if the framing is malformed. *)
type chunk_result = [ `Need_more | `Error of string | `Done of string * int ]

let decode_chunked buf ~first ~last : chunk_result =
  let rec loop i acc : chunk_result =
    (* skip optional/leading CRLFs between chunks are part of framing *)
    if i + 1 > last then `Need_more
    else
      (* read chunk-size line: "HEX\r\n" *)
      let j = ref i in
      let ok = ref false in
      while !j <= last - 1 && not !ok do
        if Bytes.get buf !j = '\r' && !j + 1 <= last && Bytes.get buf (!j + 1) = '\n'
        then ok := true
        else incr j
      done;
      if not !ok then `Need_more
      else
        let size_line = Bytes.sub_string buf i (!j - i) in
        let size =
          try int_of_string ("0x" ^ String.trim size_line) with _ -> -1
        in
        if size < 0 then `Error "malformed chunk size"
        else if size = 0 then
          (* last-chunk: optional trailer + final CRLF; consume "0\r\n"
           * plus the trailing CRLF marking end of the chunk section. *)
          let end_i = !j + 2 in
          if end_i + 1 > last then `Need_more
          else if Bytes.get buf end_i = '\r' && Bytes.get buf (end_i + 1) = '\n'
          then `Done (String.concat "" (List.rev acc), end_i + 2)
          else `Need_more
        else
          let data_start = !j + 2 in
          let data_end = data_start + size in
          if data_end + 1 > last then `Need_more
          else if
            Bytes.get buf (data_end) = '\r'
            && Bytes.get buf (data_end + 1) = '\n'
          then
            let chunk = Bytes.sub_string buf data_start size in
            loop (data_end + 2) (chunk :: acc)
          else `Error "malformed chunk data"
  in
  loop first []

(* Decode a request from a buffered header section + available body bytes.
 * Returns:
 *   [`Ok (request, consumed)]  request parsed; [consumed] body bytes used
 *   [`Need_more]               not enough buffered bytes yet; keep reading
 *   [`Error e]                 malformed / too large
 * [body_start] is the offset of the body; [body_bytes] the already-buffered
 * body (a prefix when [`Need_more]). *)
let decode_request buf hdr_end body_start body_bytes max_body =
  let crlfs = scan_crlfs buf hdr_end in
  let req_crlf = List.hd crlfs in
  let method', raw_path, version =
    Webs_connector_codec.decode_request_line buf ~first:0 ~crlf:req_crlf
  in
  let headers = Webs_connector_codec.decode_headers buf ~crlfs in
  let path, query =
    match H.Path.and_query_string_of_request_target raw_path with
    | Ok v -> v
    | Error e -> failwith e
  in
  let make content consumed =
    ( H.Request.make ~headers ~path ~query ~version method' ~raw_path content,
      consumed )
  in
  match H.Headers.request_body_length headers with
  | Ok (`Length l) ->
      if l > max_body then `Error `Too_large
      else if String.length body_bytes < l then `Need_more
      else `Ok (make (if l = 0 then H.Body.empty else H.Body.of_string (String.sub body_bytes 0 l)) l)
  | Ok `Chunked ->
      (match decode_chunked buf ~first:body_start ~last:(body_start + String.length body_bytes) with
       | `Need_more -> `Need_more
       | `Error e -> `Error (`Malformed e)
       | `Done (body, consumed) ->
           let consumed = consumed - body_start in
           if String.length body > max_body then `Error `Too_large
           else `Ok (make (H.Body.of_string body) consumed))
  | Error _ -> `Ok (make H.Body.empty 0)

(* ---------------- connection + event loop ---------------- *)

type conn = {
  fd : Unix.file_descr;
  mutable input : Bytes.t;
  mutable input_len : int;
  mutable out : Bytes.t;
  mutable out_len : int;
  mutable out_pos : int;
  mutable requests : int;
  mutable header_deadline : float;
  mutable idle_deadline : float;
  mutable keep_alive : bool;
  mutable body_reader : Bw.Reader.t option;
  mutable body_stream : stream option;
  mutable chunked : bool;
  mutable body_done : bool;
  remote : string;
}

type config = {
  addr : string;
  port : int;
  backlog : int;
  max_header_bytes : int;
  max_body_bytes : int;
  max_keepalive : int;
  header_timeout_ms : int;
  idle_timeout_ms : int;
  handler : H.Request.t -> H.Response.t;
  stop : bool Atomic.t;
}

type t = {
  server_fd : Unix.file_descr;
  port : int;
  stop_requested : bool Atomic.t;
  thread : Thread.t;
}

let input_limit_exceeded conn cfg =
  match find_header_end conn.input conn.input_len with
  | None -> conn.input_len > cfg.max_header_bytes
  | Some header_end ->
      header_end > cfg.max_header_bytes
      || conn.input_len - header_end > cfg.max_body_bytes + cfg.max_header_bytes

let now_ms () = Unix.gettimeofday () *. 1000.0

let ensure_capacity conn extra =
  let needed = conn.input_len + extra in
  if needed > Bytes.length conn.input then (
    let next = ref (max 4096 (Bytes.length conn.input * 2)) in
    while !next < needed do next := !next * 2 done;
    let grown = Bytes.create !next in
    Bytes.blit conn.input 0 grown 0 conn.input_len;
    conn.input <- grown)

let ensure_out conn extra =
  let needed = conn.out_len + extra in
  if needed > Bytes.length conn.out then (
    let next = ref (max 4096 (Bytes.length conn.out * 2)) in
    while !next < needed do next := !next * 2 done;
    let grown = Bytes.create !next in
    Bytes.blit conn.out 0 grown 0 conn.out_len;
    conn.out <- grown)

let close_noerr fd = try Unix.close fd with _ -> ()
let would_block = function Unix.Unix_error ((EAGAIN | EWOULDBLOCK), _, _) -> true | _ -> false

let append_out conn b ~start ~len =
  ensure_out conn len;
  Bytes.blit b start conn.out conn.out_len len;
  conn.out_len <- conn.out_len + len

(* Pull the next slice from the response body reader and stage it for
 * writing. On EOF, append the chunked terminator (if chunked) and clear the
 * reader. Returns [true] if there is (or may still be) data to write. *)
let rec drain_body conn =
  match (conn.body_reader, conn.body_stream) with
  | None, None -> true
  | Some reader, None ->
      if conn.body_done then true
      else
        let slice = Bw.Reader.read reader in
        if Bw.Slice.equal slice Bw.Slice.eod then (
          conn.body_done <- true;
          conn.body_reader <- None;
          if conn.chunked then
            append_out conn (Bytes.unsafe_of_string "0\r\n\r\n") ~start:0 ~len:5;
          true)
        else
          let len = Bw.Slice.length slice in
          let data = Bw.Slice.to_string slice in
          if conn.chunked then (
            let chunk_head = Printf.sprintf "%x\r\n" len in
            append_out conn (Bytes.unsafe_of_string chunk_head) ~start:0
              ~len:(String.length chunk_head);
            append_out conn (Bytes.unsafe_of_string data) ~start:0 ~len;
            append_out conn (Bytes.unsafe_of_string "\r\n") ~start:0 ~len:2)
          else append_out conn (Bytes.unsafe_of_string data) ~start:0 ~len;
          true
  | None, Some stream -> (
      match Stream.pop stream with
      | `Wait -> true
      | `Eof ->
          conn.body_stream <- None;
          conn.body_done <- true;
          if conn.chunked then
            append_out conn (Bytes.unsafe_of_string "0\r\n\r\n") ~start:0 ~len:5;
          true
      | `Data data ->
          if data <> "" then begin
            let bytes = Bytes.unsafe_of_string data in
            let len = String.length data in
            if conn.chunked then begin
              let chunk_head = Printf.sprintf "%x\r\n" len in
              append_out conn (Bytes.unsafe_of_string chunk_head) ~start:0
                ~len:(String.length chunk_head);
              append_out conn bytes ~start:0 ~len;
              append_out conn (Bytes.unsafe_of_string "\r\n") ~start:0 ~len:2
            end else append_out conn bytes ~start:0 ~len
          end;
          true)
  | Some _, Some _ ->
      (* Rendering installs exactly one body representation. If a future
         adapter violates that invariant, retain the byte reader and discard
         the competing stream rather than spinning forever. *)
      conn.body_stream <- None;
      drain_body conn

let body_active conn = conn.body_reader <> None || conn.body_stream <> None

let drain_out conn fd =
  let rec loop () =
    if conn.out_pos >= conn.out_len then
      (* output buffer empty: stream more body, or we're finished *)
      if not (body_active conn) then true
      else (
        ignore (drain_body conn);
        if conn.out_pos >= conn.out_len then
          (* body produced nothing new (e.g. finished) -> done *)
          not (body_active conn)
        else loop ())
    else
      try
        let n = Unix.single_write conn.fd conn.out conn.out_pos (conn.out_len - conn.out_pos) in
        conn.out_pos <- conn.out_pos + n;
        if conn.out_pos >= conn.out_len then loop () else false
      with
      | e when would_block e -> false
      | Unix.Unix_error (Unix.EPIPE, _, _) -> raise Exit
      | Unix.Unix_error (Unix.ECONNRESET, _, _) -> raise Exit
  in
  loop ()

(* Render a webs response into [conn]'s output buffer.
 *
 * The head is appended to [conn.out] immediately. The body is streamed
 * afterwards: if the body has a known length we send a [Content-Length]
 * header and write the bytes as they become available; otherwise we use
 * [Transfer-Encoding: chunked] and stream the body via its [Bytes.Reader].
 *
 * Body streaming is completed by [drain_body], driven from the write loop. *)
let render_response ?(without_body = false) conn r =
  let status = H.Response.status r in
  let reason = H.Response.reason r in
  let body = H.Response.body r in
  let hs = H.Headers.for_connector (H.Response.headers r) body in
  let known_len = H.Body.content_length body in
  let hs =
    match known_len with
    | Some _ -> hs
    | None ->
        H.Headers.add_value
          (H.Headers.Name.v "transfer-encoding")
          "chunked" hs
  in
  let head =
    Webs_connector_codec.encode_http11_response_head status ~reason hs
  in
  conn.out_len <- 0;
  conn.out_pos <- 0;
  append_out conn (Bytes.unsafe_of_string head) ~start:0 ~len:(String.length head);
  if without_body then (
    conn.chunked <- false;
    conn.body_done <- true;
    conn.body_reader <- None;
    conn.body_stream <- None)
  else (
    conn.chunked <- (known_len = None);
    conn.body_done <- false;
    match H.Body.content body with
    | H.Body.Custom (Iomux_stream stream) ->
        conn.body_reader <- None;
        conn.body_stream <- Some stream
    | _ -> (
        match H.Body.to_bytes_reader body with
        | Error _ -> conn.body_reader <- None; conn.body_stream <- None
        | Ok reader ->
            conn.body_reader <- Some reader;
            conn.body_stream <- None))

(* Try to parse and handle one request from the buffered input.
 * Returns [true] if a response was produced (and out buffer filled). *)
let try_handle conn cfg =
  if conn.out_len > conn.out_pos then false (* still writing previous response *)
  else
    match find_header_end conn.input conn.input_len with
    | None -> false
    | Some hdr_end ->
        let body_start = hdr_end in
        let avail_body = conn.input_len - body_start in
        let body_bytes = Bytes.sub_string conn.input body_start avail_body in
        let parsed =
          if hdr_end > cfg.max_header_bytes then `Error `Too_large
          else
            try decode_request conn.input hdr_end body_start body_bytes cfg.max_body_bytes with
            | Failure message -> `Error (`Malformed message)
            | Invalid_argument message -> `Error (`Malformed message)
        in
        (match parsed with
         | `Need_more -> false (* wait for more bytes; keep buffered input *)
         | `Error e ->
             let resp = err_to_response e in
             conn.keep_alive <- false;
             render_response conn resp;
             true
         | `Ok (request, consumed) ->
             let resp = try cfg.handler request with _ -> err_to_response `Service in
             (* keep-alive decision: close if either side said close *)
             let conn_close =
               match H.Headers.find ~lowervalue:true H.Headers.connection (H.Request.headers request) with
               | Some v -> String.lowercase_ascii v = "close"
               | None -> false
             in
             let resp_close =
               match H.Headers.find ~lowervalue:true H.Headers.connection (H.Response.headers resp) with
               | Some v -> String.lowercase_ascii v = "close"
               | None -> false
             in
             conn.requests <- conn.requests + 1;
             let keep =
               not conn_close && not resp_close
               && conn.requests < cfg.max_keepalive
             in
             conn.keep_alive <- keep;
             (* compact unconsumed body tail for next pipelined request *)
             let tail = conn.input_len - hdr_end - consumed in
             if tail > 0 then
               Bytes.blit conn.input (hdr_end + consumed) conn.input 0 tail;
             conn.input_len <- max 0 tail;
             (match find_header_end conn.input conn.input_len with
              | None -> conn.header_deadline <- now_ms () +. float cfg.header_timeout_ms
              | Some _ -> conn.header_deadline <- infinity);
             render_response ~without_body:(H.Request.method' request = `HEAD) conn resp;
             true)

let read_ready conn cfg =
  let read_buf = Bytes.create 65536 in
  let too_large = ref false in
  let rec loop () =
    if !too_large then ()
    else try
      let n = Unix.read conn.fd read_buf 0 (Bytes.length read_buf) in
      if n = 0 then (
        if conn.out_len > conn.out_pos then ()
        else conn.keep_alive <- false)
      else (
        ensure_capacity conn n;
        Bytes.blit read_buf 0 conn.input conn.input_len n;
        conn.input_len <- conn.input_len + n;
        conn.idle_deadline <- now_ms () +. float cfg.idle_timeout_ms;
        (match find_header_end conn.input conn.input_len with
         | Some _ -> conn.header_deadline <- infinity
         | None -> ());
        if input_limit_exceeded conn cfg then too_large := true else loop ())
    with
    | e when would_block e -> ()
    | Unix.Unix_error (Unix.EINTR, _, _) -> loop ()
    | Unix.Unix_error (Unix.ECONNRESET, _, _) -> conn.keep_alive <- false
    | Unix.Unix_error (Unix.EPIPE, _, _) -> conn.keep_alive <- false
  in
  loop ();
  (* Drain all complete requests currently buffered (HTTP pipelining and
   * keep-alive). try_handle returns false once no full request is available
   * or the output buffer is still being written. *)
  if !too_large then (
    conn.keep_alive <- false;
    render_response conn (err_to_response `Too_large))
  else
    let rec handle_all () =
      if try_handle conn cfg then handle_all () else ()
    in
    handle_all ()

let make_conn fd remote cfg =
  {
    fd;
    input = Bytes.create 4096;
    input_len = 0;
    out = Bytes.create 4096;
    out_len = 0;
    out_pos = 0;
    requests = 0;
    header_deadline = now_ms () +. float cfg.header_timeout_ms;
    idle_deadline = now_ms () +. float cfg.idle_timeout_ms;
    keep_alive = true;
    body_reader = None;
    body_stream = None;
    chunked = false;
    body_done = false;
    remote;
  }

let string_of_addr = function
  | Unix.ADDR_UNIX p -> "unix:" ^ p
  | Unix.ADDR_INET (a, p) -> Printf.sprintf "%s:%d" (Unix.string_of_inet_addr a) p

let listen cfg server_fd =
  let conns = Hashtbl.create 1024 in
  let rec loop () =
    if Atomic.get cfg.stop then ()
    else
    let now = now_ms () in
    (* drop timed-out connections *)
    Hashtbl.iter
      (fun _ c ->
        let header_expired =
          c.body_reader = None && c.body_stream = None
          && now > c.header_deadline
        in
        if header_expired || now > c.idle_deadline then (
          Hashtbl.remove conns c.fd;
          Option.iter Stream.close c.body_stream;
          close_noerr c.fd))
      conns;
    (* A live stream has no readable socket event when its producer pushes a
       chunk. Stage queued data before polling so the next poll observes
       [pollout]. The bounded poll timeout gives producers a finite wake-up
       latency even though Iomux has no cross-thread wake fd yet. *)
    Hashtbl.iter
      (fun _ c ->
        if c.out_len <= c.out_pos && c.body_stream <> None then begin
          ignore (drain_body c);
          if c.body_stream <> None then
            c.idle_deadline <- now +. float cfg.idle_timeout_ms
        end)
      conns;
    let arr = Hashtbl.fold (fun _ c acc -> c :: acc) conns [] |> Array.of_list in
    let poller = Iomux.Poll.create ~maxfds:(Array.length arr + 1) () in
    Iomux.Poll.set_index poller 0 server_fd Iomux.Poll.Flags.pollin;
    Array.iteri
      (fun i c ->
        let flags =
          if c.out_len > c.out_pos then
            Iomux.Poll.Flags.(pollin + pollout)
          else Iomux.Poll.Flags.pollin
        in
        Iomux.Poll.set_index poller (i + 1) c.fd flags)
      arr;
    let timeout =
      let has_active_stream =
        Hashtbl.fold (fun _ c acc -> acc || c.body_stream <> None) conns false
      in
      if has_active_stream then 1
      else if Array.length arr = 0 then 100
      else
        let min_d =
          Array.fold_left
            (fun m c -> min m (min c.header_deadline c.idle_deadline))
            infinity arr
        in
        let ms = int_of_float (min_d -. now) in
        if ms < 1 then 1 else if ms > 1000 then 1000 else ms
    in
    let ready = Iomux.Poll.poll poller (Array.length arr + 1) (Iomux.Poll.Milliseconds timeout) in
    if ready > 0 then
      Iomux.Poll.iter_ready poller ready (fun idx _fd events ->
        if idx = 0 then (
          let rec accept_loop () =
            try
              let fd, a = Unix.accept ~cloexec:true server_fd in
              if Hashtbl.length conns >= 4096 then close_noerr fd
              else (
                Unix.set_nonblock fd;
                Unix.setsockopt fd Unix.TCP_NODELAY true;
                let c = make_conn fd (string_of_addr a) cfg in
                Hashtbl.replace conns fd c);
              accept_loop ()
            with
            | e when would_block e -> ()
            | Unix.Unix_error (Unix.EINTR, _, _) -> accept_loop ()
          in
          accept_loop ())
        else if idx <= Array.length arr then (
          let c = arr.(idx - 1) in
          let drop () =
            Hashtbl.remove conns c.fd;
            Option.iter Stream.close c.body_stream;
            close_noerr c.fd
          in
          (* Pump: write any pending output, then handle the next buffered
           * request (pipelining / keep-alive). Stops when the output buffer
           * still has unwritten bytes (waits for pollout) or no full request
           * is buffered. *)
          let rec pump () =
            if c.out_len > c.out_pos then ()
            else if try_handle c cfg then (
              let _ = drain_out c c.fd in
              pump ())
            else if not c.keep_alive && not (body_active c) then drop ()
          in
          (try
             if c.out_len > c.out_pos
                && Iomux.Poll.Flags.mem events Iomux.Poll.Flags.pollout
             then (
               if drain_out c c.fd && not c.keep_alive then (drop (); ())
               else pump ())
             else if Iomux.Poll.Flags.mem events Iomux.Poll.Flags.pollin then (
               read_ready c cfg;
               let _ = drain_out c c.fd in
               pump ())
             else pump ()
           with Exit -> drop ());
          if Iomux.Poll.Flags.(mem events pollhup || mem events pollerr || mem events pollnval)
          then drop ()));
    loop ()
  in
  (try loop () with Unix.Unix_error _ -> ());
  Hashtbl.iter
    (fun _ c ->
      Option.iter Stream.close c.body_stream;
      close_noerr c.fd)
    conns;
  Hashtbl.clear conns

let start ?(backlog = 1024) ?(max_header_bytes = 16 * 1024)
    ?(max_body_bytes = 5 * 1024 * 1024) ?(max_keepalive = 100)
    ?(header_timeout_ms = 10_000) ?(idle_timeout_ms = 30_000) ~host ~port handler =
  Sys.set_signal Sys.sigpipe Sys.Signal_ignore;
  let server_fd = Unix.socket Unix.PF_INET Unix.SOCK_STREAM 0 in
  try
    Unix.setsockopt server_fd Unix.SO_REUSEADDR true;
    Unix.set_nonblock server_fd;
    let addr = Unix.ADDR_INET (Unix.inet_addr_of_string host, port) in
    Unix.bind server_fd addr;
    Unix.listen server_fd backlog;
    let assigned_port =
      match Unix.getsockname server_fd with
      | Unix.ADDR_INET (_, port) -> port
      | Unix.ADDR_UNIX _ -> assert false
    in
    let stop_requested = Atomic.make false in
    let cfg =
      {
        addr = host;
        port = assigned_port;
        backlog;
        max_header_bytes;
        max_body_bytes;
        max_keepalive;
        header_timeout_ms;
        idle_timeout_ms;
        handler;
        stop = stop_requested;
      }
    in
    let thread = Thread.create (fun () -> listen cfg server_fd) () in
    Ok { server_fd; port = assigned_port; stop_requested; thread }
  with
  | Invalid_argument message ->
      close_noerr server_fd;
      Error message
  | Unix.Unix_error (error, _, message) ->
      close_noerr server_fd;
      Error (Unix.error_message error ^ ": " ^ message)

let port t = t.port

let stop t =
  if Atomic.compare_and_set t.stop_requested false true then
    Thread.join t.thread;
  close_noerr t.server_fd

let serve ?(addr = "127.0.0.1") ?(port = 8080) ?(backlog = 1024)
    ?(max_header_bytes = 16384) ?(max_body_bytes = 5 * 1024 * 1024)
    ?(max_keepalive = 100) ?(header_timeout_ms = 10000) ?(idle_timeout_ms = 30000)
    handler =
  let cfg =
    {
      addr;
      port;
      backlog;
      max_header_bytes;
      max_body_bytes;
      max_keepalive;
      header_timeout_ms;
      idle_timeout_ms;
      handler;
      stop = Atomic.make false;
    }
  in
  let server_fd = Unix.socket Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.setsockopt server_fd Unix.SO_REUSEADDR true;
  Unix.set_nonblock server_fd;
  Unix.bind server_fd (Unix.ADDR_INET (Unix.inet_addr_of_string addr, port));
  Unix.listen server_fd backlog;
  Printf.printf "webs_iomux gateway on %s:%d\n%!" addr port;
  listen cfg server_fd
