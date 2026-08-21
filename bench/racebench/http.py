"""OpenAI-compatible racebench runner used by the reference-style CLI.

The LFM2.5 MPS benchsuite uses the in-process adapter in
``lfm25_benchsuite.py``.  This module keeps the adjacent viettel-ai-race
endpoint contract available for local server experiments and contract tests.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import ssl
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .benchmark import RequestResult, summarize
from .score import request_score
from .trace import WorkloadTrace


class HttpBenchmarkError(RuntimeError):
    """Raised for invalid endpoint configuration."""


def _open_connection(
    base_url: str, timeout_s: float
) -> tuple[http.client.HTTPConnection, str]:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HttpBenchmarkError(f"invalid base URL: {base_url}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    connection_type = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    kwargs: dict[str, Any] = {
        "host": parsed.hostname,
        "port": port,
        "timeout": timeout_s,
    }
    if parsed.scheme == "https":
        kwargs["context"] = ssl.create_default_context()
    prefix = parsed.path.rstrip("/")
    return connection_type(**kwargs), f"{prefix}/v1/chat/completions"


def _stream_request(
    *,
    connection: http.client.HTTPConnection,
    path: str,
    api_key: str,
    body: dict[str, Any],
    request_id: str,
    conversation: int,
    turn: int,
    scheduled_at: float,
    scheduled_offset_s: float,
) -> RequestResult:
    started_at = time.perf_counter()
    expected_tokens = int(body["max_tokens"])
    status: int | None = None
    first_token_at: float | None = None
    last_token_at: float | None = None
    completion_tokens = 0
    prompt_tokens = 0
    cached_prompt_tokens = 0
    content_events = 0
    output_parts: list[str] = []
    error: str | None = None
    try:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Connection": "keep-alive",
            "X-Request-Id": request_id,
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        connection.request("POST", path, body=json.dumps(body), headers=headers)
        response = connection.getresponse()
        status = response.status
        if status != 200:
            payload = response.read(8192).decode("utf-8", errors="replace")
            raise HttpBenchmarkError(f"HTTP {status}: {payload}")
        while True:
            raw_line = response.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage") or {}
            if usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])
            if usage.get("prompt_tokens") is not None:
                prompt_tokens = int(usage["prompt_tokens"])
                details = usage.get("prompt_tokens_details") or {}
                cached_prompt_tokens = int(details.get("cached_tokens") or 0)
            choices = event.get("choices") or []
            if not choices:
                continue
            content = (choices[0].get("delta") or {}).get("content")
            if not content:
                continue
            observed_at = time.perf_counter()
            first_token_at = first_token_at or observed_at
            last_token_at = observed_at
            content_events += 1
            output_parts.append(str(content))
    except Exception as exc:  # one failed request must not abort the trace
        error = f"{type(exc).__name__}: {exc}"
        connection.close()
    ended_at = time.perf_counter()
    if completion_tokens == 0:
        completion_tokens = content_events
    succeeded = error is None and first_token_at is not None and completion_tokens > 0
    ttft_ms = (first_token_at - started_at) * 1000.0 if first_token_at else None
    if succeeded and completion_tokens > 1 and last_token_at is not None:
        tpot_ms = (last_token_at - first_token_at) * 1000.0 / (completion_tokens - 1)
    elif succeeded:
        tpot_ms = 0.0
    else:
        tpot_ms = None
    return RequestResult(
        request_id=request_id,
        conversation=conversation,
        turn=turn,
        scheduled_offset_s=max(0.0, scheduled_offset_s),
        queue_delay_ms=max(0.0, (started_at - scheduled_at) * 1000.0),
        http_status=status,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        latency_ms=(ended_at - started_at) * 1000.0,
        completion_tokens=completion_tokens,
        expected_completion_tokens=expected_tokens,
        output_text="".join(output_parts),
        succeeded=succeeded,
        error=error,
        score=request_score(
            ttft_ms,
            tpot_ms,
            completion_tokens=completion_tokens,
            succeeded=succeeded,
        ),
        prompt_tokens=prompt_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
    )


def _skipped_result(
    trace: WorkloadTrace, conversation: int, turn: int, trace_started: float
) -> RequestResult:
    return RequestResult(
        request_id=f"c{conversation:04d}-t{turn:03d}",
        conversation=conversation,
        turn=turn,
        scheduled_offset_s=time.perf_counter() - trace_started,
        queue_delay_ms=0.0,
        http_status=None,
        ttft_ms=None,
        tpot_ms=None,
        latency_ms=0.0,
        completion_tokens=0,
        expected_completion_tokens=trace.output_tokens(conversation, turn),
        output_text="",
        succeeded=False,
        error="dependency_failed: an earlier turn in this conversation failed",
        score=0.0,
    )


async def run_trace(
    trace: WorkloadTrace,
    *,
    base_url: str,
    api_key: str = "",
    timeout_s: float = 120.0,
    max_workers: int | None = None,
) -> tuple[list[RequestResult], float]:
    """Run conversations concurrently and turns within each serially."""
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be positive")
    trace_started = time.perf_counter()
    loop = asyncio.get_running_loop()

    async def run_conversation(
        conversation: int, request_pool: ThreadPoolExecutor
    ) -> list[RequestResult]:
        messages: list[dict[str, str]] = []
        if trace.shared_system_prefix:
            messages.append({"role": "system", "content": trace.shared_system_prefix})
        connection, path = _open_connection(base_url, timeout_s)
        initial_offset = trace.arrival.offset_for(conversation)
        await asyncio.sleep(
            max(0.0, trace_started + initial_offset - time.perf_counter())
        )
        results: list[RequestResult] = []
        for turn, user_text in enumerate(trace.user_messages[conversation]):
            if turn > 0 and trace.arrival.think_time_s:
                await asyncio.sleep(trace.arrival.think_time_s)
            prefix = trace.conversation_prefixes[conversation] if turn == 0 else ""
            messages.append({"role": "user", "content": f"{prefix}{user_text}"})
            output_tokens = trace.output_tokens(conversation, turn)
            body: dict[str, Any] = {
                "model": trace.model,
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": output_tokens,
                "min_tokens": output_tokens,
                "ignore_eos": True,
                **trace.request,
            }
            scheduled_at = (
                trace_started + initial_offset
                if turn == 0
                else time.perf_counter()
            )
            result = await loop.run_in_executor(
                request_pool,
                partial(
                    _stream_request,
                    connection=connection,
                    path=path,
                    api_key=api_key,
                    body=body,
                    request_id=f"c{conversation:04d}-t{turn:03d}",
                    conversation=conversation,
                    turn=turn,
                    scheduled_at=scheduled_at,
                    scheduled_offset_s=scheduled_at - trace_started,
                ),
            )
            results.append(result)
            if not result.succeeded:
                results.extend(
                    _skipped_result(trace, conversation, skipped, trace_started)
                    for skipped in range(turn + 1, trace.user_turns_per_conversation)
                )
                break
            messages.append({"role": "assistant", "content": result.output_text})
        connection.close()
        return results

    worker_count = max_workers or trace.num_conversations
    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="racebench-http"
    ) as request_pool:
        grouped = await asyncio.gather(
            *(run_conversation(index, request_pool) for index in range(trace.num_conversations))
        )
    results = sorted(
        (result for group in grouped for result in group),
        key=lambda result: (result.conversation, result.turn),
    )
    return results, time.perf_counter() - trace_started


def write_report(
    path: str | Path,
    *,
    trace: WorkloadTrace,
    base_url: str,
    results: list[RequestResult],
    wall_time_s: float,
) -> dict[str, Any]:
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "trace": {
            "name": trace.name,
            "model": trace.model,
            "num_conversations": trace.num_conversations,
            "user_turns_per_conversation": trace.user_turns_per_conversation,
            "total_requests": trace.total_requests,
            "declared_tokens": trace.declared_tokens,
        },
        "endpoint": base_url,
        "wall_time_s": wall_time_s,
        "summary": summarize(results),
        "requests": [asdict(result) for result in results],
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
