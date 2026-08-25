#!/usr/bin/env python3
"""Synthetic Poisson arrival benchmark for measuring SRPT continuous serving queues."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from racebench.http import _open_connection, _stream_request


@dataclass
class PoissonRequestResult:
    index: int
    arrival_offset_s: float
    actual_start_s: float
    ttft_ms: float | None
    tpot_ms: float | None
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int
    output_text: str
    output_tokens: list[int]
    succeeded: bool
    error: str | None


def simulate_poisson_schedule(num_requests: int, rate_lambda: float) -> list[float]:
    """Generate inter-arrival offsets based on an exponential distribution."""
    offsets = [0.0]
    current_time = 0.0
    for _ in range(num_requests - 1):
        inter_arrival = -math.log(max(1e-9, random.random())) / rate_lambda
        current_time += inter_arrival
        offsets.append(current_time)
    return offsets


def run_single_request(
    index: int,
    offset_s: float,
    start_benchmark_time: float,
    base_url: str,
    prompt_messages: list[dict[str, str]],
    max_tokens: int,
) -> PoissonRequestResult:
    target_time = start_benchmark_time + offset_s
    sleep_duration = target_time - time.perf_counter()
    if sleep_duration > 0:
        time.sleep(sleep_duration)

    actual_start = time.perf_counter()
    body = {
        "model": "LiquidAI/LFM2.5-350M",
        "messages": prompt_messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    try:
        connection, path = _open_connection(base_url, timeout_s=30.0)
        res = _stream_request(
            connection=connection,
            path=path,
            api_key="",
            body=body,
            request_id=f"poisson-{index}",
            conversation=index,
            turn=0,
            scheduled_at=actual_start,
            scheduled_offset_s=offset_s,
        )
        connection.close()
        return PoissonRequestResult(
            index=index,
            arrival_offset_s=offset_s,
            actual_start_s=actual_start - start_benchmark_time,
            ttft_ms=res.ttft_ms,
            tpot_ms=res.tpot_ms,
            latency_ms=res.latency_ms,
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
            cached_prompt_tokens=res.cached_prompt_tokens,
            output_text=res.output_text,
            output_tokens=res.output_token_ids,
            succeeded=res.succeeded,
            error=res.error,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - actual_start) * 1000.0
        return PoissonRequestResult(
            index=index,
            arrival_offset_s=offset_s,
            actual_start_s=actual_start - start_benchmark_time,
            ttft_ms=None,
            tpot_ms=None,
            latency_ms=latency_ms,
            prompt_tokens=0,
            completion_tokens=0,
            cached_prompt_tokens=0,
            output_text="",
            output_tokens=[],
            succeeded=False,
            error=str(exc),
        )


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (len(sorted_v) - 1) * q
    low = int(math.floor(idx))
    high = int(math.ceil(idx))
    if low == high:
        return sorted_v[low]
    return sorted_v[low] * (high - idx) + sorted_v[high] * (idx - low)


def main() -> None:
    parser = argparse.ArgumentParser(description="Poisson arrival benchmark")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=12, help="Total requests")
    parser.add_argument("--rate", type=float, default=5.0, help="Poisson arrival rate lambda (req/s)")
    parser.add_argument("--concurrency", type=int, default=4, help="Client threadpool concurrency")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    random.seed(42)
    offsets = simulate_poisson_schedule(args.requests, args.rate)

    test_prompts = [
        [{"role": "user", "content": "Hello!"}],
        [{"role": "user", "content": "Explain the concept of priority scheduling in operating systems in one sentence."}],
        [{"role": "user", "content": "Count from 1 to 5."}],
        [{"role": "user", "content": "What is the capital of France?"}],
    ]

    print(f"Starting Poisson benchmark: lambda={args.rate} req/s, requests={args.requests}, concurrency={args.concurrency}")
    start_time = time.perf_counter()
    results: list[PoissonRequestResult] = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = []
        for i, offset in enumerate(offsets):
            prompt = test_prompts[i % len(test_prompts)]
            max_toks = 4 if i % 2 == 0 else 8
            fut = pool.submit(
                run_single_request,
                i,
                offset,
                start_time,
                args.base_url,
                prompt,
                max_toks,
            )
            futures.append(fut)

        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)

    results.sort(key=lambda r: r.index)
    total_wall_s = time.perf_counter() - start_time
    successful = [r for r in results if r.succeeded]
    ttfts = [r.ttft_ms for r in successful if r.ttft_ms is not None]
    tpots = [r.tpot_ms for r in successful if r.tpot_ms is not None]

    summary: dict[str, Any] = {
        "total_requests": args.requests,
        "successful_requests": len(successful),
        "rate_lambda": args.rate,
        "concurrency": args.concurrency,
        "wall_time_s": total_wall_s,
        "ttft_ms": {
            "p50": percentile(ttfts, 0.50),
            "p95": percentile(ttfts, 0.95),
            "p99": percentile(ttfts, 0.99),
            "mean": sum(ttfts) / len(ttfts) if ttfts else 0.0,
        },
        "tpot_ms": {
            "p50": percentile(tpots, 0.50),
            "p95": percentile(tpots, 0.95),
            "p99": percentile(tpots, 0.99),
            "mean": sum(tpots) / len(tpots) if tpots else 0.0,
        },
    }

    print("\n--- Benchmark Summary ---")
    print(f"Success: {len(successful)}/{args.requests}")
    print(f"Wall Time: {total_wall_s:.3f}s")
    if ttfts:
        print(f"TTFT (ms): P50={summary['ttft_ms']['p50']:.2f}, P95={summary['ttft_ms']['p95']:.2f}, P99={summary['ttft_ms']['p99']:.2f}")
    if tpots:
        print(f"TPOT (ms): P50={summary['tpot_ms']['p50']:.2f}, P95={summary['tpot_ms']['p95']:.2f}, P99={summary['tpot_ms']['p99']:.2f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "summary": summary,
            "results": [asdict(r) for r in results],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
