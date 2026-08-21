"""Report schema and summaries for local in-process benchmark adapters."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping

from .score import effective_request_score
from .trace import WorkloadTrace


@dataclass
class RequestResult:
    request_id: str
    conversation: int
    turn: int
    scheduled_offset_s: float
    queue_delay_ms: float
    http_status: int | None
    ttft_ms: float | None
    tpot_ms: float | None
    latency_ms: float
    completion_tokens: int
    expected_completion_tokens: int
    output_text: str
    succeeded: bool
    error: str | None
    score: float
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    output_token_ids: list[int] = field(default_factory=list)
    output_token_ids_sha256: str = ""


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "mean": mean(values),
        "median": median(values),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _summary_core(results: list[RequestResult]) -> dict[str, Any]:
    successful = [result for result in results if result.succeeded]
    ttfts = [result.ttft_ms for result in successful if result.ttft_ms is not None]
    tpots = [result.tpot_ms for result in successful if result.tpot_ms is not None]
    scores = [result.score for result in results]
    token_mismatches = sum(
        result.completion_tokens != result.expected_completion_tokens
        for result in successful
    )
    prompt_tokens = sum(result.prompt_tokens for result in successful)
    cached_prompt_tokens = sum(result.cached_prompt_tokens for result in successful)
    return {
        "requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "output_token_mismatches": token_mismatches,
        "ers": effective_request_score(scores),
        "ttft_ms": _distribution(ttfts),
        "tpot_ms": _distribution(tpots),
        "request_score": _distribution(scores),
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "cached_prompt_token_ratio": (
            cached_prompt_tokens / prompt_tokens if prompt_tokens else None
        ),
    }


def summarize(results: list[RequestResult]) -> dict[str, Any]:
    if not results:
        raise ValueError("cannot summarize an empty result set")
    summary = _summary_core(results)
    summary["by_turn"] = {
        str(turn): _summary_core([result for result in results if result.turn == turn])
        for turn in sorted({result.turn for result in results})
    }
    return summary


def engine_pass(
    candidates: Mapping[str, Mapping[str, Any]],
    correctness: Mapping[str, Any],
    *,
    expected_requests: int | None = None,
) -> bool:
    """Return whether the execution engine completed its measured contract.

    Needle retrieval is a separate semantic observation. The engine pass is
    limited to successful warmup/scored requests, pinned output counts, and
    exact fixed-input logits.
    """
    if not correctness.get("exact") or not candidates:
        return False
    for candidate in candidates.values():
        for phase in ("warmup", "scored"):
            summary = candidate.get(phase)
            if not isinstance(summary, Mapping):
                return False
            if summary.get("failed_requests") != 0:
                return False
            if summary.get("successful_requests") != summary.get("requests"):
                return False
            if summary.get("output_token_mismatches") != 0:
                return False
            if expected_requests is not None and summary.get("requests") != expected_requests:
                return False
    return True


def write_report(
    path: str | Path,
    *,
    trace: WorkloadTrace,
    candidate: str,
    target: str,
    results: list[RequestResult],
    wall_time_s: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "candidate": candidate,
        "target": target,
        "trace": {
            "name": trace.name,
            "model": trace.model,
            "num_conversations": trace.num_conversations,
            "user_turns_per_conversation": trace.user_turns_per_conversation,
            "total_requests": trace.total_requests,
            "declared_tokens": trace.declared_tokens,
        },
        "wall_time_s": wall_time_s,
        "summary": summarize(results),
        "requests": [asdict(result) for result in results],
    }
    if metadata:
        report["metadata"] = metadata
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
