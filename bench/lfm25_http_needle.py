"""Run the natural needle matrix against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from racebench.http import run_chat_request
from racebench.needle import (
    DEFAULT_LENGTHS,
    DEFAULT_POSITIONS,
    EXPECTED,
    archive_corpus,
    chat_token_ids,
    evaluate_response,
    exact_natural_messages,
)


def _integers(raw: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integers") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def _token_ids(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(
            int(value.strip()) for value in raw.split(",") if value.strip()
        )
    except ValueError as exc:
        raise ValueError("--expected-token-ids must be comma-separated integers") from exc
    if not values or any(value < 0 for value in values):
        raise ValueError("--expected-token-ids must contain non-negative integers")
    return values


def token_parity(
    actual: list[int], expected: tuple[int, ...] | None
) -> bool | None:
    return None if expected is None else tuple(actual) == expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument("--tokenizer")
    parser.add_argument(
        "--lengths", default=",".join(str(value) for value in DEFAULT_LENGTHS)
    )
    parser.add_argument(
        "--positions", default=",".join(str(value) for value in DEFAULT_POSITIONS)
    )
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument(
        "--expected-token-ids",
        help="optional comma-separated eager token sequence for exact parity reporting",
    )
    parser.add_argument(
        "--allow-eos",
        action="store_true",
        help="allow end-token termination instead of pinning the output count",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    lengths = _integers(args.lengths, name="--lengths")
    positions = _integers(args.positions, name="--positions")
    if any(position >= 100 for position in positions):
        raise ValueError("--positions must be between 1 and 99")
    if args.max_tokens < 1:
        raise ValueError("--max-tokens must be positive")
    expected_token_ids = (
        None
        if args.expected_token_ids is None
        else _token_ids(args.expected_token_ids)
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    corpus = archive_corpus()
    generation_request: dict[str, Any] = {"temperature": 0.0}
    if not args.allow_eos:
        generation_request.update(
            {"min_tokens": args.max_tokens, "ignore_eos": True}
        )
    rows: list[dict[str, Any]] = []
    for target_tokens in lengths:
        for needle_percent in positions:
            messages = exact_natural_messages(
                tokenizer, target_tokens, needle_percent, corpus
            )
            constructed_tokens = len(chat_token_ids(tokenizer, messages))
            result = run_chat_request(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                timeout_s=args.timeout,
                request_id=f"needle-{target_tokens}-{needle_percent}",
                request=generation_request,
            )
            evaluation = evaluate_response(result.output_text)
            row = {
                "target_prompt_tokens": target_tokens,
                "needle_percent": needle_percent,
                "constructed_prompt_tokens": constructed_tokens,
                "success": result.succeeded,
                "retrieved": evaluation.retrieved,
                "exact_response": evaluation.exact_response,
                "ttft_ms": result.ttft_ms,
                "tpot_ms": result.tpot_ms,
                "latency_ms": result.latency_ms,
                "completion_tokens": result.completion_tokens,
                "prompt_tokens": result.prompt_tokens,
                "cached_prompt_tokens": result.cached_prompt_tokens,
                "output_token_ids": result.output_token_ids,
                "output_token_ids_sha256": result.output_token_ids_sha256,
                "token_parity": token_parity(
                    result.output_token_ids, expected_token_ids
                ),
                "text": result.output_text,
                "text_sha256": hashlib.sha256(
                    result.output_text.encode("utf-8")
                ).hexdigest(),
                "error": result.error,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
    report = {
        "schema_version": 1,
        "endpoint": args.base_url,
        "model": args.model,
        "expected": EXPECTED,
        "lengths": list(lengths),
        "positions": list(positions),
        "max_tokens": args.max_tokens,
        "fixed_output_tokens": not args.allow_eos,
        "expected_output_token_ids": (
            None if expected_token_ids is None else list(expected_token_ids)
        ),
        "successful": sum(row["success"] for row in rows),
        "retrieved": sum(row["retrieved"] for row in rows),
        "exact_responses": sum(row["exact_response"] for row in rows),
        "token_parity": sum(row["token_parity"] is True for row in rows),
        "token_parity_total": 0 if expected_token_ids is None else len(rows),
        "total": len(rows),
        "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    try:
        report = run(args)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    summary = (
        f"needle retrieval: {report['retrieved']}/{report['total']}; "
        f"exact responses: {report['exact_responses']}/{report['total']}"
    )
    if report["token_parity_total"]:
        summary += (
            f"; token parity: {report['token_parity']}/"
            f"{report['token_parity_total']}"
        )
    print(summary)
    raise SystemExit(0 if report["successful"] == report["total"] else 2)


if __name__ == "__main__":
    main()
