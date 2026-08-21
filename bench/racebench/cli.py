"""Reference-style CLI for trace validation, HTTP runs, and ERS reporting."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .benchmark import RequestResult, summarize
from .http import run_trace, write_report
from .score import accuracy_factor, final_score, request_score
from .trace import WorkloadTrace, validate_warmup_policy, validate_warmup_traces


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"requests: {summary['successful_requests']}/{summary['requests']} successful")
    print(f"ERS:      {summary['ers']:.6f}")
    for metric in ("ttft_ms", "tpot_ms", "request_score"):
        values = summary[metric]
        if values["median"] is None:
            print(f"{metric:14s} no samples")
        else:
            print(
                f"{metric:14s} median={values['median']:.3f} "
                f"p95={values['p95']:.3f} mean={values['mean']:.3f}"
            )


def command_run(args: argparse.Namespace) -> int:
    trace = WorkloadTrace.from_path(args.trace)
    results, wall_time = asyncio.run(
        run_trace(
            trace,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout_s=args.timeout,
            max_workers=args.max_workers,
        )
    )
    report = write_report(
        args.output,
        trace=trace,
        base_url=args.base_url,
        results=results,
        wall_time_s=wall_time,
    )
    _print_summary(report["summary"])
    print(f"wall:     {wall_time:.3f}s")
    print(f"report:   {args.output}")
    return 0 if report["summary"]["failed_requests"] == 0 else 2


def command_validate(args: argparse.Namespace) -> int:
    scored = WorkloadTrace.from_path(args.trace)
    if args.warmup_trace:
        validate_warmup_policy(
            args.trace,
            args.warmup_trace,
            require_shape_matched=args.require_shape_matched,
        )
        warmup = WorkloadTrace.from_path(args.warmup_trace)
    else:
        warmup = None
    payload = {
        "name": scored.name,
        "model": scored.model,
        "num_conversations": scored.num_conversations,
        "turns": scored.user_turns_per_conversation,
        "total_requests": scored.total_requests,
        "warmup": (
            {
                "name": warmup.name,
                "total_requests": warmup.total_requests,
                "shape_matched": (
                    warmup.model == scored.model
                    and warmup.num_conversations == scored.num_conversations
                    and warmup.user_turns_per_conversation
                    == scored.user_turns_per_conversation
                    and warmup.output_tokens_per_turn
                    == scored.output_tokens_per_turn
                    and warmup.arrival == scored.arrival
                    and warmup.request == scored.request
                ),
            }
            if warmup is not None
            else None
        ),
    }
    print(json.dumps(payload, indent=2))
    return 0


def _load_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def command_summarize(args: argparse.Namespace) -> int:
    report = _load_report(args.report)
    _print_summary(report["summary"])
    return 0


def command_compare(args: argparse.Namespace) -> int:
    rows = []
    for report_path in args.reports:
        report = _load_report(report_path)
        summary = report["summary"]
        rows.append(
            (
                Path(report_path).name,
                summary["ers"],
                summary["ttft_ms"]["median"],
                summary["ttft_ms"]["p95"],
                summary["tpot_ms"]["mean"],
                summary["failed_requests"],
            )
        )
    print("report\tERS\tTTFT p50\tTTFT p95\tTPOT mean\tfailures")
    for row in sorted(rows, key=lambda item: item[1], reverse=True):
        print(
            "\t".join(
                [
                    row[0],
                    f"{row[1]:.6f}",
                    _number(row[2]),
                    _number(row[3]),
                    _number(row[4]),
                    str(row[5]),
                ]
            )
        )
    return 0


def _number(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def command_score(args: argparse.Namespace) -> int:
    print(
        f"request_score={request_score(args.ttft, args.tpot, completion_tokens=args.completion_tokens, succeeded=not args.failed):.9f}"
    )
    return 0


def command_final_score(args: argparse.Namespace) -> int:
    delta = args.baseline_accuracy - args.submission_accuracy
    print(f"delta={delta:.9f}")
    print(f"accuracy_factor={accuracy_factor(delta):.9f}")
    print(f"final_score={final_score(args.ers, delta):.9f}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="racebench")
    subcommands = root.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run", help="run one multi-turn HTTP workload trace")
    run.add_argument("--trace", required=True)
    run.add_argument("--base-url", default="http://127.0.0.1:8000")
    run.add_argument("--api-key", default="")
    run.add_argument("--timeout", type=float, default=120.0)
    run.add_argument("--max-workers", type=int)
    run.add_argument("--output", required=True)
    run.set_defaults(func=command_run)

    validate = subcommands.add_parser("validate-trace")
    validate.add_argument("trace")
    validate.add_argument("--warmup-trace")
    validate.add_argument("--require-shape-matched", action="store_true")
    validate.set_defaults(func=command_validate)

    summary = subcommands.add_parser("summarize")
    summary.add_argument("report")
    summary.set_defaults(func=command_summarize)

    compare = subcommands.add_parser("compare")
    compare.add_argument("reports", nargs="+")
    compare.set_defaults(func=command_compare)

    score = subcommands.add_parser("score")
    score.add_argument("--ttft", type=float, required=True)
    score.add_argument("--tpot", type=float, required=True)
    score.add_argument("--completion-tokens", type=int, default=1)
    score.add_argument("--failed", action="store_true")
    score.set_defaults(func=command_score)

    final = subcommands.add_parser("final-score")
    final.add_argument("--ers", type=float, required=True)
    final.add_argument("--baseline-accuracy", type=float, default=0.4)
    final.add_argument("--submission-accuracy", type=float, required=True)
    final.set_defaults(func=command_final_score)
    return root


def main() -> None:
    args = parser().parse_args()
    try:
        raise SystemExit(args.func(args))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
