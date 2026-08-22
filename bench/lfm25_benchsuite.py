"""ERS benchsuite for LFM2.5 on the local PyTorch MPS target.

The suite keeps the reference runner's separation between a validated trace,
an unscored warmup trace, per-request TTFT/TPOT measurements, and a natural
needle-in-a-haystack correctness probe.  The MPS adapter is intentionally
in-process: the model is the endpoint and the candidate is selected through
the same Dynamo backend used by the production path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.generation.utils import GenerationMixin

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "python"))

from llmopt_backend import llmopt  # noqa: E402
from llmopt_backend.quantization import quantize_model_  # noqa: E402
from racebench.benchmark import RequestResult, engine_pass, write_report  # noqa: E402
from racebench.needle import (  # noqa: E402
    DEFAULT_LENGTHS,
    DEFAULT_POSITIONS,
    EXPECTED,
    archive_corpus,
    chat_token_ids,
    exact_natural_messages,
)
from racebench.score import request_score  # noqa: E402
from racebench.profiles import profile as workload_profile  # noqa: E402
from racebench.trace import (  # noqa: E402
    WorkloadTrace,
    validate_warmup_policy,
    validate_warmup_traces,
)


class TokenClock(LogitsProcessor):
    """Record the point at which each generated token's logits are available."""

    def __init__(self) -> None:
        self.events: list[float] = []

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # MPS dispatch is asynchronous. Synchronize before recording the
        # observation so TTFT means that the first logits are actually ready,
        # rather than merely enqueued.
        synchronize()
        self.events.append(time.perf_counter())
        return scores


class RoutedGenerationModel(GenerationMixin):
    """Keep Transformers generation while routing every forward through a callable.

    ``torch.compile(model)`` returns an ``OptimizedModule`` whose delegated
    ``generate`` method remains bound to the original model. Calling that
    method would therefore bypass the backend. This small GenerationMixin
    proxy keeps the owner's generation helpers/configuration but makes the
    actual ``self(...)`` calls go through the selected forward callable.
    """

    def __init__(self, owner: Any, forwarder: Any):
        self._owner = owner
        self._forwarder = forwarder

    def __getattr__(self, name: str) -> Any:
        owner = object.__getattribute__(self, "_owner")
        return getattr(owner, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._forwarder(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._forwarder(*args, **kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        next_sequence_length: int | None = None,
        past_key_values: Any = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        is_first_iteration: bool | None = False,
        **kwargs: Any,
    ) -> Any:
        return self._owner.prepare_inputs_for_generation(
            input_ids=input_ids,
            next_sequence_length=next_sequence_length,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )


def synchronize() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def release_mps_cache() -> None:
    """Release unused MPS allocator blocks between independent requests."""
    if torch.backends.mps.is_available():
        synchronize()
        torch.mps.empty_cache()


def hardware_model() -> str | None:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "hw.model"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parse_ints(raw: str, *, name: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in raw.split(",") if value.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def _input_ids(tokenizer: Any, messages: list[dict[str, str]], device: torch.device) -> torch.Tensor:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if not torch.is_tensor(encoded):
        encoded = torch.tensor(encoded, dtype=torch.long)
    if encoded.ndim == 1:
        encoded = encoded.unsqueeze(0)
    return encoded.to(device)


def _generated_text(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _token_ids_sha256(token_ids: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def generate_request(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    expected_tokens: int,
    device: torch.device,
    request: Mapping[str, Any] | None = None,
) -> tuple[RequestResult, str]:
    input_ids = _input_ids(tokenizer, messages, device)
    prompt_tokens = int(input_ids.shape[-1])
    clock = TokenClock()
    started_at = time.perf_counter()
    error: str | None = None
    generated: torch.Tensor | None = None
    output_token_ids: list[int] = []
    try:
        with torch.no_grad():
            generate_kwargs = dict(request or {})
            # ``seed`` is an OpenAI/vLLM trace field. Greedy local Transformers
            # generation is deterministic, but Transformers does not accept
            # the server-side field as a generation kwarg.
            generate_kwargs.pop("seed", None)
            generate_kwargs.pop("ignore_eos", None)
            generate_kwargs.update(
                {
                    "input_ids": input_ids,
                    "max_new_tokens": expected_tokens,
                    "min_new_tokens": expected_tokens,
                    "do_sample": False,
                    "use_cache": True,
                    "eos_token_id": None,
                    "pad_token_id": tokenizer.eos_token_id,
                    "logits_processor": LogitsProcessorList([clock]),
                }
            )
            generated = model.generate(**generate_kwargs)
        synchronize()
    except Exception as exc:  # record one request failure in the report
        error = f"{type(exc).__name__}: {exc}"

    ended_at = time.perf_counter()
    if generated is None:
        completion_tokens = 0
        text = ""
    else:
        output_token_ids = [
            int(token)
            for token in generated[0, prompt_tokens:].detach().cpu().tolist()
        ]
        completion_tokens = len(output_token_ids)
        text = _generated_text(tokenizer, output_token_ids)
    succeeded = error is None and completion_tokens > 0
    ttft_ms = (
        (clock.events[0] - started_at) * 1000.0 if succeeded and clock.events else None
    )
    if succeeded and completion_tokens > 1 and len(clock.events) >= 2:
        tpot_ms = (clock.events[-1] - clock.events[0]) * 1000.0 / (completion_tokens - 1)
    elif succeeded:
        tpot_ms = 0.0
    else:
        tpot_ms = None
    result = RequestResult(
        request_id="",
        conversation=0,
        turn=0,
        scheduled_offset_s=0.0,
        queue_delay_ms=0.0,
        http_status=200 if succeeded else None,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        latency_ms=(ended_at - started_at) * 1000.0,
        completion_tokens=completion_tokens,
        expected_completion_tokens=expected_tokens,
        output_text=text,
        succeeded=succeeded,
        error=error,
        score=request_score(
            ttft_ms,
            tpot_ms,
            completion_tokens=completion_tokens,
            succeeded=succeeded,
        ),
        prompt_tokens=prompt_tokens,
        output_token_ids=output_token_ids,
        output_token_ids_sha256=_token_ids_sha256(output_token_ids),
    )
    return result, text


def _with_request_identity(
    result: RequestResult,
    *,
    request_id: str,
    conversation: int,
    turn: int,
    scheduled_at: float,
    trace_started: float,
    started_at: float,
) -> RequestResult:
    result.request_id = request_id
    result.conversation = conversation
    result.turn = turn
    result.scheduled_offset_s = max(0.0, scheduled_at - trace_started)
    result.queue_delay_ms = max(0.0, (started_at - scheduled_at) * 1000.0)
    return result


def _skipped_result(trace: WorkloadTrace, conversation: int, turn: int) -> RequestResult:
    return RequestResult(
        request_id=f"c{conversation:04d}-t{turn:03d}",
        conversation=conversation,
        turn=turn,
        scheduled_offset_s=0.0,
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


def run_trace(
    model: Any,
    tokenizer: Any,
    trace: WorkloadTrace,
    *,
    device: torch.device,
    max_workers: int | None = None,
    release_cache: bool = False,
) -> tuple[list[RequestResult], float]:
    """Run conversations concurrently while keeping each conversation closed-loop."""
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be positive")
    trace_started = time.perf_counter()

    def run_conversation(conversation: int) -> list[RequestResult]:
        messages: list[dict[str, str]] = []
        if trace.shared_system_prefix:
            messages.append({"role": "system", "content": trace.shared_system_prefix})
        initial_offset = trace.arrival.offset_for(conversation)
        target = trace_started + initial_offset
        time.sleep(max(0.0, target - time.perf_counter()))
        conversation_results: list[RequestResult] = []
        for turn, user_text in enumerate(trace.user_messages[conversation]):
            if turn > 0 and trace.arrival.think_time_s:
                time.sleep(trace.arrival.think_time_s)
            prefix = trace.conversation_prefixes[conversation] if turn == 0 else ""
            messages.append({"role": "user", "content": f"{prefix}{user_text}"})
            scheduled_at = target if turn == 0 else time.perf_counter()
            started_at = time.perf_counter()
            result, text = generate_request(
                model,
                tokenizer,
                messages,
                expected_tokens=trace.output_tokens(conversation, turn),
                device=device,
                request=trace.request,
            )
            conversation_results.append(
                _with_request_identity(
                    result,
                    request_id=f"c{conversation:04d}-t{turn:03d}",
                    conversation=conversation,
                    turn=turn,
                    scheduled_at=scheduled_at,
                    trace_started=trace_started,
                    started_at=started_at,
                )
            )
            if not result.succeeded:
                conversation_results.extend(
                    _skipped_result(trace, conversation, skipped_turn)
                    for skipped_turn in range(turn + 1, trace.user_turns_per_conversation)
                )
                break
            messages.append({"role": "assistant", "content": text})
            if release_cache:
                release_mps_cache()
        return conversation_results

    # PyTorch MPS does not provide the independent command-queue behavior of
    # the HTTP serving engine. Concurrent generation calls can interleave
    # Metal encoders and abort the process, so the local adapter serializes
    # requests by default. The HTTP adapter retains racebench concurrency.
    worker_count = max_workers or 1
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="racebench-mps",
    ) as pool:
        grouped = list(pool.map(run_conversation, range(trace.num_conversations)))
    results = sorted(
        (result for group in grouped for result in group),
        key=lambda result: (result.conversation, result.turn),
    )
    return results, time.perf_counter() - trace_started


def run_needle_probe(
    model: Any,
    tokenizer: Any,
    *,
    device: torch.device,
    lengths: Sequence[int],
    positions: Sequence[int],
    max_new_tokens: int,
    release_cache: bool = False,
) -> dict[str, Any]:
    corpus = archive_corpus()
    rows: list[dict[str, Any]] = []
    for target_tokens in lengths:
        for needle_percent in positions:
            messages = exact_natural_messages(
                tokenizer, target_tokens, needle_percent, corpus
            )
            exact_tokens = len(chat_token_ids(tokenizer, messages))
            result, text = generate_request(
                model,
                tokenizer,
                messages,
                expected_tokens=max_new_tokens,
                device=device,
            )
            normalized = text.strip().upper().replace("`", "")
            rows.append(
                {
                    "target_prompt_tokens": target_tokens,
                    "needle_percent": needle_percent,
                    "constructed_prompt_tokens": exact_tokens,
                    "success": result.succeeded,
                    "correct": normalized == EXPECTED,
                    "elapsed_s": result.latency_ms / 1000.0,
                    "ttft_ms": result.ttft_ms,
                    "tpot_ms": result.tpot_ms,
                    "completion_tokens": result.completion_tokens,
                    "output_token_ids": result.output_token_ids,
                    "output_token_ids_sha256": result.output_token_ids_sha256,
                    "text": text,
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "error": result.error,
                }
            )
            print(json.dumps(rows[-1], sort_keys=True), flush=True)
            if release_cache:
                release_mps_cache()
    return {
        "lengths": list(lengths),
        "positions": list(positions),
        "expected": EXPECTED,
        "max_tokens": max_new_tokens,
        "correct": sum(row["correct"] for row in rows),
        "total": len(rows),
        "results": rows,
    }


def token_output_parity(
    eager_results: Sequence[RequestResult],
    candidate_results: Sequence[RequestResult],
) -> dict[str, Any]:
    """Compare exact generated token sequences by request identity."""
    eager_by_id = {result.request_id: result for result in eager_results}
    candidate_by_id = {result.request_id: result for result in candidate_results}
    mismatches: list[dict[str, Any]] = []
    for request_id in sorted(set(eager_by_id) | set(candidate_by_id)):
        eager = eager_by_id.get(request_id)
        candidate = candidate_by_id.get(request_id)
        eager_ids = eager.output_token_ids if eager is not None else None
        candidate_ids = candidate.output_token_ids if candidate is not None else None
        if eager_ids != candidate_ids:
            mismatches.append(
                {
                    "request_id": request_id,
                    "eager_sha256": (
                        eager.output_token_ids_sha256 if eager is not None else None
                    ),
                    "candidate_sha256": (
                        candidate.output_token_ids_sha256
                        if candidate is not None
                        else None
                    ),
                    "eager_token_ids": eager_ids,
                    "candidate_token_ids": candidate_ids,
                }
            )
    return {
        "equal": not mismatches,
        "compared_requests": len(set(eager_by_id) & set(candidate_by_id)),
        "mismatches": mismatches,
    }


def fixed_forward_correctness(
    eager_model: Any,
    candidate_model: Any,
    tokenizer: Any,
    *,
    prompt: str,
    device: torch.device,
) -> dict[str, Any]:
    input_ids = _input_ids(tokenizer, [{"role": "user", "content": prompt}], device)
    with torch.no_grad():
        eager = eager_model(input_ids=input_ids, use_cache=False).logits
        candidate = candidate_model(input_ids=input_ids, use_cache=False).logits
    synchronize()
    delta = (eager - candidate).abs()
    return {
        "shape": list(eager.shape),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "exact": bool(torch.equal(eager, candidate)),
    }


def fixed_forward_observation(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str,
    device: torch.device,
) -> dict[str, Any]:
    """Record a cross-process comparable digest for one fixed forward."""
    input_ids = _input_ids(tokenizer, [{"role": "user", "content": prompt}], device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits
    synchronize()
    cpu_logits = logits.detach().cpu().contiguous()
    payload = cpu_logits.numpy().tobytes()
    return {
        "shape": list(cpu_logits.shape),
        "dtype": str(cpu_logits.dtype),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _report_token_output_parity(
    eager_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
) -> dict[str, Any]:
    eager_by_id = {row["request_id"]: row for row in eager_report["requests"]}
    candidate_by_id = {
        row["request_id"]: row for row in candidate_report["requests"]
    }
    mismatches: list[dict[str, Any]] = []
    for request_id in sorted(set(eager_by_id) | set(candidate_by_id)):
        eager = eager_by_id.get(request_id)
        candidate = candidate_by_id.get(request_id)
        eager_ids = eager.get("output_token_ids") if eager is not None else None
        candidate_ids = (
            candidate.get("output_token_ids") if candidate is not None else None
        )
        if eager_ids != candidate_ids:
            mismatches.append(
                {
                    "request_id": request_id,
                    "eager_sha256": (
                        eager.get("output_token_ids_sha256")
                        if eager is not None
                        else None
                    ),
                    "candidate_sha256": (
                        candidate.get("output_token_ids_sha256")
                        if candidate is not None
                        else None
                    ),
                    "eager_token_ids": eager_ids,
                    "candidate_token_ids": candidate_ids,
                }
            )
    return {
        "equal": not mismatches,
        "compared_requests": len(set(eager_by_id) & set(candidate_by_id)),
        "mismatches": mismatches,
    }


def _metadata(
    model_id: str,
    device: torch.device,
    load_seconds: float,
    *,
    quantization: str,
    quantization_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    mps_memory = {}
    if torch.backends.mps.is_available():
        mps_memory = {
            "recommended_max_bytes": torch.mps.recommended_max_memory(),
            "allocated_bytes_at_record": torch.mps.current_allocated_memory(),
            "driver_allocated_bytes_at_record": torch.mps.driver_allocated_memory(),
        }
    return {
        "model": model_id,
        "quantization": quantization,
        "quantization_summary": quantization_summary,
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "transformers": __import__("transformers").__version__,
        "host": {
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "model": hardware_model(),
        },
        "load_seconds": load_seconds,
        "mps_watermark_environment": {
            "high_ratio": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
            "low_ratio": os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO"),
        },
        "mps_memory": mps_memory,
        "timing": {
            "ttft": "perf_counter after device synchronization at first logits processor",
            "tpot": "mean interval between synchronized logits processor observations",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    parser.add_argument(
        "--quantization",
        choices=("q8", "fp16"),
        default="q8",
        help="weight format for the model-level run (default: q8)",
    )
    parser.add_argument("--trace", default="bench/traces/lfm25-mps-smoke.json")
    parser.add_argument(
        "--warmup-trace", default="bench/traces/lfm25-mps-warmup.json"
    )
    parser.add_argument(
        "--profile",
        choices=("semantic-5x3", "official-shape-70x6"),
        help="use a deterministic built-in long-context workload instead of files",
    )
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--needle-lengths", default=",".join(map(str, DEFAULT_LENGTHS)))
    parser.add_argument("--needle-positions", default=",".join(map(str, DEFAULT_POSITIONS)))
    parser.add_argument("--needle-max-tokens", type=int, default=12)
    parser.add_argument("--artifact-dir", default="_artifacts/lfm25-benchsuite")
    parser.add_argument("--output", default="_artifacts/lfm25-benchsuite/result.json")
    parser.add_argument(
        "--record-output",
        help="also write the compact top-level result to a tracked baseline path",
    )
    parser.add_argument("--skip-needle", action="store_true")
    parser.add_argument(
        "--require-needle",
        action="store_true",
        help="make exact needle retrieval part of the process exit status",
    )
    parser.add_argument("--require-shape-matched-warmup", action="store_true")
    parser.add_argument(
        "--release-mps-cache",
        action="store_true",
        help="release unused MPS allocator blocks after each request",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help=(
            "maximum concurrent conversations; defaults to one worker per "
            "conversation, matching the adjacent racebench runner"
        ),
    )
    parser.add_argument(
        "--isolate",
        action="store_true",
        help="run eager and llmopt in separate child processes and combine reports",
    )
    parser.add_argument(
        "--candidate",
        choices=("both", "eager", "llmopt"),
        default="both",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--combine-existing",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--candidate-order",
        default="eager,llmopt",
        help="order for isolated child processes (comma-separated)",
    )
    return parser.parse_args()


def _report_paths(root: Path, candidate: str) -> tuple[Path, Path, Path]:
    directory = root / candidate
    return directory / "warmup.json", directory / "report.json", directory


def _write_top_level_result(
    result: Mapping[str, Any], output: str | Path, record_output: str | None
) -> None:
    payload = json.dumps(result, indent=2) + "\n"
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")
    if record_output:
        record_path = Path(record_output)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(payload, encoding="utf-8")


def _load_workload(args: argparse.Namespace) -> tuple[WorkloadTrace, WorkloadTrace, str]:
    if args.profile:
        trace = workload_profile(args.profile)
        warmup_trace = workload_profile(args.profile, warmup=True)
        validate_warmup_traces(
            trace,
            warmup_trace,
            require_shape_matched=args.require_shape_matched_warmup,
        )
        return trace, warmup_trace, f"profile:{args.profile}"
    validate_warmup_policy(
        args.trace,
        args.warmup_trace,
        require_shape_matched=args.require_shape_matched_warmup,
    )
    return (
        WorkloadTrace.from_path(args.trace),
        WorkloadTrace.from_path(args.warmup_trace),
        args.trace,
    )


def _candidate_order(raw: str) -> tuple[str, str]:
    order = tuple(item.strip() for item in raw.split(",") if item.strip())
    if order not in (("eager", "llmopt"), ("llmopt", "eager")):
        raise ValueError("--candidate-order must be eager,llmopt or llmopt,eager")
    return order  # type: ignore[return-value]


def _summary_median(summary: Mapping[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if not isinstance(value, Mapping):
        return None
    median = value.get("median")
    return float(median) if median is not None else None


def _comparison(
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    isolation: bool,
) -> dict[str, Any]:
    eager = candidates.get("eager", {}).get("scored", {})
    llmopt = candidates.get("llmopt", {}).get("scored", {})
    eager_ttft = _summary_median(eager, "ttft_ms")
    llmopt_ttft = _summary_median(llmopt, "ttft_ms")
    eager_tpot = _summary_median(eager, "tpot_ms")
    llmopt_tpot = _summary_median(llmopt, "tpot_ms")
    return {
        "scored_ers_delta_llmopt_minus_eager": (
            llmopt.get("ers") - eager.get("ers")
            if llmopt.get("ers") is not None and eager.get("ers") is not None
            else None
        ),
        "scored_ttft_median_delta_ms_llmopt_minus_eager": (
            llmopt_ttft - eager_ttft
            if llmopt_ttft is not None and eager_ttft is not None
            else None
        ),
        "scored_tpot_median_delta_ms_llmopt_minus_eager": (
            llmopt_tpot - eager_tpot
            if llmopt_tpot is not None and eager_tpot is not None
            else None
        ),
        "valid_for_relative_speed_claim": False,
        "invalidity_reason": (
            "Each candidate ran in its own process, but this is one execution "
            "per candidate with no repeated or counterbalanced samples."
            if isolation
            else "Eager and llmopt were measured sequentially in one process; "
            "the candidate follows shared warm state and compilation warmup."
        ),
    }


def _worker_command(args: argparse.Namespace, candidate: str, root: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--quantization",
        args.quantization,
        "--prompt",
        args.prompt,
        "--needle-lengths",
        args.needle_lengths,
        "--needle-positions",
        args.needle_positions,
        "--needle-max-tokens",
        str(args.needle_max_tokens),
        "--artifact-dir",
        str(root),
        "--output",
        str(root / candidate / "result.json"),
        "--candidate",
        candidate,
        "--worker",
    ]
    if args.profile:
        command.extend(["--profile", args.profile])
    else:
        command.extend(["--trace", args.trace, "--warmup-trace", args.warmup_trace])
    if args.skip_needle:
        command.append("--skip-needle")
    if args.max_workers is not None:
        command.extend(["--max-workers", str(args.max_workers)])
    if args.release_mps_cache:
        command.append("--release-mps-cache")
    if args.require_shape_matched_warmup:
        command.append("--require-shape-matched-warmup")
    return command


def _combine_isolated_results(
    args: argparse.Namespace,
    order: tuple[str, str],
    child_results: Mapping[str, Mapping[str, Any]],
) -> int:
    eager_result = child_results["eager"]
    llmopt_result = child_results["llmopt"]
    candidates: dict[str, dict[str, Any]] = {}
    for child in (eager_result, llmopt_result):
        candidates.update(child["candidates"])

    eager_observation = eager_result["fixed_forward_observation"]
    llmopt_observation = llmopt_result["fixed_forward_observation"]
    correctness = {
        "shape": eager_observation.get("shape"),
        "dtype": eager_observation.get("dtype"),
        "eager_sha256": eager_observation.get("sha256"),
        "llmopt_sha256": llmopt_observation.get("sha256"),
        "max_abs": None,
        "mean_abs": None,
        "exact": (
            eager_observation.get("shape") == llmopt_observation.get("shape")
            and eager_observation.get("dtype") == llmopt_observation.get("dtype")
            and eager_observation.get("sha256") == llmopt_observation.get("sha256")
        ),
        "comparison_mode": "cross-process tensor digest",
    }
    reports = {}
    for candidate in ("eager", "llmopt"):
        candidate_info = candidates[candidate]
        reports[candidate] = {
            phase: json.loads(
                Path(
                    candidate_info[
                        "warmup_report" if phase == "warmup" else "report"
                    ]
                ).read_text(encoding="utf-8")
            )
            for phase in ("warmup", "scored")
        }
    token_parity = {
        phase: _report_token_output_parity(
            reports["eager"][phase], reports["llmopt"][phase]
        )
        for phase in ("warmup", "scored")
    }
    needle: dict[str, Any] = {}
    needle.update(eager_result.get("needle", {}))
    needle.update(llmopt_result.get("needle", {}))
    needle_ok = (
        None
        if args.skip_needle
        else all(probe["correct"] == probe["total"] for probe in needle.values())
    )
    measured_engine_pass = engine_pass(
        candidates,
        correctness,
        expected_requests=eager_result["candidates"]["eager"]["scored"].get("requests"),
    )
    baseline_summary = candidates["eager"]["scored"]
    process_ok = measured_engine_pass and (
        not args.require_needle or needle_ok is True
    )
    result = {
        "schema_version": 2,
        "measurement_status": "isolated_observation",
        "created_at": __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(),
        "target": "pytorch-mps",
        "optimization": f"fx-direct-execution+{args.quantization}",
        "trace": args.profile or args.trace,
        "warmup_trace": (
            f"profile:{args.profile}:warmup" if args.profile else args.warmup_trace
        ),
        "candidate_order": list(order),
        "metadata": {
            **eager_result["metadata"],
            "isolation": "one child process per candidate",
            "input_prompt": args.prompt,
        },
        "racebench_contract": {
            "score": "official ERS constants from viettel-ai-race/src/racebench/score.py",
            "runner": "serial MPS conversations with serial closed-loop turns",
            "max_workers": args.max_workers or 1,
            "http_reference": "concurrent conversations with serial closed-loop turns",
            "warmup": "byte-distinct and shape-matched when requested",
            "completion_tokens": "pinned per trace request",
            "mps_cache_release": args.release_mps_cache,
            "target_model_substitution": {
                "reference": "LiquidAI/LFM2.5-1.2B-Instruct",
                "local": args.model,
            },
        },
        "candidates": candidates,
        "baseline": {
            "candidate": "eager",
            "metric": "ERS",
            "ers": baseline_summary.get("ers"),
            "scored_ers": baseline_summary.get("ers"),
            "successful_requests": baseline_summary.get("successful_requests"),
            "requests": baseline_summary.get("requests"),
        },
        "comparison": _comparison(candidates, isolation=True),
        "correctness": correctness,
        "token_output_parity": token_parity,
        "needle": needle,
        "engine_pass": measured_engine_pass,
        "needle_validation": {"required": args.require_needle, "passed": needle_ok},
        "exit_code": 0 if process_ok else 1,
    }
    _write_top_level_result(result, args.output, args.record_output)
    print(json.dumps(result, indent=2))
    return 0 if process_ok else 1


def _run_isolated(args: argparse.Namespace) -> int:
    order = _candidate_order(args.candidate_order)
    root = Path(args.artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    child_results: dict[str, dict[str, Any]] = {}
    for candidate in order:
        completed = subprocess.run(
            _worker_command(args, candidate, root),
            check=False,
            capture_output=True,
            text=True,
        )
        result_path = root / candidate / "result.json"
        if completed.returncode != 0 or not result_path.exists():
            sys.stderr.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            return completed.returncode or 1
        child_results[candidate] = json.loads(result_path.read_text(encoding="utf-8"))

    return _combine_isolated_results(args, order, child_results)


def _combine_existing(args: argparse.Namespace) -> int:
    order = _candidate_order(args.candidate_order)
    root = Path(args.artifact_dir)
    child_results: dict[str, dict[str, Any]] = {}
    for candidate in order:
        result_path = root / candidate / "result.json"
        if not result_path.exists():
            raise FileNotFoundError(f"missing child result: {result_path}")
        child_results[candidate] = json.loads(result_path.read_text(encoding="utf-8"))
    return _combine_isolated_results(args, order, child_results)


def main() -> int:
    args = parse_args()
    if args.combine_existing:
        return _combine_existing(args)
    if args.isolate and not args.worker:
        return _run_isolated(args)
    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is not available on this host")
    trace, warmup_trace, trace_source = _load_workload(args)
    lengths = _parse_ints(args.needle_lengths, name="--needle-lengths")
    positions = _parse_ints(args.needle_positions, name="--needle-positions")
    if any(position >= 100 for position in positions):
        raise ValueError("--needle-positions must be between 1 and 99")
    if args.needle_max_tokens < 1:
        raise ValueError("--needle-max-tokens must be positive")

    compiler = Path(__file__).parents[1] / "_build" / "bin" / "llmopt-fx"
    if not compiler.exists() and not os.environ.get("LLMOPT_FX_COMPILER"):
        raise RuntimeError("llmopt-fx is not available; run `ninja -f ninja.build all`")
    os.environ["LLMOPT_FX_FALLBACK"] = "0"
    os.environ["LLMOPT_QUANTIZATION"] = args.quantization
    artifact_root = Path(args.artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    os.environ["LLMOPT_ARTIFACT_DIR"] = str(artifact_root / "graphs")

    device = torch.device("mps")
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    eager_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval()
    quantization_summary = None
    if args.quantization == "q8":
        quantization_summary = quantize_model_(eager_model)
    eager_model = eager_model.to(device)
    load_seconds = time.perf_counter() - load_start
    metadata = _metadata(
        args.model,
        device,
        load_seconds,
        quantization=args.quantization,
        quantization_summary=quantization_summary,
    )

    candidate_models: dict[str, Any] = {}
    if args.candidate in ("both", "eager"):
        candidate_models["eager"] = eager_model
    if args.candidate in ("both", "llmopt"):
        candidate_models["llmopt"] = torch.compile(
            eager_model,
            backend=llmopt,
            fullgraph=False,
            dynamic=False,
        )

    candidates: dict[str, dict[str, Any]] = {}
    candidate_results: dict[str, dict[str, list[RequestResult]]] = {}
    needle: dict[str, Any] = {}
    for name, model in candidate_models.items():
        generation_model = RoutedGenerationModel(eager_model, model)
        warmup_results, warmup_wall = run_trace(
            generation_model,
            tokenizer,
            warmup_trace,
            device=device,
            max_workers=args.max_workers,
            release_cache=args.release_mps_cache,
        )
        scored_results, scored_wall = run_trace(
            generation_model,
            tokenizer,
            trace,
            device=device,
            max_workers=args.max_workers,
            release_cache=args.release_mps_cache,
        )
        warmup_path, report_path, candidate_dir = _report_paths(artifact_root, name)
        candidate_metadata = {**metadata, "phase": "warmup"}
        warmup_report = write_report(
            warmup_path,
            trace=warmup_trace,
            candidate=name,
            target="pytorch-mps",
            results=warmup_results,
            wall_time_s=warmup_wall,
            metadata=candidate_metadata,
        )
        report = write_report(
            report_path,
            trace=trace,
            candidate=name,
            target="pytorch-mps",
            results=scored_results,
            wall_time_s=scored_wall,
            metadata={**metadata, "phase": "scored"},
        )
        candidates[name] = {
            "warmup": warmup_report["summary"],
            "scored": report["summary"],
            "warmup_report": str(warmup_path),
            "report": str(report_path),
        }
        candidate_results[name] = {
            "warmup": warmup_results,
            "scored": scored_results,
        }
        if not args.skip_needle:
            needle[name] = run_needle_probe(
                generation_model,
                tokenizer,
                device=device,
                lengths=lengths,
                positions=positions,
                max_new_tokens=args.needle_max_tokens,
                release_cache=args.release_mps_cache,
            )

    if args.candidate == "both":
        correctness = fixed_forward_correctness(
            eager_model,
            candidate_models["llmopt"],
            tokenizer,
            prompt=args.prompt,
            device=device,
        )
    else:
        correctness = fixed_forward_observation(
            candidate_models[args.candidate],
            tokenizer,
            prompt=args.prompt,
            device=device,
        )
    needle_ok = (
        None
        if args.skip_needle
        else all(probe["correct"] == probe["total"] for probe in needle.values())
    )
    measured_engine_pass = engine_pass(
        candidates,
        {"exact": True} if args.candidate != "both" else correctness,
        expected_requests=trace.total_requests,
    )
    token_parity = {
        phase: token_output_parity(
            candidate_results.get("eager", {}).get(phase, []),
            candidate_results.get("llmopt", {}).get(phase, []),
        )
        for phase in ("warmup", "scored")
    }
    baseline_summary = candidates.get("eager", {}).get("scored", {})
    process_ok = measured_engine_pass and (not args.require_needle or needle_ok is True)
    result = {
        "schema_version": 2,
        "measurement_status": "worker_observation" if args.worker else "in_process_observation",
        "created_at": __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(),
        "target": "pytorch-mps",
        "optimization": f"fx-direct-execution+{args.quantization}",
        "trace": trace_source,
        "warmup_trace": (
            f"profile:{args.profile}:warmup" if args.profile else args.warmup_trace
        ),
        "metadata": {**metadata, "input_prompt": args.prompt},
        "racebench_contract": {
            "score": "official ERS constants from viettel-ai-race/src/racebench/score.py",
            "runner": "serial MPS conversations with serial closed-loop turns",
            "max_workers": args.max_workers or 1,
            "http_reference": "concurrent conversations with serial closed-loop turns",
            "warmup": "byte-distinct and shape-matched when requested",
            "completion_tokens": "pinned per trace request",
            "mps_cache_release": args.release_mps_cache,
            "target_model_substitution": {
                "reference": "LiquidAI/LFM2.5-1.2B-Instruct",
                "local": args.model,
            },
        },
        "candidates": candidates,
        "baseline": {
            "candidate": "eager",
            "metric": "ERS",
            "ers": baseline_summary.get("ers"),
            "scored_ers": baseline_summary.get("ers"),
            "successful_requests": baseline_summary.get("successful_requests"),
            "requests": baseline_summary.get("requests"),
        },
        "correctness": correctness,
        "token_output_parity": token_parity,
        "needle": needle,
        "engine_pass": measured_engine_pass,
        "needle_validation": {
            "required": args.require_needle,
            "passed": needle_ok,
        },
        "exit_code": 0 if process_ok else 1,
    }
    if args.candidate == "both":
        result["comparison"] = _comparison(candidates, isolation=False)
    else:
        result["fixed_forward_observation"] = correctness
    _write_top_level_result(result, args.output, args.record_output)
    print(json.dumps(result, indent=2))
    return 0 if process_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
