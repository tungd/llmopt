"""Strict multi-turn trace loading shared by local benchmark adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Arrival:
    initial_offsets_s: tuple[float, ...] = ()
    think_time_s: float = 0.0

    def offset_for(self, conversation: int) -> float:
        if not self.initial_offsets_s:
            return 0.0
        return self.initial_offsets_s[conversation]


@dataclass(frozen=True)
class WorkloadTrace:
    name: str
    model: str
    shared_system_prefix: str
    conversation_prefixes: tuple[str, ...]
    user_messages: tuple[tuple[str, ...], ...]
    output_tokens_per_turn: tuple[tuple[int, ...], ...]
    arrival: Arrival = field(default_factory=Arrival)
    request: dict[str, Any] = field(default_factory=dict)
    declared_tokens: dict[str, Any] = field(default_factory=dict)

    @property
    def num_conversations(self) -> int:
        return len(self.user_messages)

    @property
    def user_turns_per_conversation(self) -> int:
        return len(self.user_messages[0])

    @property
    def total_requests(self) -> int:
        return self.num_conversations * self.user_turns_per_conversation

    def output_tokens(self, conversation: int, turn: int) -> int:
        return self.output_tokens_per_turn[conversation][turn]

    @classmethod
    def from_path(cls, path: str | Path) -> WorkloadTrace:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> WorkloadTrace:
        required = {
            "name",
            "model",
            "shared_system_prefix",
            "conversation_prefixes",
            "user_messages",
            "output_tokens_per_turn",
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise ValueError(f"trace is missing required fields: {', '.join(missing)}")

        messages = tuple(
            tuple(str(item) for item in turns) for turns in raw["user_messages"]
        )
        if not messages or not messages[0]:
            raise ValueError("trace needs at least one conversation and one turn")
        turns_per_conversation = len(messages[0])
        if any(len(turns) != turns_per_conversation for turns in messages):
            raise ValueError("all conversations must have the same number of turns")

        prefixes = tuple(str(item) for item in raw["conversation_prefixes"])
        if len(prefixes) != len(messages):
            raise ValueError("conversation_prefixes must match num_conversations")

        output_tokens = _expand_output_tokens(
            raw["output_tokens_per_turn"],
            num_conversations=len(messages),
            turns_per_conversation=turns_per_conversation,
        )
        arrival_raw = raw.get("arrival", {})
        offsets = tuple(
            float(value) for value in arrival_raw.get("initial_offsets_s", ())
        )
        if offsets and len(offsets) != len(messages):
            raise ValueError("arrival.initial_offsets_s must match num_conversations")
        think_time = float(arrival_raw.get("think_time_s", 0.0))
        if any(value < 0 for value in offsets) or think_time < 0:
            raise ValueError("arrival times cannot be negative")

        declared_conversations = raw.get("num_conversations")
        if declared_conversations is not None and int(declared_conversations) != len(
            messages
        ):
            raise ValueError("declared num_conversations does not match user_messages")
        declared_turns = raw.get("user_turns_per_conversation")
        if declared_turns is not None and int(declared_turns) != turns_per_conversation:
            raise ValueError(
                "declared user_turns_per_conversation does not match user_messages"
            )
        declared_total = raw.get("total_requests")
        actual_total = len(messages) * turns_per_conversation
        if declared_total is not None and int(declared_total) != actual_total:
            raise ValueError("declared total_requests does not match trace dimensions")

        request = dict(raw.get("request", {}))
        forbidden = {"messages", "model", "stream", "max_tokens", "min_tokens"}
        overlap = sorted(forbidden & request.keys())
        if overlap:
            raise ValueError(
                f"request overrides harness-owned fields: {', '.join(overlap)}"
            )

        return cls(
            name=str(raw["name"]),
            model=str(raw["model"]),
            shared_system_prefix=str(raw["shared_system_prefix"]),
            conversation_prefixes=prefixes,
            user_messages=messages,
            output_tokens_per_turn=output_tokens,
            arrival=Arrival(offsets, think_time),
            request=request,
            declared_tokens=dict(raw.get("declared_tokens", {})),
        )


def _expand_output_tokens(
    raw: int | list[int] | list[list[int]],
    *,
    num_conversations: int,
    turns_per_conversation: int,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(raw, int):
        matrix = [[raw] * turns_per_conversation for _ in range(num_conversations)]
    elif raw and all(isinstance(item, int) for item in raw):
        if len(raw) != turns_per_conversation:
            raise ValueError("per-turn output token list has the wrong length")
        matrix = [list(raw) for _ in range(num_conversations)]
    else:
        matrix = [list(row) for row in raw]
        if len(matrix) != num_conversations or any(
            len(row) != turns_per_conversation for row in matrix
        ):
            raise ValueError("output token matrix does not match trace dimensions")
    if any(not isinstance(value, int) or value <= 0 for row in matrix for value in row):
        raise ValueError("all output token counts must be positive integers")
    return tuple(tuple(row) for row in matrix)


def validate_warmup_policy(
    trace_path: str | Path,
    warmup_path: str | Path,
    *,
    require_shape_matched: bool = False,
) -> None:
    trace_file = Path(trace_path)
    warmup_file = Path(warmup_path)
    if trace_file.read_bytes() == warmup_file.read_bytes():
        raise ValueError(
            "warmup and scored traces are byte-identical; use an explicitly "
            "distinct warmup trace"
        )
    validate_warmup_traces(
        WorkloadTrace.from_path(trace_file),
        WorkloadTrace.from_path(warmup_file),
        require_shape_matched=require_shape_matched,
    )


def validate_warmup_traces(
    scored: WorkloadTrace,
    warmup: WorkloadTrace,
    *,
    require_shape_matched: bool = False,
) -> None:
    """Validate an in-memory scored/warmup pair using the same contract."""
    if require_shape_matched:
        scored_shape = (
            scored.model,
            scored.num_conversations,
            scored.user_turns_per_conversation,
            scored.output_tokens_per_turn,
            scored.arrival.initial_offsets_s,
            scored.arrival.think_time_s,
            scored.request,
        )
        warmup_shape = (
            warmup.model,
            warmup.num_conversations,
            warmup.user_turns_per_conversation,
            warmup.output_tokens_per_turn,
            warmup.arrival.initial_offsets_s,
            warmup.arrival.think_time_s,
            warmup.request,
        )
        if warmup_shape != scored_shape:
            raise ValueError(
                "warmup trace request shape does not match scored trace: "
                f"warmup={warmup_shape!r}, scored={scored_shape!r}"
            )
