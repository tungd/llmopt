"""Deterministic local workload profiles shaped like the adjacent racebench."""

from __future__ import annotations

import random

from .trace import Arrival, WorkloadTrace


MODEL = "LiquidAI/LFM2.5-2.6B"
SEMANTIC_INITIAL_OFFSETS = (
    0.0,
    0.2040120574549602,
    0.209077825263508,
    0.27340263807850107,
    0.3239198753125238,
)


def _official_initial_offsets() -> tuple[float, ...]:
    """Reproduce the adjacent runner's seeded 70-rps arrival sequence."""
    rng = random.Random(42)
    offsets = [0.0]
    elapsed = 0.0
    for _ in range(69):
        elapsed += rng.expovariate(70.0)
        offsets.append(elapsed)
    return tuple(offsets)


OFFICIAL_INITIAL_OFFSETS = _official_initial_offsets()

_TOPICS = (
    (
        "Aegis storage incident",
        "a multi-region storage outage",
        "the replica quorum, retry storm, and recovery timeline",
    ),
    (
        "Helios wealth policy",
        "a long-horizon investment policy",
        "inflation assumptions, liquidity constraints, and allocation risk",
    ),
    (
        "Northstar migration",
        "a staged infrastructure migration",
        "service-level obligations, rollback criteria, and dependency order",
    ),
    (
        "Meridian clinical audit",
        "a regulated clinical data workflow",
        "access controls, retention rules, and evidence provenance",
    ),
    (
        "Orion supply review",
        "a constrained hardware supply plan",
        "lead times, substitution decisions, and delivery risk",
    ),
)


def _records(prefix: str, topic: str, detail: str, *, count: int) -> str:
    rows = [
        f"{prefix} reference document for {topic}. Treat this record as source "
        f"material and preserve distinctions between observations, assumptions, "
        f"and recommendations."
    ]
    for index in range(1, count + 1):
        rows.append(
            f"{prefix} entry {index:03d}: the working group reviewed {detail}; "
            f"the review retained scenario {index % 4}, owner group "
            f"{(index * 7) % 19:02d}, and evidence bundle {prefix}-{index:03d}."
        )
    return "\n".join(rows)


def _turn_question(topic: str, detail: str, turn: int, *, warmup: bool) -> str:
    marker = "warmup" if warmup else "scored"
    round_number = turn + 1
    questions = (
        f"For the {topic}, produce a structured analysis of {detail}. "
        "Separate facts stated in the record from conclusions you infer, and "
        "identify the highest-impact dependency before proposing actions.",
        f"Using the same {topic} record, challenge the previous analysis of "
        f"{detail}. Compare at least two plausible interpretations, explain "
        "which evidence would distinguish them, and preserve uncertainty where "
        "the record is incomplete.",
        f"Give an implementation-oriented decision memo for the {topic}. "
        f"Rank the next actions for {detail}, include risks and rollback signals, "
        "and end with a concise recommendation grounded in the supplied record.",
    )
    return (
        f"This is the {marker} variant of turn {round_number}. "
        f"{questions[turn % len(questions)]} "
        "The answer may be detailed; do not invent identifiers or numerical "
        "facts that are absent from the context. Use a clear structure with "
        "an evidence section, an uncertainty section, and an action section. "
        "When the record contains competing priorities, state the trade-off "
        "before selecting an option. Keep references tied to the case rather "
        "than substituting generic best practices for the supplied context."
    )


def _official_topic(index: int) -> tuple[str, str, str]:
    topic, subject, detail = _TOPICS[index % len(_TOPICS)]
    return f"{topic} case {index + 1:02d}", subject, detail


def semantic_5x3(*, warmup: bool = False) -> WorkloadTrace:
    """Return the local 5-conversation x 3-turn semantic workload.

    The shape follows ``traces/sample-semantic-5x3.json`` in the adjacent
    racebench repository.  Text is local and deterministic so this project
    does not depend on a generator service or a second model.  Token counts
    are measured by the benchmark's tokenizer at runtime.
    """

    variant = "warmup" if warmup else "scored"
    shared_prefix = (
        "You are a careful production assistant. The following shared operating "
        f"brief is the {variant} workload context. Use it as background only; "
        "answer each user request from the complete conversation and distinguish "
        "source evidence from analysis.\n\n"
        + _records("SHARED-" + variant.upper(), "the operating brief", "cross-team controls", count=22)
    )
    prefixes = tuple(
        _records(
            f"CASE-{variant.upper()}-{index + 1:02d}",
            topic,
            detail,
            count=20,
        )
        for index, (topic, _, detail) in enumerate(_TOPICS)
    )
    messages = tuple(
        tuple(
            _turn_question(topic, detail, turn, warmup=warmup)
            for turn in range(3)
        )
        for topic, _, detail in _TOPICS
    )
    return WorkloadTrace(
        name=f"lfm25-mps-semantic-5x3-{variant}",
        model=MODEL,
        shared_system_prefix=shared_prefix,
        conversation_prefixes=prefixes,
        user_messages=messages,
        output_tokens_per_turn=tuple((300, 300, 300) for _ in _TOPICS),
        arrival=Arrival(SEMANTIC_INITIAL_OFFSETS, 0.0),
        request={"temperature": 0.0, "seed": 123},
        declared_tokens={
            "shape": "5 conversations x 3 turns",
            "shared_system_prefix_tokens": "tokenizer-derived; target about 1000",
            "per_conversation_prefix_tokens": "tokenizer-derived; target about 1000",
            "new_user_tokens_per_turn": "tokenizer-derived; target about 150",
            "output_tokens_per_turn_pinned": 300,
            "source_shape": "adjacent viettel-ai-race sample-semantic-5x3",
            "content_variant": variant,
        },
    )


def official_shape_70x6(*, warmup: bool = False) -> WorkloadTrace:
    """Return the full 70-conversation x 6-turn racebench workload shape.

    The adjacent racebench publishes generated prompt content for its 1.2B
    target.  This profile preserves the published execution shape, arrival
    process, pinned output budget, and request options while generating
    deterministic local content for the 2.6B MPS target.
    """
    variant = "warmup" if warmup else "scored"
    shared_prefix = (
        "You are a careful production assistant. The following shared operating "
        f"brief is the {variant} workload context. Use it as background only; "
        "answer each user request from the complete conversation and distinguish "
        "source evidence from analysis.\n\n"
        + _records(
            "OFFICIAL-SHARED-" + variant.upper(),
            "the operating brief",
            "cross-team controls",
            count=22,
        )
    )
    prefixes = []
    messages = []
    for index in range(70):
        topic, _, detail = _official_topic(index)
        prefixes.append(
            _records(
                f"OFFICIAL-{variant.upper()}-{index + 1:02d}",
                topic,
                detail,
                count=20,
            )
        )
        messages.append(
            tuple(
                _turn_question(topic, detail, turn, warmup=warmup)
                for turn in range(6)
            )
        )
    return WorkloadTrace(
        name=f"official-shape-semantic-70x6-{variant}",
        model=MODEL,
        shared_system_prefix=shared_prefix,
        conversation_prefixes=tuple(prefixes),
        user_messages=tuple(messages),
        output_tokens_per_turn=tuple((300,) * 6 for _ in range(70)),
        arrival=Arrival(OFFICIAL_INITIAL_OFFSETS, 0.0),
        request={"temperature": 0.0, "seed": 123},
        declared_tokens={
            "shared_system_prefix_tokens": 1000,
            "per_conversation_prefix_tokens": 1000,
            "new_user_tokens_per_turn": [150] * 6,
            "output_tokens_per_turn_pinned": [300] * 6,
            "tokenizer": MODEL,
            "content_seed": 20260718,
            "arrival_assumption": {
                "process": "poisson-initial-conversation-arrivals",
                "rate_rps": 70.0,
                "seed": 42,
            },
            "source_shape": "adjacent viettel-ai-race official 70x6",
            "content_variant": variant,
        },
    )


def profile(name: str, *, warmup: bool = False) -> WorkloadTrace:
    if name == "semantic-5x3":
        return semantic_5x3(warmup=warmup)
    if name == "official-shape-70x6":
        return official_shape_70x6(warmup=warmup)
    raise ValueError(f"unknown workload profile: {name}")
