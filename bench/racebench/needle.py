"""Natural needle-in-a-haystack prompt construction and validation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any

DEFAULT_LENGTHS = (7500, 9000, 16000, 30000)
DEFAULT_POSITIONS = (10, 50, 90)
NAMES = ("Mina", "An", "Ravi", "Linh", "Sofia", "Omar", "Nora", "Kenji")
ITEMS = (
    "cedar boards",
    "ceramic valves",
    "copper fasteners",
    "linen samples",
    "optical relays",
    "paper archives",
    "steel brackets",
    "water filters",
)
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
EXPECTED = "RAVEN-4271"
NEEDLE = (
    "\n\nCONTROL RECORD: The Orion archive access code is RAVEN-4271. "
    "This record supersedes every other access-code reference.\n\n"
)
QUESTION = (
    "\n\nQuestion: According to the CONTROL RECORD, what is the Orion archive "
    "access code? Reply with only the code."
)
INTRO = (
    "Read the following operations archive. Each numbered entry is ordinary "
    "background information. Use the CONTROL RECORD when answering the final "
    "question.\n\n"
)


@dataclass(frozen=True)
class ResponseEvaluation:
    retrieved: bool
    exact_response: bool


def evaluate_response(text: str, *, expected: str = EXPECTED) -> ResponseEvaluation:
    """Separate retrieval correctness from exact response-format compliance."""
    normalized = text.strip().upper().replace("`", "")
    normalized_expected = expected.strip().upper().replace("`", "")
    pattern = rf"(?<![A-Z0-9]){re.escape(normalized_expected)}(?!\d)"
    return ResponseEvaluation(
        retrieved=re.search(pattern, normalized) is not None,
        exact_response=normalized == normalized_expected,
    )


def chat_token_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("expected one rendered conversation")
        encoded = encoded[0]
    return list(encoded)


def archive_corpus(entries: int = 5000) -> str:
    rows = []
    for index in range(entries):
        rows.append(
            f"Archive item {index + 1:05d}: {NAMES[index % len(NAMES)]} "
            f"inspected {ITEMS[(index * 3) % len(ITEMS)]} on "
            f"{DAYS[(index * 7) % len(DAYS)]}; the shipment was stored in "
            f"bay {(index * 11) % 97 + 1} after routine verification.\n"
        )
    return "".join(rows)


def _rendered_count(tokenizer: Any, content: str) -> int:
    return len(chat_token_ids(tokenizer, [{"role": "user", "content": content}]))


def exact_natural_messages(
    tokenizer: Any,
    target_tokens: int,
    needle_percent: int,
    corpus: str,
) -> list[dict[str, str]]:
    if not 0 < needle_percent < 100:
        raise ValueError("needle position must be between 0 and 100")
    corpus_ids = tokenizer.encode(corpus, add_special_tokens=False)
    if len(corpus_ids) < target_tokens * 2:
        raise ValueError("natural archive corpus is too short")

    fixed_count = _rendered_count(tokenizer, INTRO + NEEDLE + QUESTION)
    wanted_corpus_tokens = target_tokens - fixed_count
    if wanted_corpus_tokens <= 0:
        raise ValueError(f"target length {target_tokens} is too short")

    before_count = wanted_corpus_tokens * needle_percent // 100
    after_count = wanted_corpus_tokens - before_count

    def candidate(delta: int) -> str:
        adjusted_after = max(0, after_count + delta)
        before = tokenizer.decode(
            corpus_ids[:before_count],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        after = tokenizer.decode(
            corpus_ids[before_count : before_count + adjusted_after],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return INTRO + before + NEEDLE + after + QUESTION

    delta = 0
    seen: set[int] = set()
    best: tuple[int, str, int] | None = None
    for _ in range(64):
        content = candidate(delta)
        actual = _rendered_count(tokenizer, content)
        distance = abs(actual - target_tokens)
        if best is None or distance < best[0]:
            best = (distance, content, actual)
        if actual == target_tokens:
            return [{"role": "user", "content": content}]
        correction = target_tokens - actual
        next_delta = delta + correction
        if next_delta in seen:
            break
        seen.add(delta)
        delta = next_delta

    assert best is not None
    if best[0] > 2:
        raise RuntimeError(
            f"could not fit {target_tokens} tokens; nearest was {best[2]}"
        )
    return [{"role": "user", "content": best[1]}]
