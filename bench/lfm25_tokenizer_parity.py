#!/usr/bin/env python3
"""Compare the native binary tokenizer and text-chat path with Transformers."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Sequence

from tokenizers import Tokenizer
from transformers import AutoTokenizer


TEXT_CASES = (
    "",
    "Hello, world!",
    "I'm testing contractions: we'd, they'll, YOU'RE.",
    "Xin chào Việt Nam — tối ưu hóa trình biên dịch.",
    "Numbers 1234567 and 000 42.",
    " leading  spaces\nnew line\r\n\ttrailing ",
    "Punctuation?! ... /\\ [] {} <tag>",
    "python Mathias",
    "emoji: 🧠⚙️🚀 and café",
    "<|im_start|>user\nXin chào<|im_end|>\n<|im_start|>assistant\n",
)

CHAT_CASES: tuple[tuple[tuple[str, str], ...], ...] = (
    (("user", "Hello"),),
    (("system", "Trả lời ngắn gọn."), ("user", "Thủ đô Việt Nam là gì?")),
    (("user", "One"), ("assistant", "First"), ("user", "Two")),
    (("system", ""), ("user", "No system")),
    (
        ("user", "One"),
        ("assistant", "<think>old</think> answer"),
        ("user", "Two"),
        ("assistant", "<think>new</think> final"),
        ("user", "Three"),
    ),
    (("tool", "plain tool result"),),
)


def native_ids(command: Sequence[str], runner: Path, archive: Path, text: str = "") -> list[int]:
    completed = subprocess.run(
        [str(runner), *command, str(archive)],
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace").strip())
    output = completed.stdout.decode("ascii").strip()
    return [] if not output else [int(value) for value in output.split(",")]


def check_text(reference: Tokenizer, runner: Path, archive: Path) -> None:
    for index, text in enumerate(TEXT_CASES):
        expected = reference.encode(text).ids
        actual = native_ids(("encode",), runner, archive, text)
        if actual != expected:
            raise AssertionError(
                f"text case {index} differs\nexpected: {expected}\nactual:   {actual}"
            )
        encoded = ",".join(map(str, actual))
        completed = subprocess.run(
            [str(runner), "decode", str(archive)],
            input=encoded.encode("ascii"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        expected_text = reference.decode(expected, skip_special_tokens=True)
        actual_text = completed.stdout.decode("utf-8")
        if actual_text != expected_text:
            raise AssertionError(
                f"decode case {index} differs: expected {expected_text!r}, got {actual_text!r}"
            )
        print(f"text-{index}: parity ({len(actual)} tokens)")


def check_chat(reference: object, runner: Path, archive: Path) -> None:
    for index, messages in enumerate(CHAT_CASES):
        dictionaries = [
            {"role": role, "content": content} for role, content in messages
        ]
        encoded = reference.apply_chat_template(
            dictionaries, tokenize=True, add_generation_prompt=True
        )
        expected = list(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)
        arguments = [str(runner), "chat", str(archive)]
        for role, content in messages:
            arguments.extend((role, content))
        completed = subprocess.run(
            arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                completed.stderr.decode("utf-8", errors="replace").strip()
            )
        actual = [int(value) for value in completed.stdout.decode("ascii").strip().split(",")]
        if actual != expected:
            raise AssertionError(
                f"chat case {index} differs\nexpected: {expected}\nactual:   {actual}"
            )
        print(f"chat-{index}: parity ({len(actual)} tokens)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--runner", default="_build/bin/llmopt-tokenize", type=Path)
    args = parser.parse_args()
    tokenizer_json = args.model / "tokenizer.json"
    check_text(Tokenizer.from_file(str(tokenizer_json)), args.runner, args.archive)
    check_chat(AutoTokenizer.from_pretrained(args.model, local_files_only=True), args.runner, args.archive)
    print(f"tokenizer parity: {len(TEXT_CASES)}/{len(TEXT_CASES)}")
    print(f"chat parity: {len(CHAT_CASES)}/{len(CHAT_CASES)}")


if __name__ == "__main__":
    main()
