#!/usr/bin/env python3
"""Compile a Hugging Face tokenizer.json into tokenizer.llmopt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmopt_backend.tokenizer_archive import write_archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    tokenizer = json.loads(args.input.read_text(encoding="utf-8"))
    summary = write_archive(tokenizer, args.output)
    print(
        f"wrote {summary.token_count} tokens and {summary.merge_count} merges "
        f"to {args.output} ({summary.file_bytes} bytes)"
    )


if __name__ == "__main__":
    main()
