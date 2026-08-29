"""Deterministic binary transport for a byte-level BPE tokenizer."""

from __future__ import annotations

import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


MAGIC = b"LLMOPTTK"
VERSION = 1
PROFILE_BPE = 1
LFM25_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
_HEADER = struct.Struct("<8sHHIII")
_TOKEN = struct.Struct("<IB3xI")
_PAIR = struct.Struct("<II")


@dataclass(frozen=True)
class ArchiveSummary:
    token_count: int
    merge_count: int
    maximum_token_id: int
    file_bytes: int


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"tokenizer {label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"tokenizer {label} must be a sequence")
    return value


def _merge(value: Any) -> tuple[str, str]:
    if isinstance(value, str):
        parts = value.split(" ")
    else:
        parts = list(_sequence(value, "merge"))
    if len(parts) != 2 or not all(isinstance(part, str) for part in parts):
        raise ValueError(f"invalid tokenizer merge: {value!r}")
    return parts[0], parts[1]


def _validate_lfm25_profile(tokenizer: Mapping[str, Any], model: Mapping[str, Any]) -> None:
    if tokenizer.get("normalizer") is not None:
        raise ValueError("LFM tokenizer profile requires no normalizer")
    expected_pre_tokenizer = {
        "type": "Sequence",
        "pretokenizers": [
            {
                "type": "Split",
                "pattern": {"Regex": LFM25_SPLIT_PATTERN},
                "behavior": "Isolated",
                "invert": False,
            },
            {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": False,
            },
        ],
    }
    if tokenizer.get("pre_tokenizer") != expected_pre_tokenizer:
        raise ValueError("tokenizer does not match the LFM pre-tokenizer profile")
    expected_model_options = {
        "dropout": None,
        "unk_token": None,
        "continuing_subword_prefix": None,
        "end_of_word_suffix": None,
        "fuse_unk": False,
        "byte_fallback": False,
        "ignore_merges": False,
    }
    for option, expected in expected_model_options.items():
        if model.get(option) != expected:
            raise ValueError(f"unsupported LFM BPE option {option}={model.get(option)!r}")
    for raw in _sequence(tokenizer.get("added_tokens", []), "added tokens"):
        added = _mapping(raw, "added token")
        for option in ("single_word", "lstrip", "rstrip"):
            if added.get(option, False) is not False:
                raise ValueError(f"unsupported added-token option {option}=true")
    post_processor = _mapping(tokenizer.get("post_processor"), "post processor")
    processors = _sequence(post_processor.get("processors"), "post processors")
    if post_processor.get("type") != "Sequence" or len(processors) != 2:
        raise ValueError("LFM tokenizer requires its two-stage post processor")
    template = _mapping(processors[1], "template post processor")
    special_tokens = _mapping(template.get("special_tokens"), "special tokens")
    bos = _mapping(special_tokens.get("<|startoftext|>"), "BOS token")
    if template.get("type") != "TemplateProcessing" or bos.get("ids") != [1]:
        raise ValueError("LFM tokenizer requires BOS token 1")


def encode_archive(tokenizer: Mapping[str, Any], strict_lfm: bool = False) -> tuple[bytes, ArchiveSummary]:
    model = _mapping(tokenizer.get("model"), "model")
    if model.get("type") != "BPE":
        raise ValueError(f"unsupported tokenizer model: {model.get('type')!r}")
    if strict_lfm:
        _validate_lfm25_profile(tokenizer, model)
    else:
        try:
            _validate_lfm25_profile(tokenizer, model)
        except Exception:
            pass

    by_id: dict[int, tuple[str, int]] = {}
    vocab = _mapping(model.get("vocab"), "vocabulary")
    for token, encoded_id in vocab.items():
        if not isinstance(token, str) or not isinstance(encoded_id, int):
            raise TypeError("tokenizer vocabulary entries must be string-to-int")
        if encoded_id < 0 or encoded_id > 0xFFFF_FFFF:
            raise ValueError(f"token id is outside uint32: {encoded_id}")
        previous = by_id.get(encoded_id)
        if previous is not None and previous[0] != token:
            raise ValueError(f"token id {encoded_id} has multiple spellings")
        by_id[encoded_id] = (token, 0)

    for raw in _sequence(tokenizer.get("added_tokens", []), "added tokens"):
        added = _mapping(raw, "added token")
        token = added.get("content")
        encoded_id = added.get("id")
        if not isinstance(token, str) or not isinstance(encoded_id, int):
            raise TypeError("added token requires string content and integer id")
        previous = by_id.get(encoded_id)
        if previous is not None and previous[0] != token:
            raise ValueError(f"added token id {encoded_id} conflicts with vocabulary")
        flags = 0x01 | (0x02 if added.get("special") is True else 0)
        by_id[encoded_id] = (token, flags)

    if not by_id:
        raise ValueError("cannot export an empty tokenizer vocabulary")
    merges = [_merge(value) for value in _sequence(model.get("merges"), "merges")]
    if len(set(merges)) != len(merges):
        raise ValueError("tokenizer merge list contains duplicates")

    maximum = max(by_id)
    parts = [
        _HEADER.pack(
            MAGIC, VERSION, PROFILE_BPE, len(by_id), len(merges), maximum
        )
    ]
    for encoded_id, (token, flags) in sorted(by_id.items()):
        encoded = token.encode("utf-8")
        parts.append(_TOKEN.pack(encoded_id, flags, len(encoded)))
        parts.append(encoded)
    for left, right in merges:
        encoded_left = left.encode("utf-8")
        encoded_right = right.encode("utf-8")
        parts.append(_PAIR.pack(len(encoded_left), len(encoded_right)))
        parts.append(encoded_left)
        parts.append(encoded_right)
    payload = b"".join(parts)
    return payload, ArchiveSummary(len(by_id), len(merges), maximum, len(payload))


def write_archive(
    tokenizer: Mapping[str, Any], destination: str | Path
) -> ArchiveSummary:
    payload, summary = encode_archive(tokenizer)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return summary


__all__ = [
    "MAGIC",
    "VERSION",
    "PROFILE_BPE",
    "LFM25_SPLIT_PATTERN",
    "ArchiveSummary",
    "encode_archive",
    "write_archive",
]
