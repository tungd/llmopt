import json
import struct
import tempfile
import unittest
from pathlib import Path

from llmopt_backend.tokenizer_archive import (
    LFM25_SPLIT_PATTERN,
    MAGIC,
    PROFILE_BPE,
    encode_archive,
    write_archive,
)


class TokenizerArchiveTests(unittest.TestCase):
    def fixture(self):
        return {
            "normalizer": None,
            "pre_tokenizer": {
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
            },
            "post_processor": {
                "type": "Sequence",
                "processors": [
                    {"type": "ByteLevel"},
                    {
                        "type": "TemplateProcessing",
                        "special_tokens": {"<|startoftext|>": {"ids": [1]}},
                    },
                ],
            },
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": None,
                "continuing_subword_prefix": None,
                "end_of_word_suffix": None,
                "fuse_unk": False,
                "byte_fallback": False,
                "ignore_merges": False,
                "vocab": {"<bos>": 0, "a": 1, "b": 2, "ab": 3},
                "merges": [["a", "b"]],
            },
            "added_tokens": [
                {
                    "id": 0,
                    "content": "<bos>",
                    "special": True,
                },
                {
                    "id": 4,
                    "content": "python",
                    "special": False,
                },
            ],
        }

    def test_archive_is_deterministic_binary(self):
        first, summary = encode_archive(self.fixture())
        second, _ = encode_archive(self.fixture())
        self.assertEqual(first, second)
        self.assertTrue(first.startswith(MAGIC))
        self.assertEqual(summary.token_count, 5)
        self.assertEqual(summary.merge_count, 1)
        self.assertEqual(summary.maximum_token_id, 4)
        magic, version, profile, tokens, merges, maximum = struct.unpack_from(
            "<8sHHIII", first
        )
        self.assertEqual((magic, version, profile), (MAGIC, 1, PROFILE_BPE))
        self.assertEqual((tokens, merges, maximum), (5, 1, 4))
        with self.assertRaises((UnicodeDecodeError, json.JSONDecodeError)):
            json.loads(first.decode("utf-8"))

    def test_atomic_file_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.llmopt"
            summary = write_archive(self.fixture(), path)
            self.assertEqual(path.stat().st_size, summary.file_bytes)
            self.assertEqual(path.read_bytes()[:8], MAGIC)

    def test_rejects_conflicting_added_id(self):
        fixture = self.fixture()
        fixture["added_tokens"][0]["content"] = "different"
        with self.assertRaisesRegex(ValueError, "conflicts"):
            encode_archive(fixture)

    def test_accepts_non_lfm_bpe_by_default(self):
        fixture = self.fixture()
        fixture["pre_tokenizer"] = {"type": "ByteLevel"}
        payload, summary = encode_archive(fixture)
        self.assertTrue(payload.startswith(MAGIC))
        self.assertEqual(summary.token_count, 5)

    def test_strict_lfm_probe_rejects_non_lfm_pretokenizer(self):
        fixture = self.fixture()
        fixture["pre_tokenizer"] = {"type": "ByteLevel"}
        with self.assertRaisesRegex(ValueError, "LFM pre-tokenizer"):
            encode_archive(fixture, strict_lfm=True)


if __name__ == "__main__":
    unittest.main()
