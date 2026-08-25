from __future__ import annotations

import unittest

import torch

from lfm25_reference_tokens import compare_f16_rows, last_f16_row


class ReferenceLogitsTest(unittest.TestCase):
    def test_last_f16_row_extracts_only_the_final_vocabulary_row(self) -> None:
        logits = torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=torch.float32
        )

        actual = torch.frombuffer(
            bytearray(last_f16_row(logits)), dtype=torch.float16
        )

        self.assertTrue(torch.equal(actual, torch.tensor([4.0, 5.0, 6.0])))

    def test_compare_f16_rows_reports_numeric_and_argmax_parity(self) -> None:
        reference = torch.tensor([1.0, 3.0, 2.0], dtype=torch.float16)
        candidate = torch.tensor([1.0, 3.0, 1.5], dtype=torch.float16)

        comparison = compare_f16_rows(
            reference.numpy().tobytes(), candidate.numpy().tobytes()
        )

        self.assertFalse(comparison["exact"])
        self.assertEqual(comparison["max_abs"], 0.5)
        self.assertAlmostEqual(comparison["mean_abs"], 1.0 / 6.0)
        self.assertEqual(comparison["reference_argmax"], 1)
        self.assertEqual(comparison["candidate_argmax"], 1)
        self.assertTrue(comparison["argmax_parity"])

    def test_compare_f16_rows_rejects_wrong_byte_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 4 bytes"):
            compare_f16_rows(b"\x00\x00\x01\x00", b"\x00\x00")


if __name__ == "__main__":
    unittest.main()
