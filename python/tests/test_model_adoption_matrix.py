from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from model_adoption_matrix import DEFAULT_MATRIX, load_matrix, validate_matrix


class ModelAdoptionMatrixTests(unittest.TestCase):
    def test_seeded_matrix_is_valid(self) -> None:
        matrix = load_matrix(DEFAULT_MATRIX)
        self.assertEqual(validate_matrix(matrix), [])
        self.assertEqual(len(matrix["models"]), 5)
        self.assertEqual(len(matrix["next_cohort"]), 4)
        self.assertIsNone(matrix["models"][0]["coverage"]["new_backend_tactics"])

    def test_unknown_measurements_are_allowed_but_invalid_ratio_is_not(self) -> None:
        matrix = load_matrix(DEFAULT_MATRIX)
        changed = copy.deepcopy(matrix)
        changed["models"][0]["coverage"]["opaque_commands"] = None
        changed["models"][0]["performance"][0]["llmopt"] = None
        changed["models"][0]["performance"][0]["ratio"] = None
        self.assertEqual(validate_matrix(changed), [])

        changed = copy.deepcopy(matrix)
        changed["models"][0]["performance"][0]["ratio"] += 0.1
        self.assertTrue(
            any("ratio does not equal" in error for error in validate_matrix(changed))
        )

    def test_missing_evidence_is_rejected(self) -> None:
        matrix = load_matrix(DEFAULT_MATRIX)
        del matrix["models"][0]["evidence"]
        self.assertTrue(
            any("evidence must be a non-empty list" in error for error in validate_matrix(matrix))
        )

    def test_json_is_stable_and_source_receipt_is_present(self) -> None:
        raw = Path(DEFAULT_MATRIX).read_text(encoding="utf-8")
        self.assertEqual(json.loads(raw)["schema_version"], 1)
        for resource in (
            "bench/results/compiler-generalization-slice-31-2026-08-30.json",
            "bench/results/smollm-slice31-llamacpp-matched-trace-probe-2026-09-01.json",
            "bench/results/uoc-gemma4-performance-exp0173-2026-09-14.txt",
        ):
            self.assertTrue((Path(__file__).resolve().parents[2] / resource).exists())


if __name__ == "__main__":
    unittest.main()
