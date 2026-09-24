from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from audit_coverage_receipts import audit_receipt, audit_paths


REPO_ROOT = Path(__file__).resolve().parents[2]
RECEIPT_ROOT = REPO_ROOT / "_artifacts/gemma4-12b-qat-uoc-g4p12-2026-09-14"


class AuditCoverageReceiptsTests(unittest.TestCase):
    def test_real_receipt_reports_coverage_and_queue_counts(self) -> None:
        observations = audit_paths([RECEIPT_ROOT], package_checker=None)
        self.assertEqual(len(observations), 2)
        for observation in observations:
            self.assertEqual(observation["schema_version"], 1)
            self.assertGreater(observation["kernel_count"], 0)
            self.assertGreater(observation["lowering"]["unified_count"], 0)
            self.assertEqual(observation["coverage"]["uncovered_count"], 0)
            self.assertGreater(observation["aot_queue"]["stages"], 0)

    def test_nested_uncovered_and_boundary_data_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optimization-receipt.json"
            path.write_text(
                json.dumps(
                    {
                        "target": {"device_name": "test"},
                        "fusion": {
                            "decisions": [
                                {
                                    "status": "cut",
                                    "boundaries": [{"kind": "unsupported-operation"}],
                                }
                            ],
                            "events": [{"kind": "gated-linear"}],
                            "uncovered": [{"reason": "unsupported-pattern"}],
                        },
                        "lowering": {
                            "unified-count": 1,
                            "accepted-count": 2,
                            "cut-count": 1,
                            "legacy-count": 0,
                            "legacy-invocation-count": 0,
                            "legacy-rewrite-count": 0,
                        },
                        "kernels": [{"operation": "linear"}],
                        "linear-tactics": [
                            {"chosen": {"name": "q4_0_x8"}}
                        ],
                        "aot-queue": {
                            "stages": [1],
                            "dependencies": [1, 2],
                            "barriers": [3],
                        },
                    }
                ),
                encoding="utf-8",
            )
            observation = audit_receipt(path, package_checker=None)
            self.assertEqual(observation["coverage"]["uncovered_count"], 1)
            self.assertEqual(
                observation["coverage"]["uncovered_reasons"],
                {"unsupported-pattern": 1},
            )
            self.assertEqual(
                observation["coverage"]["boundary_kinds"],
                {"unsupported-operation": 1},
            )
            self.assertEqual(
                observation["aot_queue"],
                {"stages": 1, "dependencies": 2, "barriers": 1},
            )


if __name__ == "__main__":
    unittest.main()
