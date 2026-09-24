from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analyze_projection_cost_model import (
    CALIBRATION_RECEIPTS,
    DEFAULT_RECEIPTS,
    AnalysisError,
    analyze,
    _parse_measurements,
    _predict_candidate,
    _predict_shape,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class AnalyzeProjectionCostModelTests(unittest.TestCase):
    def test_historical_receipts_expose_precalibration_mismatch(self) -> None:
        report = analyze(DEFAULT_RECEIPTS)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(
            report["source_sha256"],
            "151cede6aa87ee64636ac9ccec46197e933598e16ecc1038993f67a7e5d321ce",
        )
        self.assertEqual(len(report["shapes"]), 4)
        self.assertTrue(all(shape["status"] == "complete" for shape in report["shapes"]))
        self.assertTrue(
            all(
                row["parity"] == "PASS" and row["samples"] == 2
                for shape in report["shapes"]
                for row in shape["observed"]
            )
        )

        evidence = report["ffn_down_evidence"]
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertEqual(evidence["model_winner"], 16)
        self.assertEqual(evidence["observed_winner"], 16)
        self.assertTrue(evidence["winner_match"])
        self.assertGreater(evidence["x16_vs_x8_speedup"], 1.0)

    def test_selector_legality_excludes_partial_tile_candidates(self) -> None:
        self.assertEqual([row["tile"] for row in _predict_shape(68, 96)], [4])

    def test_three_run_calibration_matches_all_measured_shapes(self) -> None:
        report = analyze(CALIBRATION_RECEIPTS)
        self.assertEqual(len(report["shapes"]), 8)
        self.assertTrue(all(shape["status"] == "complete" for shape in report["shapes"]))
        self.assertTrue(
            all(
                shape["winner_match"]
                for shape in report["shapes"]
                if shape["n"] != 68 or shape["k"] != 96
            )
        )
        evidence = report["ffn_down_evidence"]
        assert evidence is not None
        self.assertEqual(evidence["model_winner"], 16)
        self.assertEqual(evidence["observed_winner"], 16)
        self.assertTrue(evidence["winner_match"])

    def test_candidate_prediction_mirrors_cost_model_inputs(self) -> None:
        x8 = _predict_candidate(3840, 15360, 8)
        x16 = _predict_candidate(3840, 15360, 16)
        self.assertEqual(x8["workgroups"], 60)
        self.assertEqual(x16["workgroups"], 30)
        self.assertEqual(x8["registers_per_thread"], 32)
        self.assertEqual(x16["registers_per_thread"], 56)
        self.assertEqual(x8["occupancy_factor"], 1.0)
        self.assertLess(x16["occupancy_factor"], 1.0)
        self.assertEqual(x8["bandwidth_utilization"], 1.0)
        self.assertLess(x16["bandwidth_utilization"], 1.0)
        self.assertEqual(x8["weight_traffic_bytes"], 33177600.0)
        self.assertEqual(x16["weight_traffic_bytes"], 33177600.0)
        self.assertLess(x16["total_us"], x8["total_us"])
        self.assertAlmostEqual(
            x8["memory_bytes"],
            x8["weight_traffic_bytes"]
            + x8["activation_traffic_bytes"]
            + x8["output_bytes"],
        )

    def test_parser_rejects_failed_parity_and_unaligned_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiles.txt"
            path.write_text(
                "\n".join(
                    [
                        "n=4 k=32 tile=x4 groups=1 weight_bytes=72 "
                        "median_us=1.0 min_us=1.0 max_us=1.0 GBps=0.1 "
                        "parity=PASS(3/4)",
                        "n=4 k=33 tile=x4 groups=1 weight_bytes=72 "
                        "median_us=1.0 min_us=1.0 max_us=1.0 GBps=0.1 "
                        "parity=PASS(4/4)",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(AnalysisError):
                _parse_measurements(path)


if __name__ == "__main__":
    unittest.main()
