from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from bench_speculative_sustained import (
    acquire_benchmark_lock,
    build_comparison_report,
    summarize_campaigns,
    write_json_atomic,
)


class BenchSpeculativeSustainedTests(unittest.TestCase):
    def test_summarize_campaigns_computes_medians_and_acceptance(self) -> None:
        campaigns = [
            {
                "tpot_ms": 30.0,
                "tokens_per_second": 33.33,
                "prompt_time_ms": 500.0,
                "eval_time_ms": 3840.0,
                "wall_time_s": 4.5,
                "accepted_tokens": 90,
                "total_draft_tokens": 130,
                "acceptance_rate": 0.692,
                "mean_accepted_length": 3.7,
            },
            {
                "tpot_ms": 31.0,
                "tokens_per_second": 32.26,
                "prompt_time_ms": 510.0,
                "eval_time_ms": 3968.0,
                "wall_time_s": 4.6,
                "accepted_tokens": 92,
                "total_draft_tokens": 137,
                "acceptance_rate": 0.671,
                "mean_accepted_length": 3.6,
            },
            {
                "tpot_ms": 32.0,
                "tokens_per_second": 31.25,
                "prompt_time_ms": 520.0,
                "eval_time_ms": 4096.0,
                "wall_time_s": 4.7,
                "accepted_tokens": 94,
                "total_draft_tokens": 140,
                "acceptance_rate": 0.671,
                "mean_accepted_length": 3.6,
            },
        ]
        summary = summarize_campaigns(campaigns)
        self.assertAlmostEqual(summary["median_tpot_ms"], 31.0)
        self.assertAlmostEqual(summary["median_tokens_per_second"], 32.26)
        self.assertAlmostEqual(summary["median_acceptance_rate"], 0.671)
        self.assertEqual(summary["total_accepted_tokens"], 276)
        self.assertEqual(summary["total_draft_tokens"], 407)

    def test_write_json_atomic_and_build_report(self) -> None:
        seq = {
            "median_tokens_per_second": 32.0,
            "median_tpot_ms": 31.25,
            "mean_tokens_per_second": 32.0,
            "mean_tpot_ms": 31.25,
            "mean_prompt_time_ms": 500.0,
            "mean_eval_time_ms": 4000.0,
            "mean_wall_time_s": 4.5,
            "stdev_tpot_ms": 0.0,
            "campaigns": [],
        }
        mtp = {
            "median_tokens_per_second": 20.0,
            "median_tpot_ms": 50.0,
            "mean_tokens_per_second": 20.0,
            "mean_tpot_ms": 50.0,
            "mean_prompt_time_ms": 500.0,
            "mean_eval_time_ms": 6000.0,
            "mean_wall_time_s": 6.5,
            "stdev_tpot_ms": 0.0,
            "campaigns": [],
            "median_acceptance_rate": 0.67,
            "mean_acceptance_rate": 0.67,
            "mean_accepted_length": 3.6,
            "total_accepted_tokens": 92,
            "total_draft_tokens": 137,
        }
        args = argparse.Namespace(
            model_repo="unsloth/gemma-4-12B-it-qat-GGUF",
            tokens=128,
            campaigns=3,
            seed=0,
            draft_n_max=4,
        )
        baseline = {
            "results": {
                "sequential": {"median_tokens_per_second": 31.13},
                "mtp": {"median_tokens_per_second": 18.55},
            }
        }
        report = build_comparison_report(
            llmopt_seq=seq,
            llmopt_mtp=mtp,
            llamacpp_baseline=baseline,
            args=args,
            model_path=Path("target.gguf"),
            draft_path=Path("draft.gguf"),
        )
        self.assertEqual(report["kind"], "llmopt-gemma4-12b-mtp-sustained-benchmark")
        self.assertIn("baseline_comparison", report)
        self.assertAlmostEqual(
            report["baseline_comparison"]["llmopt_vs_llamacpp_sequential_speedup"],
            32.0 / 31.13,
            places=3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "report.json"
            write_json_atomic(out_file, report)
            loaded = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(loaded["kind"], "llmopt-gemma4-12b-mtp-sustained-benchmark")


if __name__ == "__main__":
    unittest.main()
