from __future__ import annotations

import json
import stat
import tempfile
import time
import unittest
from pathlib import Path

from llama_cpp_speculative_bench import (
    BenchmarkError,
    BenchmarkTimeoutError,
    acquire_benchmark_lock,
    parse_running_llama_cli,
    run_llama_cli,
    summarize_campaigns,
    write_json_atomic,
)


class LlamaCppSpeculativeBenchTests(unittest.TestCase):
    def write_executable(self, directory: str, body: str) -> Path:
        path = Path(directory) / "fake-llama-cli"
        path.write_text("#!/bin/sh\nset -eu\n" + body, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def test_runner_is_single_turn_and_parses_real_timings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = self.write_executable(
                directory,
                """
single_turn=false
simple_io=false
show_timings=false
verbose=false
flash_attn=false
target_gpu_layers=false
draft_gpu_layers=false
seeded=false
output_file=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --single-turn) single_turn=true; shift ;;
    --simple-io) simple_io=true; shift ;;
    --show-timings) show_timings=true; shift ;;
    --verbose) verbose=true; shift ;;
    -fa) [ "$2" = "on" ] && flash_attn=true; shift 2 ;;
    -ngl) [ "$2" = "999" ] && target_gpu_layers=true; shift 2 ;;
    --spec-draft-ngl) [ "$2" = "999" ] && draft_gpu_layers=true; shift 2 ;;
    --seed) [ "$2" = "0" ] && seeded=true; shift 2 ;;
    -o) output_file=$2; shift 2 ;;
    *) shift ;;
  esac
done
[ "$single_turn" = true ]
[ "$simple_io" = true ]
[ "$show_timings" = true ]
[ "$verbose" = true ]
[ "$flash_attn" = true ]
[ "$target_gpu_layers" = true ]
[ "$draft_gpu_layers" = true ]
[ "$seeded" = true ]
[ -n "$output_file" ]
printf 'User:\nprobe\n\nAssistant:\nstable output\n' > "$output_file"
printf 'prompt eval time = 10.00 ms / 5 tokens (2.00 ms per token, 500.00 tokens per second)\n' >&2
printf 'eval time = 20.00 ms / 4 runs (5.00 ms per token, 200.00 tokens per second)\n' >&2
printf 'draft acceptance = 0.75000 (3 accepted / 4 generated), mean len = 2.50\n' >&2
""",
            )

            result = run_llama_cli(
                Path("model.gguf"),
                draft_path=Path("draft.gguf"),
                executable=executable,
                timeout_seconds=2.0,
            )

            self.assertEqual(result["prompt_tokens"], 5)
            self.assertEqual(result["generated_tokens"], 4)
            self.assertEqual(result["tpot_ms"], 5.0)
            self.assertEqual(result["tokens_per_sec"], 200.0)
            self.assertEqual(result["acceptance_rate"], 0.75)
            self.assertEqual(result["accepted_tokens"], 3)
            self.assertEqual(result["total_draft_tokens"], 4)
            self.assertEqual(result["mean_accepted_length"], 2.5)
            self.assertEqual(
                result["generated_text"],
                "User:\nprobe\n\nAssistant:\nstable output\n",
            )

    def test_runner_parses_compact_throughput_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = self.write_executable(
                directory,
                """
printf '[ Prompt: 70.3 t/s | Generation: 30.8 t/s ]\n' >&2
""",
            )

            result = run_llama_cli(
                Path("model.gguf"),
                executable=executable,
                n_predict=128,
                timeout_seconds=2.0,
            )

            self.assertIsNone(result["prompt_tokens"])
            self.assertIsNone(result["prompt_time_ms"])
            self.assertIsNone(result["generated_tokens"])
            self.assertIsNone(result["eval_time_ms"])
            self.assertAlmostEqual(result["tpot_ms"], 1000.0 / 30.8)
            self.assertEqual(result["tokens_per_sec"], 30.8)

    def test_process_scan_finds_only_matching_llama_cli(self) -> None:
        output = """
  101 /opt/local/bin/llama-cli -m model.gguf
  102 /usr/bin/python3 bench/llama_cpp_speculative_bench.py
  103 /usr/local/bin/other-cli
"""

        self.assertEqual(
            parse_running_llama_cli(output, Path("/opt/local/bin/llama-cli")),
            ["101 /opt/local/bin/llama-cli -m model.gguf"],
        )

    def test_campaign_summary_and_json_receipt_preserve_exact_counts(self) -> None:
        campaigns = [
            {
                "wall_time_s": 1.0,
                "prompt_tokens": 5,
                "prompt_time_ms": 10.0,
                "generated_tokens": 128,
                "eval_time_ms": 4000.0,
                "tpot_ms": 31.25,
                "tokens_per_sec": 32.0,
                "accepted_tokens": 81,
                "total_draft_tokens": 120,
                "acceptance_rate": 0.675,
                "mean_accepted_length": 3.0,
                "generated_text_sha256": "abc",
            },
            {
                "wall_time_s": 1.1,
                "prompt_tokens": 5,
                "prompt_time_ms": 10.0,
                "generated_tokens": 128,
                "eval_time_ms": 3900.0,
                "tpot_ms": 30.5,
                "tokens_per_sec": 33.0,
                "accepted_tokens": 82,
                "total_draft_tokens": 120,
                "acceptance_rate": 82 / 120,
                "mean_accepted_length": 3.1,
                "generated_text_sha256": "abc",
            },
            {
                "wall_time_s": 1.2,
                "prompt_tokens": 5,
                "prompt_time_ms": 10.0,
                "generated_tokens": 128,
                "eval_time_ms": 4100.0,
                "tpot_ms": 32.0,
                "tokens_per_sec": 31.0,
                "accepted_tokens": 80,
                "total_draft_tokens": 120,
                "acceptance_rate": 2 / 3,
                "mean_accepted_length": 2.9,
                "generated_text_sha256": "abc",
            },
        ]

        summary = summarize_campaigns(campaigns)

        self.assertEqual(summary["median_tpot_ms"], 31.25)
        self.assertEqual(summary["median_tokens_per_second"], 32.0)
        self.assertEqual(summary["campaigns"][0]["accepted_tokens"], 81)
        self.assertEqual(summary["campaigns"][0]["total_draft_tokens"], 120)
        self.assertEqual(summary["generated_text_sha256"], ["abc"])

        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "nested" / "receipt.json"
            write_json_atomic(receipt, {"results": summary})
            saved = json.loads(receipt.read_text(encoding="utf-8"))

        self.assertEqual(saved["results"]["median_tokens_per_second"], 32.0)
        self.assertEqual(saved["results"]["campaigns"][0]["accepted_tokens"], 81)

    def test_benchmark_lock_rejects_a_second_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "benchmark.lock"
            first = acquire_benchmark_lock(lock_path)
            try:
                with self.assertRaisesRegex(BenchmarkError, "already running"):
                    acquire_benchmark_lock(lock_path)
            finally:
                first.close()

    def test_timeout_terminates_the_child_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = self.write_executable(
                directory,
                """
trap 'exit 0' TERM INT
while :; do sleep 1; done
""",
            )
            started = time.monotonic()

            with self.assertRaisesRegex(BenchmarkTimeoutError, "timed out"):
                run_llama_cli(
                    Path("model.gguf"),
                    executable=executable,
                    timeout_seconds=0.1,
                    termination_grace_seconds=0.2,
                )

            self.assertLess(time.monotonic() - started, 2.0)

    def test_nonzero_exit_is_not_reported_as_a_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = self.write_executable(directory, "echo broken >&2\nexit 7\n")

            with self.assertRaisesRegex(BenchmarkError, "exit code 7"):
                run_llama_cli(
                    Path("model.gguf"),
                    executable=executable,
                    timeout_seconds=2.0,
                )

    def test_missing_timing_output_is_not_fabricated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executable = self.write_executable(directory, "echo no-timings\n")

            with self.assertRaisesRegex(BenchmarkError, "timing output"):
                run_llama_cli(
                    Path("model.gguf"),
                    executable=executable,
                    timeout_seconds=2.0,
                )


if __name__ == "__main__":
    unittest.main()
