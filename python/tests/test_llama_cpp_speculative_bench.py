from __future__ import annotations

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
for arg in "$@"; do
  [ "$arg" = "--single-turn" ] && single_turn=true
  [ "$arg" = "--simple-io" ] && simple_io=true
  [ "$arg" = "--show-timings" ] && show_timings=true
  [ "$arg" = "--verbose" ] && verbose=true
done
[ "$single_turn" = true ]
[ "$simple_io" = true ]
[ "$show_timings" = true ]
[ "$verbose" = true ]
printf 'prompt eval time = 10.00 ms / 5 tokens (2.00 ms per token, 500.00 tokens per second)\n' >&2
printf 'eval time = 20.00 ms / 4 runs (5.00 ms per token, 200.00 tokens per second)\n' >&2
printf 'spec draft-mtp: statistics: #gen tokens = 4, #acc tokens = 3\n' >&2
""",
            )

            result = run_llama_cli(
                Path("model.gguf"),
                executable=executable,
                timeout_seconds=2.0,
            )

            self.assertEqual(result["prompt_tokens"], 5)
            self.assertEqual(result["generated_tokens"], 4)
            self.assertEqual(result["tpot_ms"], 5.0)
            self.assertEqual(result["tokens_per_sec"], 200.0)
            self.assertEqual(result["acceptance_rate"], 0.75)

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
