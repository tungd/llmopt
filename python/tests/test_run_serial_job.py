"""Timeout cleanup must retain an honest receipt even when signaling is denied."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch


class SerialJobTests(unittest.TestCase):
    def run_timeout(self, waits, signal_error=None):
        script = Path(__file__).resolve().parents[2] / "bench/run_serial_job.py"
        child = Mock(pid=12345)
        child.wait.side_effect = waits
        child.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "job"
            with patch("sys.argv", [str(script), "--output", str(output),
                                    "--timeout", "1", "--", "probe"]), \
                 patch("subprocess.check_output", return_value=""), \
                 patch("subprocess.Popen", return_value=child) as launch, \
                 patch("os.killpg", side_effect=signal_error), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as result:
                    runpy.run_path(str(script), run_name="__main__")
                self.assertEqual(result.exception.code, 1)
                launch.assert_called_once()
            return json.loads(output.with_suffix(".json").read_text())

    def test_timeout_reaps_owned_process(self):
        receipt = self.run_timeout([subprocess.TimeoutExpired("probe", 1), -15])
        self.assertEqual(receipt["status"], "timed_out_and_reaped")
        self.assertEqual(receipt["exit_code"], -15)

    def test_denied_cleanup_preserves_unknown_outcome(self):
        receipt = self.run_timeout(
            [subprocess.TimeoutExpired("probe", 1)] * 3,
            PermissionError("signal denied"))
        self.assertEqual(receipt["status"], "timed_out_outcome_unknown")
        self.assertIsNone(receipt["exit_code"])
        self.assertEqual(receipt["cleanup_error"], "signal denied")
