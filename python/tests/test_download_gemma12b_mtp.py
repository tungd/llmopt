from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from download_gemma12b_mtp import verify_file


class DownloadGemma12bMtpTests(unittest.TestCase):
    def test_verify_file_checks_size_and_sha256(self) -> None:
        payload = b"pinned-gguf-fixture"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.gguf"
            path.write_bytes(payload)

            receipt = verify_file(
                path,
                expected_size=len(payload),
                expected_sha256=hashlib.sha256(payload).hexdigest(),
            )

            self.assertEqual(receipt["size"], len(payload))
            self.assertEqual(receipt["sha256"], hashlib.sha256(payload).hexdigest())

    def test_verify_file_rejects_size_mismatch_before_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.gguf"
            path.write_bytes(b"short")

            with self.assertRaisesRegex(ValueError, "expected 6 bytes"):
                verify_file(path, expected_size=6, expected_sha256="unused")

    def test_verify_file_rejects_sha256_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.gguf"
            path.write_bytes(b"payload")

            with self.assertRaisesRegex(ValueError, "expected SHA256"):
                verify_file(path, expected_size=7, expected_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()
