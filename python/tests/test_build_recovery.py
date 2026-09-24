from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class BuildRecoveryTests(unittest.TestCase):
    def test_webs_artifacts_consume_installed_interface_snapshot(self) -> None:
        targets = (
            "_build/ocaml/webs_iomux.cmi",
            "_build/ocaml/webs_iomux.cmx",
            "_build/ocaml/webs_connector_codec.cmx",
        )
        for target in targets:
            completed = subprocess.run(
                ["ninja", "-f", "ninja.build", "-t", "query", target],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                "_build/ocaml/ocaml-package-interfaces.stamp",
                completed.stdout,
            )

    def test_snapshot_rule_covers_webs_and_iomux_interfaces(self) -> None:
        build = (REPO_ROOT / "ninja.build").read_text(encoding="utf-8")
        self.assertIn("for package in webs iomux", build)
        self.assertIn("*.cmi", build)
        self.assertIn("*.cmti", build)
        self.assertIn("restat = 1", build)


if __name__ == "__main__":
    unittest.main()
