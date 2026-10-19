"""Date cutoff applies to batch compilation, but explicit files remain usable."""

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import utils
import lint

compiler = importlib.import_module("compile")


class CompileFromTests(unittest.TestCase):
    def test_cutoff_includes_today_and_future_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for day in ("2026-10-18", "2026-10-19", "2026-10-20"):
                (root / f"{day}.md").write_text("entry")
            with patch.object(utils, "DAILY_DIR", root):
                self.assertEqual(len(utils.list_raw_files()), 3)
                self.assertEqual([p.stem for p in utils.list_raw_files("2026-10-19")],
                                 ["2026-10-19", "2026-10-20"])
                with self.assertRaises(ValueError):
                    utils.list_raw_files("invalid")

    def test_batch_and_all_apply_cutoff(self):
        for flags in ([], ["--all"]):
            with patch.object(sys, "argv", ["compile.py", "--dry-run", *flags]), \
                 patch.object(compiler, "load_state", return_value={"compile_from": "2026-10-19"}), \
                 patch.object(compiler, "list_raw_files", return_value=[]) as files:
                compiler.main()
                files.assert_called_once_with(since="2026-10-19")

    def test_explicit_old_file_bypasses_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "2026-10-18.md"
            source.write_text("entry")
            with patch.object(sys, "argv", ["compile.py", "--dry-run", "--file", str(source)]), \
                 patch.object(compiler, "load_state", return_value={"compile_from": "2026-10-19"}), \
                 patch.object(compiler, "list_raw_files") as files:
                compiler.main()
                files.assert_not_called()

    def test_source_lint_respects_cutoff(self):
        for check in (lint.check_orphan_sources, lint.check_stale_articles):
            with patch.object(lint, "load_state", return_value={"compile_from": "2026-10-19"}), \
                 patch.object(lint, "list_raw_files", return_value=[]) as files:
                self.assertEqual(check(), [])
                files.assert_called_once_with(since="2026-10-19")
