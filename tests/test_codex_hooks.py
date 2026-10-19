"""Codex hook capture tests with temporary transcripts and no model calls."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import codex_capture
from llm import LLMResult

spec = importlib.util.spec_from_file_location("codex_hook", ROOT / "hooks/codex-session.py")
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)


def message(text, role="user"):
    return json.dumps({"type": "response_item", "payload": {
        "type": "message", "role": role,
        "content": [{"type": "input_text" if role == "user" else "output_text", "text": text}],
    }}) + "\n"


class CaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_incremental_capture_and_duplicate_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            daily = root / "daily"
            snapshot = root / "snapshot.jsonl"
            first = message("Use SQLite")
            with patch.object(codex_capture, "QUEUE_DIR", root / "state"), \
                 patch.object(codex_capture, "DAILY_DIR", daily), \
                 patch("llm.run_llm", AsyncMock(return_value=LLMResult("Decision: SQLite", {}))) as run:
                snapshot.write_text(first)
                await codex_capture.capture(snapshot, "session/unsafe")
                snapshot.write_text(first)
                await codex_capture.capture(snapshot, "session/unsafe")
                self.assertEqual(run.await_count, 1)
                snapshot.write_text(first + message("Add backups"))
                await codex_capture.capture(snapshot, "session/unsafe")
                self.assertEqual(run.await_count, 2)
                prompt = run.call_args.args[0]
                self.assertIn("Add backups", prompt)
                self.assertNotIn("Use SQLite", prompt)
                self.assertEqual(next(daily.glob("*.md")).read_text().count("### Codex Session"), 2)

    async def test_failed_capture_retains_snapshot_and_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.jsonl"
            snapshot.write_text(message("Important lesson"))
            with patch.object(codex_capture, "QUEUE_DIR", root / "state"), \
                 patch.object(codex_capture, "DAILY_DIR", root / "daily"), \
                 patch("llm.run_llm", AsyncMock(side_effect=RuntimeError("offline"))):
                with self.assertRaises(RuntimeError):
                    await codex_capture.capture(snapshot, "session")
                self.assertTrue(snapshot.exists())
                self.assertEqual(list((root / "state").glob("*.json")), [])
                self.assertFalse((root / "daily").exists())

    async def test_no_useful_content_advances_cursor_without_daily(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.jsonl"
            snapshot.write_text(message("Hello"))
            with patch.object(codex_capture, "QUEUE_DIR", root / "state"), \
                 patch.object(codex_capture, "DAILY_DIR", root / "daily"), \
                 patch("llm.run_llm", AsyncMock(return_value=LLMResult("FLUSH_OK", {}))):
                await codex_capture.capture(snapshot, "session")
                self.assertFalse(snapshot.exists())
                self.assertFalse((root / "daily").exists())
                self.assertEqual(len(list((root / "state").glob("*.json"))), 1)

    def test_parser_skips_tools_and_duplicate_events_and_partial_tail(self):
        self.assertEqual(codex_capture.message_text({"type": "event_msg", "payload": {
            "type": "user_message", "message": "duplicate"}}), "")
        self.assertEqual(codex_capture.message_text(json.loads(message("policy", "developer"))), "")
        self.assertEqual(codex_capture.message_text(json.loads(message("# AGENTS.md instructions x"))), "")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transcript.jsonl"
            complete = message("Привет")
            path.write_text(complete + '{"partial":')
            text, offset = codex_capture.read_pending(path, 0)
            self.assertIn("Привет", text)
            self.assertEqual(offset, len(complete.encode()))

    def test_hook_snapshots_and_detaches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "transcript.jsonl"
            source.write_text(message("Decision"))
            with patch.dict(os.environ, {}, clear=True), \
                 patch.object(hook, "QUEUE_DIR", root / "queue"), \
                 patch.object(hook, "SCRIPTS_DIR", root), \
                 patch.object(hook.subprocess, "Popen") as popen:
                hook.handle_event({"hook_event_name": "Stop", "session_id": "session",
                                   "transcript_path": str(source)})
                self.assertTrue(popen.call_args.kwargs["start_new_session"])
                self.assertEqual(next((root / "queue").glob("*.jsonl")).read_text(), source.read_text())

    def test_internal_sdk_session_does_not_trigger_capture(self):
        with patch.dict(os.environ, {"CODEX_MEMORY_INTERNAL": "1"}), \
             patch.object(hook.subprocess, "Popen") as popen, \
             patch("builtins.print") as output:
            hook.handle_event({"hook_event_name": "Stop"})
            hook.handle_event({"hook_event_name": "SessionStart"})
            popen.assert_not_called()
            output.assert_not_called()
