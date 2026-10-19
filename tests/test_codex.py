"""Offline regression checks for the Codex migration."""

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import llm
import query
import lint

compiler = importlib.import_module("compile")


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_response_and_usage(self):
        from openai_codex import ApprovalMode, Sandbox
        from openai_codex.api import AsyncThread

        thread = MagicMock(spec=AsyncThread)
        thread.run = AsyncMock(return_value=SimpleNamespace(
            status=SimpleNamespace(value="completed"), error=None,
            final_response=" answer ",
            usage=SimpleNamespace(total=SimpleNamespace(
                model_dump=lambda: {"input_tokens": 12, "output_tokens": 3})),
        ))
        with patch("openai_codex.AsyncCodex", autospec=True) as sdk:
            client = sdk.return_value.__aenter__.return_value
            client.thread_start.return_value = thread
            result = await llm.run_llm("question", tools=False)
            self.assertEqual(result.text, "answer")
            self.assertEqual(result.usage["input_tokens"], 12)
            options = client.thread_start.call_args.kwargs
            self.assertEqual(options["sandbox"], Sandbox.read_only)
            self.assertEqual(options["approval_mode"], ApprovalMode.deny_all)
            self.assertTrue(options["ephemeral"])
            self.assertIn("features.shell_tool=false", sdk.call_args.args[0].config_overrides)

    async def test_rejects_interrupted_failed_and_empty_results(self):
        for status, text in (("interrupted", "partial"), ("failed", "partial"),
                             ("completed", " ")):
            with self.subTest(status=status, text=text):
                with patch("openai_codex.AsyncCodex", autospec=True) as sdk:
                    thread = sdk.return_value.__aenter__.return_value.thread_start.return_value
                    thread.run = AsyncMock(return_value=SimpleNamespace(
                        status=SimpleNamespace(value=status), error=None,
                        final_response=text, usage=None,
                    ))
                    with self.assertRaises(RuntimeError):
                        await llm.run_llm("question")

    async def test_writer_uses_wiki_workspace(self):
        from openai_codex import Sandbox

        with tempfile.TemporaryDirectory() as tmp, patch("llm.KNOWLEDGE_DIR", Path(tmp)):
            with patch("openai_codex.AsyncCodex", autospec=True) as sdk:
                client = sdk.return_value.__aenter__.return_value
                client.thread_start.return_value.run = AsyncMock(return_value=SimpleNamespace(
                    status=SimpleNamespace(value="completed"), error=None,
                    final_response="done", usage=None,
                ))
                await llm.run_llm("compile", write=True)
                options = client.thread_start.call_args.kwargs
                self.assertEqual(options["cwd"], tmp)
                self.assertEqual(options["sandbox"], Sandbox.workspace_write)


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_compile_verifies_files_before_marking_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "2026-10-19.md"
            source.write_text("A lesson")
            schema = root / "AGENTS.md"
            schema.write_text("Schema")
            wiki = root / "wiki"
            (wiki / "concepts").mkdir(parents=True)
            state = {"total_cost": 1.25}
            result = llm.LLMResult('{"articles":["concepts/lesson.md"]}', {"input_tokens": 10})
            with patch.multiple(compiler, AGENTS_FILE=schema, KNOWLEDGE_DIR=wiki), \
                 patch.object(compiler, "read_wiki_index", return_value="Index"), \
                 patch.object(compiler, "now_iso", return_value="2026-10-19T12:00:00"), \
                 patch.object(compiler, "save_state") as save, \
                 patch("llm.run_llm", AsyncMock(return_value=result)):
                self.assertFalse(await compiler.compile_daily_log(source, state))
                save.assert_not_called()
                self.assertNotIn("ingested", state)
                (wiki / "index.md").write_text("[[concepts/lesson]]")
                (wiki / "log.md").write_text("2026-10-19T12:00:00 compile | 2026-10-19.md")
                (wiki / "concepts/lesson.md").write_text(
                    "---\ntitle: Lesson\nsources: [daily/2026-10-19.md]\n---\nLesson")
                self.assertTrue(await compiler.compile_daily_log(source, state))
                self.assertEqual(state["total_cost"], 1.25)
                self.assertEqual(state["codex_usage"]["input_tokens"], 10)
                self.assertEqual(state["ingested"][source.name]["provider"], "codex")

    async def test_compile_failure_preserves_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "day.md"
            source.write_text("lesson")
            with patch.object(compiler, "AGENTS_FILE", source), \
                 patch.object(compiler, "read_wiki_index", return_value=""), \
                 patch.object(compiler, "save_state") as save, \
                 patch("llm.run_llm", AsyncMock(side_effect=RuntimeError("interrupted"))):
                state = {}
                self.assertFalse(await compiler.compile_daily_log(source, state))
                self.assertEqual(state, {})
                save.assert_not_called()

    async def test_query_modes_and_usage(self):
        for file_back in (False, True):
            state = {"total_cost": 2.0}
            with patch.object(query, "read_all_wiki_content", return_value="wiki"), \
                 patch.object(query, "load_state", return_value=state), \
                 patch.object(query, "save_state"), \
                 patch("llm.run_llm", AsyncMock(return_value=llm.LLMResult(
                     "answer", {"output_tokens": 5}))) as run:
                self.assertEqual(await query.run_query("question", file_back), "answer")
                self.assertEqual(run.call_args.kwargs["write"], file_back)
                self.assertEqual(state["query_count"], 1)
                self.assertEqual(state["total_cost"], 2.0)

    async def test_query_failure_does_not_increment_count(self):
        with patch.object(query, "read_all_wiki_content", return_value="wiki"), \
             patch.object(query, "save_state") as save, \
             patch("llm.run_llm", AsyncMock(side_effect=RuntimeError("failed"))):
            with self.assertRaises(RuntimeError):
                await query.run_query("question", True)
            save.assert_not_called()

    async def test_lint_rejects_invalid_output(self):
        for text, count in (("NO_ISSUES", 0), ("CONTRADICTION: a vs b - conflict", 1),
                            ("I could not check the articles", 1)):
            with patch.object(lint, "read_all_wiki_content", return_value="wiki"), \
                 patch("llm.run_llm", AsyncMock(return_value=llm.LLMResult(text, {}))):
                issues = await lint.check_contradictions()
                self.assertEqual(len(issues), count)
                if text.startswith("I could"):
                    self.assertEqual(issues[0]["severity"], "error")

    async def test_flush_uses_text_mode(self):
        with patch("logging.basicConfig"):
            flush = importlib.import_module("flush")
        with patch("llm.run_llm", AsyncMock(return_value=llm.LLMResult("FLUSH_OK", {}))) as run:
            self.assertEqual(await flush.run_flush("conversation"), "FLUSH_OK")
            self.assertFalse(run.call_args.kwargs["tools"])

    def test_failed_flush_keeps_context_for_retry(self):
        with patch("logging.basicConfig"):
            flush = importlib.import_module("flush")
        with tempfile.TemporaryDirectory() as tmp:
            context = Path(tmp) / "context.md"
            context.write_text("conversation")
            with patch.object(sys, "argv", ["flush.py", str(context), "session"]), \
                 patch.object(flush, "load_flush_state", return_value={}), \
                 patch.object(flush, "run_flush", AsyncMock(return_value="FLUSH_ERROR: failed")), \
                 patch.object(flush, "save_flush_state") as save, \
                 patch.object(flush, "append_to_daily_log") as append:
                flush.main()
                self.assertTrue(context.exists())
                save.assert_not_called()
                append.assert_not_called()


if __name__ == "__main__":
    unittest.main()
