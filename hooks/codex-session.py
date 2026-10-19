"""Codex lifecycle hook: inject context or snapshot a transcript for background capture."""

from __future__ import annotations

import json
import logging
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
QUEUE_DIR = SCRIPTS_DIR / "codex-capture"


def handle_event(event: dict) -> None:
    if os.environ.get("CODEX_MEMORY_INTERNAL"):
        return

    name = event.get("hook_event_name")
    if name == "SessionStart":
        context = runpy.run_path(str(ROOT / "hooks" / "session-start.py"))["build_context"]()
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "SessionStart", "additionalContext": context,
        }}))
        return
    if name not in {"Stop", "PreCompact", "SessionEnd"}:
        return

    transcript = event.get("transcript_path")
    session_id = event.get("session_id")
    if not transcript or not session_id:
        return
    source = Path(transcript)
    if not source.is_file():
        logging.warning("Codex transcript missing: %s", source)
        return

    # Snapshot before returning: the original transcript can change or disappear.
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=QUEUE_DIR, suffix=".jsonl", delete=False) as handle:
        snapshot = Path(handle.name)
    try:
        shutil.copyfile(source, snapshot)
        command = [sys.executable, str(SCRIPTS_DIR / "codex_capture.py"),
                   str(snapshot), session_id]
        with (SCRIPTS_DIR / "codex-capture.log").open("a") as log:
            subprocess.Popen(
                command, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                stdout=log, stderr=log, start_new_session=True,
            )
    except Exception:
        snapshot.unlink(missing_ok=True)
        raise


def main() -> None:
    if os.environ.get("CODEX_MEMORY_INTERNAL"):
        return
    logging.basicConfig(filename=str(SCRIPTS_DIR / "codex-capture.log"),
                        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        handle_event(json.load(sys.stdin))
    except Exception:
        # Capture must not block or fail the user's Codex turn.
        logging.exception("Codex memory hook failed")


if __name__ == "__main__":
    main()
