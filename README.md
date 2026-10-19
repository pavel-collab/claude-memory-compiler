# LLM Personal Knowledge Base

**Your AI conversations compile themselves into a searchable knowledge base.**

Adapted from [Karpathy's LLM Knowledge Base](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) architecture, but instead of clipping web articles, the raw data is your own conversations with Claude Code. When a session ends (or auto-compacts mid-session), Claude Code hooks capture the conversation transcript and spawn a background process that uses the [Codex SDK](https://learn.chatgpt.com/docs/codex-sdk) to extract the important stuff - decisions, lessons learned, patterns, gotchas - and appends it to a daily log. You then compile those daily logs into structured, cross-referenced knowledge articles organized by concept. Retrieval uses a simple index file instead of RAG - no vector database, no embeddings, just markdown.

## Quick Start

1. Run `uv sync --frozen` to install the Python dependencies and pinned Codex runtime.
2. Authenticate Codex with `codex login` using your account or supported API credentials.
3. Check the vault paths in `scripts/config.py`. This installation stores daily logs
   in `ProjectsRootVault/daily/` and compiled articles in `ProjectsRootVault/wiki/`.
4. Keep the existing Claude Code hooks for session capture. The project-local
   `.claude/settings.json` is empty; see AGENTS.md for hook configuration examples.

The compiler uses Codex's configured default model. Set `CODEX_MODEL` to override it.
The SDK reuses local Codex authentication; Claude authentication is not used for
compilation, extraction, queries, or semantic lint. Usage is subject to your Codex
account or API limits.

Auto-compilation is currently disabled (`COMPILE_AFTER_HOUR = 25`). Run the compiler
manually, or change that constant to `18` to enable the evening trigger.

## How It Works

```
Conversation -> SessionEnd/PreCompact hooks -> flush.py extracts knowledge
    -> daily/YYYY-MM-DD.md -> compile.py -> wiki/concepts/, connections/, qa/
        -> SessionStart hook injects index into next session -> cycle repeats
```

- **Hooks** capture conversations automatically (session end + pre-compaction safety net)
- **flush.py** calls the Codex SDK to decide what's worth saving, and optionally triggers end-of-day compilation
- **compile.py** turns daily logs into organized concept articles with cross-references (triggered automatically or run manually)
- **query.py** answers questions using index-guided retrieval (no RAG needed at personal scale)
- **lint.py** runs 7 health checks (broken links, orphans, contradictions, staleness)

## Key Commands

```bash
uv run python scripts/compile.py                    # compile new daily logs
uv run python scripts/query.py "question"            # ask the knowledge base
uv run python scripts/query.py "question" --file-back # ask + save answer back
uv run python scripts/lint.py                        # run health checks
uv run python scripts/lint.py --structural-only      # free structural checks only
```

## Why No RAG?

Karpathy's insight: at personal scale (50-500 articles), the LLM reading a structured `index.md` outperforms vector similarity. The LLM understands what you're really asking; cosine similarity just finds similar words. RAG becomes necessary at ~2,000+ articles when the index exceeds the context window.

## Technical Reference

See **[AGENTS.md](AGENTS.md)** for the complete technical reference: article formats, hook architecture, script internals, cross-platform details, costs, and customization options. AGENTS.md is designed to give an AI agent everything it needs to understand, modify, or rebuild the system.

## Compilation Start Date

Set `compile_from` in `scripts/state.json` to an ISO date, for example
`"2026-10-19"`. Automatic selection and `--all` include only logs on or after
that date. Older logs are also excluded from orphan-source and stale-source
checks. An explicit `--file` can still compile an older log.

The cutoff is fixed, not a rolling “today” filter: later runs include all pending
logs since that date. Removing the field restores selection across all dates.

## Codex Integration

`scripts/llm.py` is the shared SDK adapter. It uses independent ephemeral threads,
rejects failed/interrupted turns and empty responses, and records token usage.
Writing tasks run with the wiki as their workspace; other tasks use a read-only
sandbox. Escalated permissions are denied. Shell tools are disabled for flush and
semantic lint; prompts also instruct these text-only tasks not to use tools.
This is not a general-purpose tool allowlist equivalent to Claude's `allowed_tools`.

Compilation validates the returned article paths, source references, index, and
build log before updating the source hash. Failed runs exit nonzero and remain
eligible for retry. File edits are not transactional: a failed run can leave partial
wiki changes. Failed flushes retain their temporary context file for manual retry.

Historical Claude `total_cost` values remain unchanged. New compilation records
store `provider: codex` and `usage`; aggregate tokens are stored in `codex_usage`.
No dollar cost is inferred from tokens or subscription usage.

Claude Code hooks remain unchanged. Codex capture is available through
`hooks/codex-session.py` and `scripts/codex_capture.py`.

## Tests

```bash
uv run --frozen python -m unittest discover -s tests -v
```

Tests use temporary files and mocked SDK turns; they do not call a model or change
the personal vault.

## Codex Session Hooks

`hooks/codex-hooks.json` contains the user-level hook configuration with absolute
paths for this installation. Install it as `~/.codex/hooks.json` (merge events if
that file already exists), then review and trust the four handlers using `/hooks`
in Codex CLI. Restart or resume sessions after configuring hooks. Codex skips
untrusted hooks; this project does not bypass hook trust.

- `SessionStart` injects the wiki index and recent daily notes.
- `Stop`, `PreCompact`, and `SessionEnd` snapshot the transcript and launch a
  detached extraction worker. `Stop` saves completed turns without waiting for
  the entire session to end.
- SDK tasks set `CODEX_MEMORY_INTERNAL=1` and disable hooks to prevent recursion.
- The worker uses per-session byte cursors and locks; repeated events skip
  previously processed content. Only user/assistant text is extracted.
- New useful summaries append to `ProjectsRootVault/daily/YYYY-MM-DD.md`.
  An excerpt with no useful information creates no daily entry.
- Capture uses Codex model calls in the background. Wiki compilation remains
  manual; the start-date cutoff applies to compilation, not capture.

Diagnostics: `scripts/codex-capture.log`. Failed transcript snapshots remain under
`scripts/codex-capture/` and can be retried with:

```bash
.venv/bin/python scripts/codex_capture.py <snapshot.jsonl> <original-session-id>
```

The worker currently targets macOS/Linux (`fcntl` file locks). Codex's transcript
format is not a stable hook interface; a changed transcript prefix or malformed
record is retained for inspection rather than silently marked processed.
