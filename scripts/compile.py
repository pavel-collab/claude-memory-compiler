"""
Compile daily conversation logs into structured knowledge articles.

This is the "LLM compiler" - it reads daily logs (source code) and produces
organized knowledge articles (the executable).

Usage:
    uv run python compile.py                    # compile new/changed logs only
    uv run python compile.py --all              # force recompile everything
    uv run python compile.py --file daily/2026-04-01.md  # compile a specific log
    uv run python compile.py --dry-run          # show what would be compiled
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from config import AGENTS_FILE, CONCEPTS_DIR, CONNECTIONS_DIR, DAILY_DIR, KNOWLEDGE_DIR, now_iso
from utils import (
    file_hash,
    list_raw_files,
    list_wiki_articles,
    load_state,
    read_wiki_index,
    save_state,
)

# ── Paths for the LLM to use ──────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent


def is_article_path(path: Path) -> bool:
    """Distinguish wiki article paths from metadata and attachments."""
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and path.suffix == ".md"
        and path.as_posix() not in {"index.md", "log.md", "AGENTS.md", "CLAUDE.md"}
        and not any(part.startswith(".") or part == "assets" for part in path.parts)
    )


async def compile_daily_log(log_path: Path, state: dict) -> bool:
    """Compile a single daily log into knowledge articles.

    Returns whether compilation completed and its output was verified.
    """
    from llm import record_usage, run_llm

    source_hash = file_hash(log_path)
    log_content = log_path.read_text(encoding="utf-8")
    schema = AGENTS_FILE.read_text(encoding="utf-8")
    wiki_index = read_wiki_index()
    existing_articles = {
        path.relative_to(KNOWLEDGE_DIR).as_posix()
        for path in KNOWLEDGE_DIR.rglob("*.md")
        if path.is_file() and not path.is_symlink()
        and is_article_path(path.relative_to(KNOWLEDGE_DIR))
    }

    timestamp = now_iso()

    prompt = f"""You are a knowledge compiler. Your job is to read a daily conversation log
and extract knowledge into structured wiki articles.

## Schema (AGENTS.md)

{schema}

## Current Wiki Index

{wiki_index}

The index above lists every existing article. Use file reading and search tools to open any article
you need to check or update — do not assume its content, read it first.

## Daily Log to Compile

**File:** {log_path.name}

{log_content}

## Your Task

The actual wiki root is {KNOWLEDGE_DIR}. All knowledge/ paths in the schema
refer to this directory. Do not create a knowledge/ subdirectory.
Daily logs live in {DAILY_DIR} and must not be modified.
Return a JSON object with an articles array of wiki-relative paths of all
created or updated articles (for example concepts/example.md).

Read the daily log above and compile it into wiki articles following the schema exactly.

### Rules:

1. **Extract key concepts** - Identify 3-7 distinct concepts worth their own article
2. **Create concept articles** in `knowledge/concepts/` - One .md file per concept
   - Use the exact article format from AGENTS.md (YAML frontmatter + sections)
   - Include `sources:` in frontmatter pointing to the daily log file
   - Use `[[concepts/slug]]` wikilinks to link to related concepts
   - Write in encyclopedia style - neutral, comprehensive
3. **Create connection articles** in `knowledge/connections/` if this log reveals non-obvious
   relationships between 2+ existing concepts
4. **Update existing articles** if this log adds new information to concepts already in the wiki
   - Read the existing article, add the new information, add the source to frontmatter
   - Existing articles may live in any wiki section, such as projects/, entities/,
     or self-education/. Update them in place and include their paths in the manifest.
   - Create new articles only in concepts/ or connections/.
5. **Update knowledge/index.md** - Add new entries to the table
   - Each entry: `| [[path/slug]] | One-line summary | source-file | {timestamp[:10]} |`
6. **Do not edit knowledge/log.md**, even though the general schema describes
   appending to it. The Python compiler appends the build entry itself after
   validating your articles and index. Return every created or updated article
   in the JSON manifest.

### File paths:
- Write concept articles to: {CONCEPTS_DIR}
- Write connection articles to: {CONNECTIONS_DIR}
- Update index at: {KNOWLEDGE_DIR / 'index.md'}
- Build log (managed by Python; do not edit): {KNOWLEDGE_DIR / 'log.md'}

### Quality standards:
- Every article must have complete YAML frontmatter
- Every article must link to at least 2 other articles via [[wikilinks]]
- Key Points section should have 3-5 bullet points
- Details section should have 2+ paragraphs
- Related Concepts section should have 2+ entries
- Sources section should cite the daily log with specific claims extracted
"""

    try:
        result = await run_llm(
            prompt,
            write=True,
            output_schema={
                "type": "object",
                "properties": {
                    "articles": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string"},
                    },
                },
                "required": ["articles"],
                "additionalProperties": False,
            },
        )
        articles = json.loads(result.text)["articles"]
        if not isinstance(articles, list) or not articles:
            raise ValueError("Compilation returned no articles")
        index = (KNOWLEDGE_DIR / "index.md").read_text(encoding="utf-8")
        verified_articles = []
        for article in articles:
            if not isinstance(article, str) or not is_article_path(Path(article)):
                raise ValueError(f"Invalid article path: {article}")
            path = (KNOWLEDGE_DIR / article).resolve()
            relative = path.relative_to(KNOWLEDGE_DIR.resolve())
            if not is_article_path(relative) or (
                relative.as_posix() not in existing_articles
                and (len(relative.parts) < 2 or relative.parts[0] not in {"concepts", "connections"})
            ):
                raise ValueError(f"Invalid article path: {article}")
            content = path.read_text(encoding="utf-8")
            if not content.startswith("---\n") or log_path.name not in content:
                raise ValueError(f"Article missing frontmatter or source: {article}")
            if f"[[{relative.with_suffix('').as_posix()}]]" not in index:
                raise ValueError(f"Article missing from index: {article}")
            verified_articles.append(relative.as_posix())
        if file_hash(log_path) != source_hash:
            raise ValueError("Source changed during compilation; retry on the next run")

        created, updated = [], []
        for article in dict.fromkeys(verified_articles):
            link = f"[[{Path(article).with_suffix('').as_posix()}]]"
            (updated if article in existing_articles else created).append(link)
        build_log_path = KNOWLEDGE_DIR / "log.md"
        needs_header = not build_log_path.exists() or build_log_path.stat().st_size == 0
        with build_log_path.open("a", encoding="utf-8") as build_log:
            if needs_header:
                build_log.write("# Build Log\n")
            build_log.write(
                f"\n\n## [{timestamp}] compile | {log_path.name}\n"
                f"- Source: daily/{log_path.name}\n"
                f"- Articles created: {', '.join(created) or '(none)'}\n"
                f"- Articles updated: {', '.join(updated) or '(none)'}\n"
            )
    except Exception as e:
        print(f"  Error: {e}")
        return False

    state.setdefault("ingested", {})[log_path.name] = {
        "hash": source_hash,
        "compiled_at": now_iso(),
        "provider": "codex",
        "usage": result.usage,
    }
    record_usage(state, result.usage)
    save_state(state)
    return True


def main():
    parser = argparse.ArgumentParser(description="Compile daily logs into knowledge articles")
    parser.add_argument("--all", action="store_true", help="Force recompile all logs")
    parser.add_argument("--file", type=str, help="Compile a specific daily log file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be compiled")
    args = parser.parse_args()

    state = load_state()

    # Determine which files to compile
    if args.file:
        target = Path(args.file)
        if not target.is_absolute():
            target = DAILY_DIR / target.name
        if not target.exists():
            # Try resolving relative to project root
            target = ROOT_DIR / args.file
        if not target.exists():
            print(f"Error: {args.file} not found")
            sys.exit(1)
        to_compile = [target]
    else:
        all_logs = list_raw_files(since=state.get("compile_from"))
        if args.all:
            to_compile = all_logs
        else:
            to_compile = []
            for log_path in all_logs:
                rel = log_path.name
                prev = state.get("ingested", {}).get(rel, {})
                if not prev or prev.get("hash") != file_hash(log_path):
                    to_compile.append(log_path)

    if not to_compile:
        print("Nothing to compile - no new or changed daily logs in the selected date range.")
        return

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Files to compile ({len(to_compile)}):")
    for f in to_compile:
        print(f"  - {f.name}")

    if args.dry_run:
        return

    # Compile each file sequentially
    failures = 0
    for i, log_path in enumerate(to_compile, 1):
        print(f"\n[{i}/{len(to_compile)}] Compiling {log_path.name}...")
        if asyncio.run(compile_daily_log(log_path, state)):
            print("  Done.")
        else:
            failures += 1

    articles = list_wiki_articles()
    print(f"\nCompilation complete. Failed: {failures}")
    print(f"Knowledge base: {len(articles)} articles")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
