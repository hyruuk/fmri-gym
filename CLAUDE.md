@AGENTS.md

This file only imports `AGENTS.md`, the rules for every agent and human, so that Claude Code always reads them: it reads `AGENTS.md` by itself only when no `CLAUDE.md` or `CLAUDE.local.md` exists, and a personal one would silently drop the rules. Your own instructions go in `CLAUDE.local.md` (gitignored) or `~/.claude/CLAUDE.md`, which load alongside this file, never instead of it. Rules for everyone go in `AGENTS.md`, not here.
