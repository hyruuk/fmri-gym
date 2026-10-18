---
name: pull-request
description: Take an fmri-gym change from a branch to a merged pull request on tomov/fmri-gym: the worktree, the local test run, the lint, what the PR description must say (core changes, backends not played locally), draft versus ready for review, and watching CI. Use when asked to open, update or finish a PR, or to push a branch.
---

# From a branch to a merged PR

## Branch

Work on a branch off `main`, in a git worktree when the main checkout is in use (`git worktree add ../fmri-gym-worktrees/<branch> -b <branch> main`). A fresh worktree's `.venv` is empty until `uv sync --extra <the backends you touch>`; `uv run --no-sync` afterwards keeps uv from re-syncing on every command. Submodules come with `git submodule update --init`.

## Before pushing

```bash
uv run pytest --changed
uv run ruff check fmri_gym fmri_play.py agents tests gym/*/tests
```

Both clean, or the PR says which backend you could not play here and why (the summary's skip line is the sentence to quote). The `testing` skill explains the options and how to read a failure. Commit messages carry the reasoning behind a change; the code, the docstrings and the configs describe the present only (AGENTS.md, rule 3).

## The description

- What changed and why, in a few sentences.
- **Core changes, listed.** An edit to `run.py`, `display.py`, `keys.py`, `logging.py`, `adapters/base.py` or `adapters/keymap.py` is the part a reviewer must read closely (AGENTS.md, rule 2): name the file, say it is additive and default-off and names no backend.
- **What was played.** Which backends the local suite ran, which it skipped and why, and what you ran by hand (`uv run fmri-play ... --dummy-trigger`).
- A new backend: its config, how it installs, and that `tests/backends.py` has its row.

## Draft, ready, CI

Open as a draft while the work is in flux; CI does not run on drafts. Marking it ready for review runs the suite (`.github/workflows/tests.yml`: a core job, then one job per backend a Linux runner can play, every config, no skipping), and every push after that runs it again. Watch with `gh pr checks <number> --watch`; a failed backend job keeps its run folders as an artifact. Make the checks green before asking for review, or say in the PR which job fails and why that is expected.
