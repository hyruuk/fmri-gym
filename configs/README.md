# configs/

Three kinds of file, kept apart because each is shared differently: a curriculum is the same for every subject at every site, a rig is the same for every run on one machine, and a session is one subject's one visit.

```text
configs/
  curricula/<backend>/<name>.json                   what one run plays
  rigs/<site>/<rig>.json                            how a site's machine plays it (rigs/README.md)
  subjects/<site>/sub-<label>/ses-<NNN>/<name>.sh   which runs one subject plays in one session
```

## curricula/

One folder per backend (the `backend` its game phases name), plus `checks/` (the rig check, whose phases are checks rather than games) and `demos/` (runs that span backends). A file is either one level -- `minihack/room5x5.json`, `vizdoom/freedoom1_e1m1.json` -- or a game's sequence of levels -- `rushhour/complete.json`, `vgdl/all.json`. The name says the level, not the backend: the folder already does.

A curriculum's path is its BIDS task label, the folder's name then the file's, letters and digits only: `curricula/minihack/room5x5.json` is `task-minihackroom5x5`. Renaming a file renames the task, and the task keys the run's seeds, so a level that has been recorded keeps its file name.

Every file here is played by the test suite (`tests/test_configs.py` validates each one; `pytest --thorough` plays every game phase of each), so a curriculum that does not play does not stay. The games tried and left out are listed in the top-level README, "Games without a curriculum".

## rigs/

See `rigs/README.md`.

## subjects/

One shell script per session, `subjects/<site>/sub-<label>/ses-<NNN>/<name>.sh`, as BIDS lays out the data it records: one `fmri-play` line per run, in order (README, "Curriculum format"). A script plays into the session its folder names (`SES=${1:-001}`); its own argument overrides that, which is how a session that stopped is resumed into a new one. `fmri-edit --session <script>` opens one; run it from the repo root with `sh <script>`.
