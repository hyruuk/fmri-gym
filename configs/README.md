# configs/

Three kinds of file, kept apart because each is shared differently: a curriculum is the same for every subject at every site, a rig is the same for every run on one machine, and a session is one subject's one visit.

```text
configs/
  curricula/<backend>/levels/<name>.json            what one run plays: one level
  curricula/<backend>/<name>.json                   what one run plays: several levels in a row
  curricula/<backend>/common/<name>.json            a game phase several of them include
  curricula/<backend>/common/README.md              what they play, and how
  rigs/<site>/<rig>.json                            how a site's machine plays it (rigs/README.md)
  subjects/<site>/sub-<label>/ses-<NNN>/<name>.sh   which runs one subject plays in one session
```

## curricula/

One folder per backend (the `backend` its game phases name), plus `checks/` (the rig check, whose phases are checks rather than games) and `demos/` (runs that span backends). A file is either one level, in the backend's `levels/` folder -- `minihack/levels/room5x5.json`, `vizdoom/levels/freedoom1_e1m1.json` -- or a sequence of levels, in the backend's folder itself -- `rushhour/complete.json`, `vgdl/all.json`. `aigamestore/` has no `levels/`: its ten games keep one layout, one file each in the folder, the one-level games too. The name says the level, not the backend: the folder already does. What a backend's curricula share -- a game's keys, instructions and settings -- is written once, in `<backend>/common/<name>.json`, and each phase includes it (`"__include": "common/<name>.json"`, or `"../common/<name>.json"` from `levels/`) and states only what differs (README, "Shared phases"); `common/` holds no curricula. `<backend>/common/README.md` is the backend's documentation: what each curriculum plays, setup, what the keys' indices mean, and what a level needs said. Comments in a curriculum (`"## ..."` keys) are few, for what is particular to that one file.

A curriculum's path after `curricula/`, in camelCase (BIDS labels are letters and digits only), is its BIDS task label: `curricula/minihack/levels/room5x5.json` is `task-minihackLevelsRoom5x5`, `curricula/vgdl/all.json` is `task-vgdlAll`. Moving or renaming a file renames the task, and the task keys the run's seeds, so a level that has been recorded keeps its file name.

Every file here is played by the test suite (`tests/test_configs.py` validates each one; `pytest --thorough` plays every game phase of each), so a curriculum that does not play does not stay. The games tried and left out are listed in the top-level README, "Games without a curriculum".

## rigs/

See `rigs/README.md`.

## subjects/

One shell script per session, `subjects/<site>/sub-<label>/ses-<NNN>/<name>.sh`, as BIDS lays out the data it records: one `fmri-play` line per run, in order (README, "Curriculum format"). A script plays into the session its folder names (`SES=${1:-001}`); its own argument overrides that, which is how a session that stopped is resumed into a new one. `fmri-edit --session <script>` opens one; run it from the repo root with `sh <script>`.
