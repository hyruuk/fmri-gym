# configs/rigs/

One file per rig, in a folder per site: `rigs/<site>/<rig>.json`, e.g. `rigs/boston-college/prisma-3t-gamepad.json`. A rig is one setup a site runs sessions on -- a monitor or projector, a scanner, a response device, a trigger line, an audio path -- and a site can have several. Its file says everything about how a run is played there that a curriculum must not: the screen (size, fullscreen, monitor, vsync), the participant's device and its keys, audio, the data root, the trigger hardware. The `site` and `rig` fields inside the file are its folder and file name, and `fmri-play --rig <site>/<rig>` picks it (`fmri_gym/config/rig.py`; README "Rigs" lists its fields). The rig check's form writes one when its rig has none: commit it.

A rig is site-specific and shared by every run and every game on it; a curriculum (`../curricula/`) is game-specific and shared by every site; a session (`../subjects/`) is one subject's one visit. Keep each setting in the one place it belongs.

`rig-check.sh` plays the rig check (`../curricula/checks/rig-check-long.json`) on this machine, filed under `sub-rig`.
