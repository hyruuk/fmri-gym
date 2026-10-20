# configs/rigs/

One file per rig, in a folder per site: `rigs/<site>/<rig>.json`, e.g. `rigs/boston-college/prisma-3t-gamepad.json`. A rig is one setup a site runs sessions on -- a monitor or projector, a scanner, a response device, a trigger line, an audio path -- and a site can have several. Its file says everything about how a run is played there that a curriculum must not: the screen (size, fullscreen, monitor, vsync), the participant's device and its keys, audio, the data root, the trigger hardware. The `site` and `rig` fields inside the file are its folder and file name, and `fmri-play --rig <site>/<rig>` picks it (`fmri_gym/config/rig.py`; README "Rigs" lists its fields). The rig check's form writes one when its rig has none: commit it.

Two example rigs play out of the box, both a 1024x768 window with audio on and no scanner (a run starts on SPACE): `example/laptop.json`, a laptop and its keyboard alone (no controller read, so no bindings), and `example/desk.json`, a desk PC with the Current Designs controller, its controls bound to the keys the curricula expect (README "Rigs"). Pick one with `--rig example/laptop`; with more than one rig file every run names its rig, so a session never lands on an example by default. Copy one as the start of a real rig.

A rig is site-specific and shared by every run and every game on it; a curriculum (`../curricula/`) is game-specific and shared by every site; a session (`../subjects/`) is one subject's one visit. Keep each setting in the one place it belongs.

`rig-check.sh` plays the rig check (`../curricula/checks/rig-check-long.json`) on this machine, filed under `sub-rig`.

The rig's `"triggers"` say how a run starts and what the recording gets (README, "Triggers"). A rig whose MEG or EEG recording starts from the trigger line, codes over serial:

```json
"triggers": {
  "sync": {"mode": "send", "delay": 0.0},
  "backend": "serial", "port": "/dev/ttyUSB0",
  "codes": {"frame_bits": 3, "task_start": 8, "task_stop": 16, "episode_start": 32, "scanner_start": 64}
}
```

An fMRI rig whose trigger box types `=` at every volume needs only `"triggers": {"sync": {"mode": "wait", "key": "="}, "backend": "null"}`.
