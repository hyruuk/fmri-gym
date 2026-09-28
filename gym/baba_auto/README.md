# baba-auto-gym

Baba Is Auto: [Baba Is You](https://hempuli.com/baba/) as simulated by [utilForever/baba-is-auto](https://github.com/utilForever/baba-is-auto), a C++17 engine with a pybind11 module (`pyBaba`), as a Gymnasium env. `pyBaba` exposes the `Game` alone -- the RL envs upstream ships speak `gym` 0.24, hard-code a map path and render into a window -- so this holds one `pyBaba.Game`, presents the Gymnasium contract in front of it and draws the picture with the GUI's own sprites. The rules stay in the simulator; nothing of upstream is copied here.

The engine comes from fmri-gym's `external/baba_auto` checkout, at the commit its README pins. `setup.py` builds `pyBaba` from it (cmake, ninja and pybind11 come from `build-system.requires`; you need a C++17 compiler and the Python headers, `python3-dev` on Ubuntu), and the env reads its maps and sprites from it at run time:

```bash
git clone https://github.com/utilForever/baba-is-auto.git ../../external/baba_auto   # from this directory
git -C ../../external/baba_auto checkout <the README's commit>
pip install -e .            # ~30 s of compiling
```

```python
import baba_auto_gym
env = baba_auto_gym.BabaAutoEnv("baba_is_you")   # or gym.make("BabaAuto-v0", game="baba_is_you")
obs, info = env.reset(seed=0)                     # obs is (16, H, W) float32; info["play_state"]
obs, reward, terminated, truncated, info = env.step(1)   # up
frame = env.render()                              # (24 * H, 24 * W, 3) uint8
```

`game` is a map name under the checkout's `Resources/Maps` (`baba_is_you`, `out_of_reach`, `volcano`, `off_limits`, `grass_yard`, `pillar_yard`, `brick_wall`, `icy_waters`, `novice_locksmith`, `lock`, `affection`, `turns`, ... -- many of the others are rule-engine fixtures rather than puzzles) or the path of a map file of your own; `repo=` reads the maps and sprites from another checkout. Actions are `Discrete(5)`: 0 = wait, 1 = up, 2 = down, 3 = left, 4 = right. The reward is the one upstream's RL examples use: +200 on WON, -100 on LOST, -0.5 per other turn; the episode ends on either. No savestate: an episode replays from its seed and actions.
