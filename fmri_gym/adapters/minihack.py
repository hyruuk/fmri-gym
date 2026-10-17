"""MiniHack adapter (facebookresearch/minihack, NetHack Learning Environment).

MiniHack's default observation is ASCII/tty and env.render() returns None, so we
request a pixel observation and display that. By default we show `pixel_crop` --
a 144x144 square window centered on the agent -- because the full `pixel`
observation is the entire 80-column NetHack terminal (336x1264, ~3.8:1) in which
a small room fills only a few percent of the frame, so aspect-fitting it makes
the game look tiny. Set the phase field "full_screen": true to display the whole
`pixel` frame instead. Actions are 8 compass directions (N,E,S,W,NE,SE,SW,NW ->
Discrete(8), indices 0..7), which a phase's ``keys`` index; some tasks append
more (Eat: 29 = eat). Play ``turn_based``: there is no no-op (0 is "move N").

NLE has no savestate API and its game lives in a NetHack library loaded per
env, out of Python's reach, so the savestate is :class:`~.replaystate.ReplayState`'s:
the seed and the actions since, restored by replay. That is exact only because
:class:`NLESeeding` seeds NetHack's own RNGs on every reset and refuses one
where that did not take. The full obs dict (glyphs, blstats, message, ...) is
the analysis state; we log the compact non-pixel fields and keep the pixel
frame for display only.

Requires `pkg_resources` (install setuptools<81) as minihack imports it.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from .base import EnvAdapter, FrameState
from .replaystate import ReplayState


class NLESeeding(gym.Wrapper):
    """Seed NetHack's own RNGs on every reset, and refuse a reset where that did not take.

    Gymnasium's ``reset(seed=)`` seeds only the env's ``np_random``; NetHack's
    two RNGs (core and disp) are set by NLE's ``seed(core=, disp=, reseed=)``,
    which applies to the next reset and that one only. Without it every reset
    draws true randomness and the game reseeds itself mid-play (NetHack's
    anti-automation measure), and no replay can match it. The other half is
    the env's ``fix_moon_phase`` kwarg: NetHack reads the real date for luck
    (full moon, Friday the 13th); with it the date derives from the seed.
    ``get_seeds()`` reports what the reset actually used, so a reset that was
    not seeded is caught here rather than found in a replay that diverges.
    """

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict]:
        if seed is None:
            raise ValueError("NetHack needs reset(seed=...): an unseeded game reseeds itself "
                             "from true randomness and cannot be replayed")
        self.env.unwrapped.seed(core=seed, disp=seed, reseed=False)
        out = self.env.reset(seed=seed, options=options)
        core, disp, reseed = self.env.unwrapped.get_seeds()[:3]
        if (core, disp, reseed) != (seed, seed, False):
            raise RuntimeError(f"NetHack reset with seeds {(core, disp, reseed)}, not "
                               f"{(seed, seed, False)}: the episode would not replay")
        return out


class MiniHackAdapter(EnvAdapter):
    name: str = "minihack"

    def _make(self, spec: dict) -> gym.Env:
        import minihack  # noqa: F401  (registers MiniHack-* env ids)
        # Prefer the agent-centered square crop for display; the full terminal
        # ("pixel") only looks good with "full_screen": true.
        self._pixel_key = "pixel" if spec.get("full_screen") else "pixel_crop"
        keys = tuple(spec.get("observation_keys",
                              (self._pixel_key, "glyphs", "blstats", "message")))
        if self._pixel_key not in keys:
            keys = (self._pixel_key,) + keys
        env = gym.make(spec["game"], observation_keys=keys, fix_moon_phase=True)
        self._last = None
        return ReplayState(NLESeeding(env))

    def reset(self, seed: int | None) -> tuple[Any, dict]:
        obs, info = super().reset(seed)
        self._last = obs
        return obs, info

    def render(self) -> np.ndarray:
        # Display the pixel observation (env.render() is None for MiniHack).
        return np.asarray(self._last[self._pixel_key])

    def outcome(self, terminated: bool, truncated: bool) -> tuple[str, str]:
        # NLE's end_status: TASK_SUCCESSFUL, DEATH, or ABORTED at the step limit.
        status = getattr(self.last_info.get("end_status"), "name", "")
        if status == "TASK_SUCCESSFUL":
            return "won", "Task complete"
        if status == "DEATH":
            return "lost", "You died"
        if status == "ABORTED":
            return "truncated", "Timeout"
        return super().outcome(terminated, truncated)

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        self._last = obs
        variables = {}
        if isinstance(obs, dict):
            for k, v in obs.items():
                if k in ("pixel", "pixel_crop"):
                    continue
                variables[k] = np.asarray(v).copy()
        variables.update(info or {})
        return FrameState(blob=self.env.get_state() if want_blob else None, variables=variables)

    def restore(self, blob: bytes) -> None:
        self._last, _ = self.env.set_state(blob)
