"""MiniHack adapter (facebookresearch/minihack, NetHack Learning Environment).

MiniHack's default observation is ASCII/tty and env.render() returns None, so we
request a pixel observation and display that. The phase field ``view`` picks
which one:

- ``"map"`` (the default): the level's map, cut from the whole-terminal
  ``pixel`` frame by ``minihack_levels.MapView`` and fixed across the episode,
  so the level holds still and the agent moves. A level without a fixed map
  (``RANDOM_CORRIDORS``) gets the whole terminal.
- ``"agent"``: MiniHack's own ``pixel_crop``, a 9x9-tile window centred on the
  agent, so the agent holds still and the level slides. What a task that
  wants the view to follow the player asks for.
- ``"screen"``: the whole ``pixel`` frame, the 80-column NetHack terminal
  (336x1264, ~3.8:1), in which a small room is a few percent of the picture.

Actions are 8 compass directions (N,E,S,W,NE,SE,SW,NW ->
Discrete(8), indices 0..7), which a phase's ``keys`` index; some tasks append
more (Eat: 29 = eat). Play ``turn_based``: there is no no-op (0 is "move N").

NLE has no savestate API and its game lives in a NetHack library loaded per
env, out of Python's reach, so the savestate is ``replay_state_gym.ReplayState``'s:
the seed and the actions since, restored by replay. That is exact only because
``minihack_levels.NLESeeding`` seeds NetHack's own RNGs on every reset and refuses one
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

#: the phase's ``view``: how the pixel frame is cut (see the module docstring)
_VIEWS = ("map", "agent", "screen")


class MiniHackAdapter(EnvAdapter):
    name: str = "minihack"

    def _make(self, spec: dict) -> gym.Env:
        import minihack  # noqa: F401  (registers MiniHack-* env ids)
        from minihack_levels import MapView, NLESeeding
        from replay_state_gym import ReplayState
        view = spec.get("view", "map")
        if view not in _VIEWS:
            raise ValueError(f"minihack phase 'view' is one of {_VIEWS}, not {view!r}")
        self._pixel_key = "pixel" if view == "screen" else "pixel_crop"
        keys = tuple(spec.get("observation_keys",
                              (self._pixel_key, "glyphs", "blstats", "message")))
        if self._pixel_key not in keys:
            keys = (self._pixel_key,) + keys
        if view == "map" and "pixel" not in keys:
            keys = ("pixel",) + keys  # MapView cuts pixel_crop from it
        env = gym.make(spec["game"], observation_keys=keys, fix_moon_phase=True)
        if view == "map":
            env = MapView(env)
        self._last = None
        return ReplayState(NLESeeding(env))

    def reset(self, seed: int) -> tuple[Any, dict]:
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
        return FrameState(variables=variables)

    def restore(self, blob: bytes) -> None:
        self._last, _ = self.env.set_state(blob)

    def block_extra(self) -> dict:
        return {"state": self.env.get_state()}
