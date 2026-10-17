"""Two wrappers over ViZDoom's Gymnasium envs, for play without a mouse and for replay.

ViZDoom presents a scenario with both keyboard buttons and *delta* buttons (the
mouse axes; Deathmatch, the full-game maps) as a ``Dict{"binary",
"continuous"}`` action space. :class:`KeyboardOnlyAction` holds the axes at
rest and leaves the binary part, for a player (or a model) with keys only.
The layout is ViZDoom's: no other env here has it.

ViZDoom's env starts its first episode with ``game.init()`` and every later
one with ``game.new_episode()``, and an episode begun by ``init()`` has its
sound mixed a little differently (pixels and game variables are the same). A
replay that resets a second time could then never match a first episode's
audio. :class:`StartedEpisodes` resets once when it is built, so every episode
the caller sees begins with ``new_episode()``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class KeyboardOnlyAction(gym.ActionWrapper):
    """Drop a scenario's delta (mouse) axes, keeping its keyboard buttons."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = env.action_space["binary"]
        self._axes = np.zeros(env.action_space["continuous"].shape, dtype=np.float32)

    def action(self, action: Any) -> dict[str, Any]:
        """Return ``action`` as a Dict action with the delta axes at rest.

        :param action: a binary action (Discrete index or button vector).
        :return: the ``{"binary", "continuous"}`` action ViZDoom expects.
        """
        return {"binary": action, "continuous": self._axes}


class StartedEpisodes(gym.Wrapper):
    """Reset once on construction: every episode the caller resets begins with new_episode()."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        env.reset(seed=0)
