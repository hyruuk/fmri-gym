"""A savestate for deterministic envs that keep none: the seed and the actions since.

Many engines hold their state where Python cannot reach it -- a NetHack library
loaded per instance, a Doom process, a Go process, a browser page, a C++ object
with no serializer -- and offer no snapshot of it. If such an engine is
deterministic given ``reset(seed=)`` and the actions, the pair *(seed, actions
so far)* is a savestate in all but speed: restoring it is a reset and a
replay. :class:`ReplayState` records that pair and offers it through
``get_state()`` / ``set_state()``, the names the envs with a real savestate use
(procgen, retro's emulator, VGDL).

That only works when ``reset(seed)`` returns the engine to the same state
every time; an engine whose reset leaves something of the previous episode
behind must be relaunched before ``set_state``.
"""

from __future__ import annotations

import pickle
from typing import Any

import gymnasium as gym
import numpy as np


class ReplayState(gym.Wrapper):
    """Record the episode's ``reset`` seed and ``step`` actions; replay them on ``set_state``.

    The blob grows with the episode (one action per step), so save it sparingly
    in a long real-time episode.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._episode: dict[str, Any] = {"seed": None, "options": None, "actions": []}

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict]:
        """Reset the env and start recording a new episode."""
        self._episode = {"seed": seed, "options": options, "actions": []}
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Step the env and record the action."""
        out = self.env.step(action)
        # Copied: a caller may reuse its action array.
        self._episode["actions"].append(np.array(action, copy=True)
                                        if isinstance(action, np.ndarray) else action)
        return out

    def get_state(self) -> bytes:
        """The episode's seed, reset options and actions so far, as bytes for :meth:`set_state`."""
        return pickle.dumps(self._episode, protocol=5)

    def set_state(self, blob: bytes) -> tuple[Any, dict]:
        """Bring the env to the state ``blob`` was taken in, by resetting and replaying.

        :param blob: bytes from :meth:`get_state`.
        :return: the ``(obs, info)`` of the last replayed step (or of the reset).
        """
        episode = pickle.loads(blob)
        obs, info = self.env.reset(seed=episode["seed"], options=episode["options"])
        for action in episode["actions"]:
            obs, _, _, _, info = self.env.step(action)
        self._episode = {**episode, "actions": list(episode["actions"])}
        return obs, info

