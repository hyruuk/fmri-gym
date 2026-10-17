"""A savestate for a deterministic env that keeps none: its seed and the actions since.

Most engines here hold their state where Python cannot reach it -- a NetHack
library loaded per instance, a Doom process, a Go process, a browser page, a
C++ object with no serializer -- and offer no snapshot of it. Every one of them
is deterministic given ``reset(seed=)`` and the actions, which makes the pair
*(seed, actions so far)* a savestate in all but speed: restoring it is a reset
and a replay. This wrapper records that pair and presents it through
``get_state()`` / ``set_state()``, the names the envs with a real savestate
use (VGDL here, procgen, retro's emulator), so an adapter over it reads the
same as one over those.

The blob is a pickle of the episodes recorded: each its seed, its reset
options and its actions. By default that is the current episode alone, because
``reset(seed)`` is a fixed point of the engine and replaying this episode is
enough. An engine whose reset is *not* a fixed point -- a restart leaves it a
little different from a fresh launch -- is given ``relaunch``, a callable that
builds a fresh inner env; then every episode since launch is kept and
``set_state`` closes the env, builds a new one and replays them all.

``reset(seed=None)`` is refused: there is nothing to replay from, and a
recording that could not be replayed is the failure this exists to prevent.
The blob grows with the episode, so a real-time block over this wrapper wants
a ``state_stride`` of about a second's worth of frames.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np


class ReplayState(gym.Wrapper):
    """Record ``reset`` seeds and ``step`` actions; replay them on ``set_state``.

    :param env: the deterministic env to wrap.
    :param relaunch: builds a fresh copy of ``env``, for an engine whose reset
        does not return it to its launch state; ``None`` when reset does.
    """

    def __init__(self, env: gym.Env, *, relaunch: Callable[[], gym.Env] | None = None) -> None:
        super().__init__(env)
        self._relaunch = relaunch
        #: the episodes to replay, oldest first; the last is the current one.
        self._episodes: list[dict[str, Any]] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict]:
        """Reset the env and start recording a new episode.

        :param seed: the episode's seed; required.
        :param options: passed to the env's ``reset`` and replayed with it.
        :raises ValueError: ``seed`` is ``None``.
        """
        if seed is None:
            raise ValueError("ReplayState: reset(seed=...) is required -- an episode without a "
                             "seed cannot be replayed, so it cannot be saved")
        out = self.env.reset(seed=seed, options=options)
        episode = {"seed": seed, "options": options, "actions": []}
        if self._relaunch is None:
            self._episodes = [episode]
        else:
            self._episodes.append(episode)
        return out

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Step the env and record the action."""
        out = self.env.step(action)
        self._episodes[-1]["actions"].append(_frozen(action))
        return out

    def get_state(self) -> bytes:
        """The recorded seed(s) and actions, as bytes for :meth:`set_state`."""
        episodes = [{**e, "actions": _packed(e["actions"])} for e in self._episodes]
        return pickle.dumps({"episodes": episodes}, protocol=5)

    def set_state(self, blob: bytes) -> tuple[Any, dict]:
        """Bring the env to the state ``blob`` was taken in, by replaying it.

        :param blob: bytes from :meth:`get_state`.
        :return: the ``(obs, info)`` of the last replayed step (or reset).
        """
        episodes = pickle.loads(blob)["episodes"]
        if self._relaunch is not None:
            self.env.close()
            self.env = self._relaunch()
        for episode in episodes:
            obs, info = self.env.reset(seed=episode["seed"], options=episode["options"])
            for action in episode["actions"]:
                obs, _, _, _, info = self.env.step(action)
        self._episodes = [{**e, "actions": list(e["actions"])} for e in episodes]
        return obs, info


def _frozen(action: Any) -> Any:
    """``action`` as something the env cannot change under us (a keymap may reuse its array)."""
    return np.array(action, copy=True) if isinstance(action, np.ndarray) else action


def _packed(actions: list) -> Any:
    """``actions`` as one array when they are numbers or arrays of one shape, else the list."""
    packed = np.asarray(actions)
    return actions if packed.dtype == object else packed
