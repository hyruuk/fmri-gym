"""Baba Is You (baba-is-auto) as a Gymnasium environment.

``gym.make("BabaAuto-v0", game="baba_is_you")`` or :class:`BabaAutoEnv`
directly. See :mod:`baba_auto_gym.env`.
"""

from __future__ import annotations

import gymnasium as gym

from .env import BabaAutoEnv

__all__ = ["BabaAutoEnv"]

gym.register(id="BabaAuto-v0", entry_point="baba_auto_gym.env:BabaAutoEnv",
             disable_env_checker=True)
