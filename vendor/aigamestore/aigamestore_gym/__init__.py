"""AI GameStore browser games as Gymnasium environments.

``gym.make("AIGameStore/game4-v0")`` or ``AIGameStoreEnv("game4")``; see
:mod:`aigamestore_gym.env` for the design.
"""

from __future__ import annotations

import gymnasium as gym

from .env import GAME_KEYS, AIGameStoreEnv

__all__ = ["GAME_KEYS", "AIGameStoreEnv"]

for _game in GAME_KEYS:
    gym.register(id=f"AIGameStore/{_game}-v0", entry_point="aigamestore_gym.env:AIGameStoreEnv",
                 kwargs={"game": _game}, disable_env_checker=True)
