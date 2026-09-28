"""AI GameStore browser games as Gymnasium environments.

``gym.make("AIGameStore/game4-v0")``, ``gym.make("AIGameStore/game6-level3-v0")``
or ``AIGameStoreEnv("game6", level=3)``; see :mod:`aigamestore_gym.env` for the
design.
"""

from __future__ import annotations

import gymnasium as gym

from .env import GAME_KEYS, GAME_LEVELS, AIGameStoreEnv

__all__ = ["GAME_KEYS", "GAME_LEVELS", "AIGameStoreEnv"]

_ENTRY = "aigamestore_gym.env:AIGameStoreEnv"
for _game, _n in GAME_LEVELS.items():
    # The bare id plays level 1 (or the whole game, if it has no levels);
    # games with a known number of levels also get one id per level.
    gym.register(id=f"AIGameStore/{_game}-v0", entry_point=_ENTRY,
                 kwargs={"game": _game}, disable_env_checker=True)
    for _level in range(1, (_n or 0) + 1):
        gym.register(id=f"AIGameStore/{_game}-level{_level}-v0", entry_point=_ENTRY,
                     kwargs={"game": _game, "level": _level}, disable_env_checker=True)
