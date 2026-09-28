"""Crafter as a Gymnasium environment.

``gym.make("Crafter-v0")`` or :class:`CrafterEnv` directly. See
:mod:`crafter_gym.env`. ``gym.make("CrafterMenu-v0")`` is the same game
through the eight-button interface of :mod:`crafter_gym.menu`.
"""

from __future__ import annotations

import gymnasium as gym

from .env import CrafterEnv, import_crafter
from .menu import MenuWrapper

__all__ = ["CrafterEnv", "MenuWrapper", "import_crafter", "make_menu"]


def make_menu(**kwargs) -> MenuWrapper:
    """Build a :class:`~crafter_gym.env.CrafterEnv` behind the eight-button menu.

    :param kwargs: ``CrafterEnv``'s, plus ``direct`` for
        :class:`~crafter_gym.menu.MenuWrapper`.
    :return: the wrapped env.
    """
    direct = kwargs.pop("direct", None)
    env = CrafterEnv(**kwargs)
    return MenuWrapper(env) if direct is None else MenuWrapper(env, direct)


gym.register(id="Crafter-v0", entry_point="crafter_gym.env:CrafterEnv",
             disable_env_checker=True)
gym.register(id="CrafterMenu-v0", entry_point="crafter_gym:make_menu",
             disable_env_checker=True)
