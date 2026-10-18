"""Crafter as a Gymnasium environment.

``gym.make("Crafter-v0")`` is crafter's seventeen actions (:mod:`crafter_gym.env`),
``gym.make("CrafterMenu-v0")`` the same game through the eight-button
interface of :mod:`crafter_gym.menu`. Either takes ``level=``, one of the
scanner paradigm's four rule variants (:mod:`crafter_gym.levels`), which on the
levels that ask for it also names one task at a time (:mod:`crafter_gym.tasks`).

The wrappers stack from the inside out as what the world does (the level), how
it is played (the menu), and what it is played for (the tasks): the task chain
is outermost because its skip entry is a menu press the game never sees.
"""

from __future__ import annotations

from typing import TypeVar

import gymnasium as gym

from .env import CrafterEnv, import_crafter
from .levels import LEVELS, LevelWrapper, reachable_achievements
from .menu import MenuWrapper
from .tasks import ALL_DONE_TEXT, SKIP_ENTRY, TaskWrapper, chain_for, task_label, task_text

__all__ = ["ALL_DONE_TEXT", "LEVELS", "CrafterEnv", "LevelWrapper", "MenuWrapper",
           "TaskWrapper", "chain_for", "find_wrapper", "import_crafter", "level_of",
           "make_menu", "make_plain", "reachable_achievements", "task_label", "task_text"]

W = TypeVar("W", bound=gym.Wrapper)


def make_plain(level: str | None = None, **kwargs) -> gym.Env:
    """Crafter with its seventeen actions, in a level if one is given.

    With no menu there is no skip entry, so a task level built this way names
    its tasks with no way to put one off.

    :param level: a key of :data:`~crafter_gym.levels.LEVELS`, or ``None``
        for stock crafter.
    :param kwargs: ``CrafterEnv``'s.
    :return: the env.
    """
    env: gym.Env = CrafterEnv(**kwargs)
    if level is None:
        return env
    env = LevelWrapper(env, level)
    chain = chain_for(level)
    return TaskWrapper(env, chain) if chain else env


def make_menu(level: str | None = None, direct: tuple[int, ...] | None = None,
              **kwargs) -> gym.Env:
    """Crafter behind the eight-button menu, in a level if one is given.

    :param level: a key of :data:`~crafter_gym.levels.LEVELS`, or ``None``
        for stock crafter.
    :param direct: the actions that keep a button (see :class:`MenuWrapper`).
    :param kwargs: ``CrafterEnv``'s.
    :return: the env.
    """
    env: gym.Env = CrafterEnv(**kwargs)
    chain = ()
    if level is not None:
        env = LevelWrapper(env, level)
        chain = chain_for(level)
    menu_kwargs = {} if direct is None else {"direct": direct}
    env = MenuWrapper(env, extra=(SKIP_ENTRY,) if chain else (), **menu_kwargs)
    return TaskWrapper(env, chain) if chain else env


def find_wrapper(env: gym.Env, kind: type[W]) -> W | None:
    """The wrapper of this kind in an env chain, or ``None``.

    Gymnasium forwards no attribute through a wrapper, so a caller that needs
    the menu's cursor or the chain's task asks for that layer by kind.
    """
    while isinstance(env, gym.Wrapper):
        if isinstance(env, kind):
            return env
        env = env.env
    return None


def level_of(env: gym.Env) -> str | None:
    """The level this env chain plays, or ``None`` for stock crafter."""
    wrapper = find_wrapper(env, LevelWrapper)
    return None if wrapper is None else wrapper.level


gym.register(id="Crafter-v0", entry_point="crafter_gym:make_plain", disable_env_checker=True)
gym.register(id="CrafterMenu-v0", entry_point="crafter_gym:make_menu", disable_env_checker=True)
