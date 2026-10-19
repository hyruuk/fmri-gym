"""MiniHack layouts with native navigation and optional river-entry blocking.

River puzzles forbid entering unfilled water; pushing still uses NetHack's
stochastic boulder filling. Rejected moves consume an action-limit step but
not an engine turn. The env's ``pixel_crop`` is MiniHack's, a 9x9-tile window
centred on the agent; :func:`make`, the registered entry point, wraps it in
:class:`~minihack_levels.view.MapView` for the layouts seen fixed on the map.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from minihack import LevelGenerator, MiniHackNavigation
from nle import nethack

from . import LAYOUTS
from .view import MapView

_DIRECTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


class PlanningEnv(MiniHackNavigation):
    """One fixed layout, with Discrete(4) actions north/east/south/west.

    :param layout: name registered by :mod:`minihack_levels`.
    :param observation_keys: MiniHack observations to expose.
    :param kwargs: additional upstream MiniHack options.
    """

    def __init__(
        self,
        layout: str,
        observation_keys: tuple[str, ...] = ("pixel_crop", "glyphs", "blstats", "message"),
        **kwargs: Any,
    ) -> None:
        self._layout = LAYOUTS[layout]
        self._river = self._layout["family"] == "river"
        terrain = self._layout["map"]
        level = LevelGenerator(
            map=terrain,
            lit=self._river,
            flags=("hardfloor", "premapped") if self._river else ("hardfloor",),
        )
        level.set_start_pos(tuple(self._layout["start"]))
        level.add_goal_pos(tuple(self._layout["goal"]))
        for position in self._layout.get("rocks", []):
            level.add_object("boulder", "`", tuple(position))
        keys = tuple(dict.fromkeys((*observation_keys, "chars", "blstats", "pixel")))
        super().__init__(
            des_file=level.get_des(),
            observation_keys=keys,
            actions=tuple(nethack.CompassCardinalDirection),
            max_episode_steps=1200 if self._layout["family"] == "maze" else 500,
            pet=False,
            spawn_monsters=False,
            obs_crop_h=9,
            obs_crop_w=9,
            obs_crop_pad=nethack.GLYPH_CMAP_OFF,
            **kwargs,
        )
        self._finished = True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        """Reset both NetHack RNGs.

        :param seed: core and display RNG seed for action replay.
        :param options: upstream reset options.
        :return: observations and rule metadata.
        """
        if seed is not None:
            self.seed(core=seed, disp=seed, reseed=False)
        obs, info = super().reset(seed=seed, options=options)
        self._finished = False
        self._obs = obs
        return self._obs, {**info, **self._rule_info(False)}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        """Apply a cardinal move, refusing entry into unfilled river water.

        :param action: 0=north, 1=east, 2=south, 3=west.
        :return: Gymnasium step tuple, including explicit blocked-water status.
        :raises ValueError: action is outside the four-direction action space.
        :raises RuntimeError: the environment needs a reset.
        """
        if self._finished:
            raise RuntimeError("Reset the environment before stepping")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid cardinal action: {action!r}")
        if self._river and self._water_ahead(int(action)):
            # Return independent arrays: callers may retain or modify observations.
            obs = {key: value.copy() for key, value in self._obs.items()}
            return obs, 0.0, False, False, self._rule_info(True)
        obs, reward, terminated, truncated, info = super().step(action)
        self._obs = obs
        self._finished = terminated or truncated
        return self._obs, reward, terminated, truncated, {**info, **self._rule_info(False)}

    def _water_ahead(self, action: int) -> bool:
        dx, dy = _DIRECTIONS[action]
        x, y = map(int, self._obs["blstats"][:2])
        x, y = x + dx, y + dy
        chars = self._obs["chars"]
        return 0 <= y < chars.shape[0] and 0 <= x < chars.shape[1] and chars[y, x] == ord("}")

    def _rule_info(self, blocked: bool) -> dict:
        if not self._river:
            return {}
        info = {"water_entry_forbidden": True, "blocked_water": blocked}
        if blocked:
            info["end_status"] = self.StepStatus.RUNNING
        return info


def make(layout: str, view: str = "map", **kwargs: Any) -> gym.Env:
    """Build a layout's env seen one way: the registered entry point.

    :param layout: name registered by :mod:`minihack_levels`.
    :param view: ``"map"`` for ``pixel_crop`` fixed on the level's map
        (:class:`~minihack_levels.view.MapView`), ``"agent"`` for MiniHack's
        window centred on the agent.
    :param kwargs: options for :class:`PlanningEnv`.
    :return: the env, wrapped when the view is the map's.
    :raises ValueError: ``view`` is neither.
    """
    if view not in ("map", "agent"):
        raise ValueError(f"view is 'map' or 'agent', not {view!r}")
    env = PlanningEnv(layout, **kwargs)
    return MapView(env) if view == "map" else env
