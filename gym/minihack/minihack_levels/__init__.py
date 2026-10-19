"""Levels and views for MiniHack, as Gymnasium ids reached as ``minihack_levels:<id>``.

Two kinds of id. Each layout in ``layouts.json`` is ``MiniHack-<Layout>-v0``
(:class:`~minihack_levels.env.PlanningEnv`): Room and River are seen fixed on
the map, MazeWalk through MiniHack's window centred on the agent, so the player
has to remember the way; ``gym.make(id, view="map")`` or ``view="agent"``
overrides that. And every id MiniHack registers gets an ``-Allocentric-v0``
twin (``MiniHack-Room-5x5-v0`` -> ``MiniHack-Room-5x5-Allocentric-v0``): the
same task with ``pixel_crop`` fixed on the level's map
(:class:`~minihack_levels.view.MapView`). MiniHack's own id is the egocentric
view, a window centred on the agent.
"""

from __future__ import annotations

import json
import re
from importlib.resources import files

import minihack  # noqa: F401  (registers the MiniHack-* ids the twins are made from)
from gymnasium import registry
from gymnasium.envs.registration import register

LAYOUTS = json.loads(files(__package__).joinpath("layouts.json").read_text())

#: MiniHack's own ids, each the egocentric view of its task.
STOCK_IDS = tuple(env_id for env_id in registry if env_id.startswith("MiniHack-"))

for env_id in STOCK_IDS:
    register(
        id=re.sub(r"-v(\d+)$", r"-Allocentric-v\1", env_id),
        entry_point="minihack_levels.view:allocentric",
        kwargs={"env_id": env_id},
    )

for name, layout in LAYOUTS.items():
    register(
        id=f"MiniHack-{name}-v0",
        entry_point="minihack_levels.env:make",
        kwargs={"layout": name, "view": "agent" if layout["family"] == "maze" else "map"},
        max_episode_steps=1200 if layout["family"] == "maze" else 500,
    )

from .seeding import NLESeeding  # noqa: E402
from .view import MapView  # noqa: E402

__all__ = ["LAYOUTS", "STOCK_IDS", "MapView", "NLESeeding"]
