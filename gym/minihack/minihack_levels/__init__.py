"""Register layouts through Gymnasium's module-qualified environment IDs."""

from __future__ import annotations

import json
from importlib.resources import files

from gymnasium.envs.registration import register

LAYOUTS = json.loads(files(__package__).joinpath("layouts.json").read_text())

for name, layout in LAYOUTS.items():
    register(
        id=f"MiniHack-{name}-v0",
        entry_point="minihack_levels.env:PlanningEnv",
        kwargs={"layout": name},
        max_episode_steps=1200 if layout["family"] == "maze" else 500,
    )
