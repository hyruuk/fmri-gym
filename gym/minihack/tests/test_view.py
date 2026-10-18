"""The view each id gives: a fixed window on the map, or MiniHack's window on the agent."""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np

from minihack_levels import LAYOUTS

KEYS = ("pixel", "pixel_crop", "chars", "blstats")
TILE = 16


def make(env_id: str, **kwargs):
    return gym.make(f"minihack_levels:{env_id}", observation_keys=KEYS, **kwargs)


class ViewTest(unittest.TestCase):
    def test_allocentric_twin_shows_exactly_the_lit_room(self):
        # A lit room is wholly visible at reset, and nothing else is.
        for name, size in {"Room-5x5": 5, "Room-15x15": 15}.items():
            with self.subTest(level=name):
                env = make(f"MiniHack-{name}-Allocentric-v0")
                try:
                    obs, _info = env.reset(seed=1)
                    self.assertTrue(env.observation_space.contains(obs))
                    self.assertEqual(obs["pixel_crop"].shape, (TILE * size, TILE * size, 3))
                    shown = np.argwhere(obs["chars"] != ord(" "))
                    (row, col), (last_row, last_col) = shown.min(0), shown.max(0)
                    rows = slice(TILE * row, TILE * (last_row + 1))
                    cols = slice(TILE * col, TILE * (last_col + 1))
                    np.testing.assert_array_equal(obs["pixel_crop"], obs["pixel"][rows, cols])
                finally:
                    env.close()

    def test_stock_id_keeps_the_window_on_the_agent(self):
        env = gym.make("MiniHack-Room-5x5-v0", observation_keys=KEYS)
        try:
            obs, _info = env.reset(seed=1)
            self.assertEqual(obs["pixel_crop"].shape, (TILE * 9, TILE * 9, 3))
        finally:
            env.close()

    def test_layout_window_starts_where_the_layout_does(self):
        # Room and River: the window's corner is the agent's start minus the
        # layout's start tile, an origin found without the des-file.
        for name, layout in LAYOUTS.items():
            if layout["family"] == "maze":
                continue
            with self.subTest(level=name):
                env = make(f"MiniHack-{name}-v0")
                try:
                    obs, _info = env.reset(seed=0)
                    x, y = map(int, obs["blstats"][:2])
                    col, row = x - layout["start"][0], y - layout["start"][1]
                    lines = layout["map"].splitlines()
                    rows = slice(TILE * row, TILE * (row + len(lines)))
                    cols = slice(TILE * col, TILE * (col + len(lines[0])))
                    np.testing.assert_array_equal(obs["pixel_crop"], obs["pixel"][rows, cols])
                finally:
                    env.close()

    def test_maze_layout_follows_the_agent_unless_told_otherwise(self):
        shapes = {"agent": (TILE * 9, TILE * 9, 3), "map": (TILE * 15, TILE * 15, 3)}
        for view, shape in shapes.items():
            with self.subTest(view=view):
                env = make("MiniHack-MazeWalk-Branches-v0", view=view)
                try:
                    obs, _info = env.reset(seed=0)
                    self.assertEqual(obs["pixel_crop"].shape, shape)
                    self.assertTrue(env.observation_space.contains(obs))
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
