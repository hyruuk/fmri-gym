"""MapView's window is the level's map: what a lit level shows, and PlanningEnv's own crop."""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np

from minihack_levels import LAYOUTS, MapView

KEYS = ("pixel", "pixel_crop", "chars", "blstats")


class MapViewTest(unittest.TestCase):
    def test_window_is_what_a_lit_level_shows(self):
        # Lit upstream rooms: every map tile is visible at reset, nothing else is.
        for name, (height, width) in {"Room-5x5": (5, 5), "Room-15x15": (15, 15)}.items():
            with self.subTest(level=name):
                env = MapView(gym.make(f"MiniHack-{name}-v0", observation_keys=KEYS))
                try:
                    obs, _info = env.reset(seed=1)
                    self.assertTrue(env.observation_space.contains(obs))
                    self.assertEqual(obs["pixel_crop"].shape, (16 * height, 16 * width, 3))
                    shown = np.argwhere(obs["chars"] != ord(" "))
                    (row, col), (last_row, last_col) = shown.min(0), shown.max(0)
                    rows = slice(16 * row, 16 * (last_row + 1))
                    cols = slice(16 * col, 16 * (last_col + 1))
                    window = obs["pixel"][rows, cols]
                    np.testing.assert_array_equal(obs["pixel_crop"], window)
                finally:
                    env.close()

    def test_window_matches_planning_env_fixed_view(self):
        for name, layout in LAYOUTS.items():
            if layout["family"] == "maze":
                continue
            with self.subTest(level=name):
                inner = gym.make(f"minihack_levels:MiniHack-{name}-v0", observation_keys=KEYS)
                env = MapView(inner)
                try:
                    obs, _info = env.reset(seed=0)
                    own = inner.unwrapped._view(dict(obs))["pixel_crop"]
                    np.testing.assert_array_equal(obs["pixel_crop"], own)
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
