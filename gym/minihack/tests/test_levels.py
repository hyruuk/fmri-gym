"""Native-engine solutions, replay, and river-entry regression checks."""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np

CASES = {
    "River-Reposition": (0, "NNEEESSNNWWSSEEESSWNWNEEEEEE"),
    "River-ThreeDeliveries": (0, "ESEESENWNESWWWNEEENNWSESEWSWWNEEEEEEE"),
    "River-FourStoneWarehouse": (1, "SSENEESENWNESWWWNEEENNWSESEWSWWNEEEEWWNNWWWSWSEEEEEEEEEE"),
    "River-CommitToARoute": (3, "NEEEESSSWWWNNSEENENNWWSEEEWNWWWWSEEEEEEWWSWSSENNWNEEEEEEEES"),
    "Room-LookBeforeMoving": (0, "EESSSSSSEENEEENNNNNE"),
    "Room-FindTheConnections": (0, "EEEEENEEENEENNNNNNNNEEEES"),
    "Room-BeyondTheNextWall": (0, "NWWWWWNNNNNWWWNNNNNNNWWSSSSSWWWSSSSSSSWWNNNNNNNNNNNNWWW"),
    "MazeWalk-Branches": (0, "EEEESSWWWWSSEESSEESSWWWWNNWWWWSSWWSSEESSWWWWNN"),
    "MazeWalk-Loops": (0, "SSSSSSSSSSSWWNNNNNWWWWNNWWWWSSSSWWWWSSEESSWW"),
}


def make(name, **kwargs):
    return gym.make(f"minihack_levels:MiniHack-{name}-v0", **kwargs)


class LevelsTest(unittest.TestCase):
    def test_solutions_and_replay(self):
        for name, (seed, route) in CASES.items():
            with self.subTest(level=name):
                env = make(name)
                try:
                    records = []
                    for _ in range(2):
                        obs, _info = env.reset(seed=seed)
                        self.assertTrue(env.observation_space.contains(obs))
                        frames = [obs["glyphs"].copy()]
                        rewards = []
                        for index, direction in enumerate(route):
                            obs, reward, done, truncated, _info = env.step("NESW".index(direction))
                            self.assertTrue(env.observation_space.contains(obs))
                            self.assertFalse(truncated)
                            self.assertEqual(done, index == len(route) - 1)
                            frames.append(obs["glyphs"].copy())
                            rewards.append(reward)
                        self.assertGreater(rewards[-1], 0)
                        records.append((np.stack(frames), rewards))
                    np.testing.assert_array_equal(records[0][0], records[1][0])
                    self.assertEqual(records[0][1], records[1][1])
                finally:
                    env.close()

    def test_water_rejection_and_action_limit(self):
        env = make("River-Reposition", max_episode_steps=8)
        try:
            obs, info = env.reset(seed=0)
            for direction in "SEEEEN":
                obs, _, done, truncated, _ = env.step("NESW".index(direction))
                self.assertFalse(done or truncated)
            position = obs["blstats"][:2].copy()
            for attempt in range(2):
                obs, reward, done, truncated, info = env.step(1)
                self.assertTrue(info["blocked_water"])
                self.assertEqual(reward, 0)
                np.testing.assert_array_equal(position, obs["blstats"][:2])
                self.assertFalse(done)
                self.assertEqual(truncated, attempt == 1)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
