"""Seed NetHack's own RNGs on every reset, for any NLE or MiniHack env.

Gymnasium's ``reset(seed=)`` seeds only the env's ``np_random``. NetHack's two
RNGs (core and disp) are set by NLE's ``seed(core=, disp=, reseed=)``, which
applies to the next reset and to that one only. Without it every reset draws
true randomness and the game reseeds itself mid-play (NetHack's
anti-automation measure), so no replay can match it. The env also needs its
``fix_moon_phase`` kwarg: NetHack reads the real date for luck (full moon,
Friday the 13th), and with it the date derives from the seed.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym


class NLESeeding(gym.Wrapper):
    """Seed NetHack's RNGs from ``reset(seed=)``, and refuse a reset where that did not take.

    ``get_seeds()`` reports what the reset actually used, so a reset the seed
    did not reach is caught here rather than found in a replay that diverges.
    """

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict]:
        """Seed core and disp with ``seed`` (no reseeding), reset, and check the seeds held.

        :raises RuntimeError: NetHack reset with other seeds.
        """
        wanted = self.env.unwrapped.seed(core=seed, disp=seed, reseed=False)[:3]
        out = self.env.reset(seed=seed, options=options)
        used = self.env.unwrapped.get_seeds()[:3]
        if tuple(used) != tuple(wanted):
            raise RuntimeError(f"NetHack reset with seeds {used}, not {wanted}: "
                               "the episode would not replay")
        return out
