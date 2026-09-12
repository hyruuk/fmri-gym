"""SuperTuxKart adapter (chrplr/stk-code, ``stk_gym``) -- the real game, a
person at the wheel, fmri-gym watching.

The ``supertuxkart`` backend drives pystk2 (an older, in-process STK) and
blits its offscreen render like every other backend. This one drives the
current game from https://github.com/chrplr/stk-code, whose ``--gym-human``
mode is the reverse of a Gymnasium env: the game opens its own window
(fullscreen, on top of fmri-gym's), reads the keyboard or gamepad itself, and
runs on its own clock with its own frame pacing and sound; a JSON protocol on
its stdin/stdout only *reports*. So there is nothing to blit and no key to
forward: each fmri-gym frame this adapter polls the race state and logs the
row ``stk_gym.HumanSession.sample`` makes of it -- position, speed, progress,
rank, and the ``ctrl_*`` controls the kart applied, the record of what the
participant did. The pygame window shows a placeholder behind the game.

What that buys: the participant plays the actual game, with native input
latency and rendering, and the log is a trajectory sampled at the block's
``fps`` on fmri-gym's session clock. What it does not: no frames are logged
(the game's own ``--history`` recording is the tool for a tick-exact replay),
keypresses are not timestamped individually, and a block is not replayable
from ``episode_seeds`` + ``actions`` -- ``actions`` are empty, the participant
drove the game directly. fmri-gym's ESC does not reach a window that does not
have focus; quitting the game (its pause menu) ends the block instead: the
adapter sets ``quit_requested`` and the session saves the block and stops.

Phase fields: ``track`` (default hacienda), ``laps``, ``num_karts``,
``difficulty`` (0-3), ``fullscreen`` (default true), ``screensize`` ("WxH"),
``race_now`` (skip the ready-set-go countdown; default false), ``extra_args``
(any other supertuxkart flags), ``binary`` (else ``$STK_ENV_BIN``, else the
fork's build directory, as ``stk_gym.find_binary`` looks).

Install: build the fork (``cmake -B build && cmake --build build -j``) and
``pip install -e <fork>/python``; point ``STK_ENV_BIN`` at
``<fork>/build/bin/supertuxkart`` if it is not found on its own.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import EnvAdapter, FrameState
from .keyspec import PassthroughKeySpec

_W, _H = 1024, 768
_BG = (20, 20, 20)
_TEXT = (200, 200, 200)


class STKGymAdapter(EnvAdapter):
    name: str = "stk_gym"

    def _make(self, spec: dict) -> Any:
        import stk_gym
        self._t0 = time.perf_counter()
        session = stk_gym.HumanSession(
            track=spec.get("track", "hacienda"),
            laps=spec.get("laps"),
            num_karts=spec.get("num_karts"),
            difficulty=spec.get("difficulty"),
            fullscreen=bool(spec.get("fullscreen", True)),
            screensize=spec.get("screensize"),
            race_now=bool(spec.get("race_now", False)),
            extra_args=list(spec.get("extra_args", [])),
            binary=spec.get("binary"),
        )
        self._prev: dict = {}
        self._frame = _placeholder()
        #: Set once the race is gone (the game quit, or left through its pause
        #: menu); session.py ends the block on it, saving the data.
        self.quit_requested = False
        return session

    def _keyspec(self) -> PassthroughKeySpec:
        # Keys go to the game's window, not to fmri-gym: nothing to map.
        return PassthroughKeySpec(combos={}, noop="")

    def reset(self, seed: int | None) -> tuple[Any, dict]:
        self._prev = self._poll(lambda: self.env.reset(seed))
        return None, self._info()

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        from stk_gym import compute_reward
        state = self._poll(self.env.state)
        if self.quit_requested:
            return None, 0.0, False, True, self._info()
        reward = compute_reward(self._prev, state, scheme="progress_only",
                                track_length=self.env.track_length)
        self._prev = state
        return None, reward, bool(state.get("finished", False)), False, self._info()

    def render(self) -> np.ndarray:
        return self._frame

    def capture(
        self, obs: Any, info: dict, want_blob: bool = True
    ) -> FrameState:
        variables = dict(info["_sample"])
        # The game's race clock against ours: constant while the race runs,
        # stepping whenever the race clock is stopped (intro, countdown, pause).
        variables["stk_time_minus_adapter"] = (
            variables["time"] - (time.perf_counter() - self._t0))
        return FrameState(blob=None, variables=variables)

    def close(self) -> None:
        self.env.close()

    def _poll(self, fn: Any) -> dict:
        """One protocol call. A race that is gone -- the game exited, or the
        state has no tick because the pause menu dropped it to the main menu
        -- ends the block rather than raising: the data must be saved."""
        import stk_gym
        if self.quit_requested:
            return self._prev
        try:
            state = fn()
        except stk_gym.EngineDied:
            state = {}
        if "tick" not in state:
            self.quit_requested = True
            return self._prev
        return state

    def _info(self) -> dict:
        info = dict(self.env.info())
        info["_sample"] = self.env.sample()
        return info


def _placeholder() -> np.ndarray:
    """What the pygame window shows while the game's own window is on top."""
    img = np.empty((_H, _W, 3), dtype=np.uint8)
    img[:] = _BG
    try:
        import pygame
    except ImportError:
        return img
    if not pygame.font.get_init():
        pygame.font.init()
    font = pygame.font.Font(pygame.font.get_default_font(), 28)
    surf = font.render("SuperTuxKart is running in its own window", True, _TEXT, _BG)
    arr = pygame.surfarray.array3d(surf).transpose(1, 0, 2)
    h, w = arr.shape[:2]
    y0, x0 = (_H - h) // 2, (_W - w) // 2
    img[y0:y0 + h, x0:x0 + w] = arr
    return img
