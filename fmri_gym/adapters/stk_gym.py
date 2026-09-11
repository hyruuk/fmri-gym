"""SuperTuxKart adapter (chrplr/stk-code, ``stk_gym``) -- the real game, a
person at the wheel, fmri-gym watching.

The ``supertuxkart`` backend drives pystk2 (an older, in-process STK) and
blits its offscreen render like every other backend. This one drives the
current game from https://github.com/chrplr/stk-code, whose ``--gym-human``
mode is the reverse of a Gymnasium env: the game opens its own window
(fullscreen, on top of fmri-gym's), reads the keyboard or gamepad itself, and
runs on its own clock with its own frame pacing and sound; a JSON protocol on
its stdin/stdout only *reports*. So there is nothing to blit and no key to
forward. Each fmri-gym frame this adapter polls the race state -- position,
speed, progress, rank, and the ``controls`` the kart applied, which are the
record of what the participant did -- and logs it. The pygame window shows a
placeholder behind the game.

What that buys: the participant plays the actual game, with native input
latency and rendering, and the log is a trajectory sampled at the block's
``fps`` plus the applied controls, on fmri-gym's session clock. What it does
not: no frames are logged (the game's own ``--history`` recording would be the
tool for a tick-exact replay), keypresses are not timestamped individually,
and fmri-gym's ESC does not reach a window that does not have focus -- quitting
the game (its pause menu) ends the block instead: the adapter sets
``quit_requested`` and the session saves the block and stops, as after ESC.

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

from .keyspec import PassthroughKeySpec
from .base import EnvAdapter, FrameState

_W, _H = 1024, 768
_BG = (20, 20, 20)
_TEXT = (200, 200, 200)

# Scalars copied from the race state into the logged variables, verbatim.
_STATE_SCALARS = (
    "tick", "time", "phase", "speed", "max_speed", "heading", "pitch", "roll",
    "distance_down_track", "distance_to_center", "overall_distance",
    "on_road", "on_ground", "wrong_way", "rank", "finished_laps", "finished",
    "finish_time", "nitro_energy", "powerup", "num_powerup", "eliminated",
)
_CONTROLS = ("steer", "accel", "brake", "nitro", "skid", "fire", "rescue", "look_back")


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
        self._track_length = float(session.track_length)
        self._prev_progress = 0.0
        self._frame: np.ndarray | None = None
        self._n_polls = 0
        self._last_state: dict = {}
        #: Set once the game has exited; session.py ends the block on it.
        self.quit_requested = False
        return session

    def _keyspec(self) -> PassthroughKeySpec:
        # Keys go to the game's window, not to fmri-gym: nothing to map. The
        # loop still resolves the held set each frame; it is always empty.
        return PassthroughKeySpec(combos={}, noop="")

    def reset(self, seed: int | None) -> tuple[Any, dict]:
        state = self._call(lambda: self.env.reset(seed))
        self._prev_progress = self._progress(state)
        return None, self._info(state)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        state = self._call(self.env.state)
        if "tick" not in state:
            # No race to report: the participant left it through the pause
            # menu (STK is back at its main menu) or the game is gone. Either
            # way the block is over; _game then saves it.
            self.quit_requested = True
        if self.quit_requested:
            return None, 0.0, False, True, self._info(state)
        self._n_polls += 1
        progress = self._progress(state)
        reward = progress - self._prev_progress   # metres gained since the last sample
        self._prev_progress = progress
        terminated = bool(state.get("finished", False))
        return None, reward, terminated, False, self._info(state)

    def render(self) -> np.ndarray:
        if self._frame is None:
            self._frame = _placeholder()
        return self._frame

    def capture(
        self, obs: Any, info: dict, want_blob: bool = True
    ) -> FrameState:
        state = info.get("_state", {}) if isinstance(info, dict) else {}
        # Every key, every frame -- a missing field (the race gone) is NaN,
        # never an absent row, or the columns would fall out of step with
        # session_time.
        variables: dict[str, Any] = {}
        for key in _STATE_SCALARS:
            variables[key] = _number(state.get(key))
        xyz = state.get("xyz")
        variables["xyz"] = ([float(c) for c in xyz]
                            if isinstance(xyz, (list, tuple)) and len(xyz) == 3
                            else [float("nan")] * 3)
        controls = state.get("controls") or {}
        for key in _CONTROLS:
            variables["ctrl_" + key] = _number(controls.get(key))
        variables["progress_m"] = self._progress(state) if "tick" in state else float("nan")
        # Race time versus the time since this adapter started polling: with
        # the game on its own clock, this is what says whether the two drift.
        variables["stk_time_minus_adapter"] = (
            float(state.get("time", 0.0)) - (time.perf_counter() - self._t0))
        return FrameState(blob=None, variables=variables)

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass

    # -- Internals -----------------------------------------------------------

    def _call(self, fn: Any) -> dict:
        """Run one protocol call. A game that has exited (its pause menu,
        a crash) is not an error here: the block ends, and the data stays."""
        import stk_gym
        if self.quit_requested:
            return self._last_state
        try:
            self._last_state = fn()
        except stk_gym.EngineDied:
            self.quit_requested = True
            return self._last_state
        return self._last_state

    def _progress(self, state: dict) -> float:
        from stk_gym import progress_of
        return float(progress_of(state, self._track_length))

    def _info(self, state: dict) -> dict:
        # The flat summary stk_gym's env gives, plus the raw state for capture.
        info = dict(self.env.info())
        info["_state"] = state
        return info


def _number(v: Any) -> float:
    """A state field as a float (bools become 0/1), NaN when absent."""
    if isinstance(v, (bool, int, float)):
        return float(v)
    return float("nan")


def _placeholder() -> np.ndarray:
    """What the pygame window shows while the game's own window is on top."""
    img = np.empty((_H, _W, 3), dtype=np.uint8)
    img[:] = _BG
    try:
        import pygame
        if not pygame.font.get_init():
            pygame.font.init()
        font = pygame.font.Font(pygame.font.get_default_font(), 28)
        surf = font.render("SuperTuxKart is running in its own window", True, _TEXT, _BG)
        arr = pygame.surfarray.array3d(surf).transpose(1, 0, 2)
        h, w = arr.shape[:2]
        y0, x0 = (_H - h) // 2, (_W - w) // 2
        img[y0:y0 + h, x0:x0 + w] = arr
    except Exception:
        pass
    return img
