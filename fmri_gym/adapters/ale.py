"""ALE (Atari 2600) EnvAdapter.

Exposes the Atari-specific bits behind the standard EnvAdapter interface:
- per-frame exact savestate via clone_state (restorable, determinism-free);
- the block's end state with the screen it showed: the emulator's frame buffer
  is not part of its state, and after a restore it still holds the last frame
  emulated, so a resumed episode would open on its reset's picture. ``render``
  shows the saved screen until the next step. The per-frame savestates leave
  the screen out: a replay judges them by the frames stepped after them;
- state variables: the 128-byte console RAM (+ optional lossless indexed pixels)
  surfaced through FrameState.variables.

The phase's ``keys`` are indices into the game's own action set, which differs
per game: ``gym.make(id).unwrapped.get_action_meanings()`` lists it (Pong:
NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE). NOOP is 0 for every game, so
``"": 0`` is the no-key entry.

Paddle games (Pong, Breakout, ...) were made for a rotary controller; ALE
turns a held RIGHT/LEFT into a sweep at a fixed rate, fast enough to cross
Pong's field in about half a second. ``"paddle_strength"`` (0 < s <= 1,
default 1) scales that sweep through ALE's own ``act(action, paddle_strength)``;
joystick games ignore it. The phase records it, so a replay uses the same value.
"""

from __future__ import annotations

import pickle
from typing import Any

import gymnasium as gym
import numpy as np

from .base import EnvAdapter, FrameState


class _PaddleStrength:
    """An ALE interface whose ``act`` moves the paddles at ``strength`` of full speed.

    ``AtariEnv.step`` always calls ``ale.act(action, 1.0)`` in its discrete
    mode; this stands in for ``env.unwrapped.ale`` and forwards everything else.
    """

    def __init__(self, ale: Any, strength: float) -> None:
        self._ale, self._strength = ale, strength

    def act(self, action: Any, paddle_strength: float = 1.0) -> int:
        return self._ale.act(action, paddle_strength * self._strength)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ale, name)


class ALEAdapter(EnvAdapter):
    name: str = "ale"
    #: the screen a restored end state showed, until the next step
    _screen: np.ndarray | None = None

    def _make(self, spec: dict) -> gym.Env:
        import ale_py
        gym.register_envs(ale_py)
        # Set "save_pixels": true in the phase for lossless indexed-pixel logging.
        self.save_pixels = bool(spec.get("save_pixels", False))
        # Block-wide palette for that logging.
        self._palette = np.zeros((256, 3), dtype=np.uint8)
        self._palette_seen = np.zeros(256, dtype=bool)
        env = gym.make(
            spec["game"], render_mode="rgb_array",
            frameskip=1, repeat_action_probability=0.0)
        strength = float(spec.get("paddle_strength", 1.0))
        if strength != 1.0:
            env.unwrapped.ale = _PaddleStrength(env.unwrapped.ale, strength)
        return env

    def native_fps(self) -> float:
        """The console's 60 frames per second; ``_make`` steps one frame at a time.

        Stated here because ALE does not tell: its ``render_fps`` metadata is a
        fixed 30 whatever the frameskip, which is half the speed of the machine.
        """
        return 60.0

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        ale = self.env.unwrapped.ale
        variables = {"ram": ale.getRAM().copy(), **(info or {})}
        if self.save_pixels:
            idx = ale.getScreen()  # (210,160) uint8 palette indices
            new = np.unique(idx)
            new = new[~self._palette_seen[new]]
            if new.size:
                flat_i, flat_c = idx.reshape(-1), obs.reshape(-1, 3)
                for i in new:
                    self._palette[i] = flat_c[flat_i == i][0]
                    self._palette_seen[i] = True
            variables["screen_index"] = idx.copy()
        blob = (pickle.dumps(self.env.unwrapped.clone_state(include_rng=True))
                if want_blob else None)
        return FrameState(blob=blob, variables=variables)

    def reset(self, seed: int) -> tuple[Any, dict]:
        self._screen = None
        return super().reset(seed)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        self._screen = None
        return super().step(action)

    def render(self) -> np.ndarray:
        return self._screen if self._screen is not None else super().render()

    def restore(self, blob: bytes) -> None:
        """Restore a frame's savestate, or the block's end state with its screen."""
        state = pickle.loads(blob)
        if isinstance(state, tuple):        # block_extra's: (state, screen)
            state, self._screen = state
        self.env.unwrapped.restore_state(state)

    def block_extra(self) -> dict:
        """The state the block ended in with its screen, and the palette if save_pixels."""
        state = self.env.unwrapped.clone_state(include_rng=True)
        extra = {"state": pickle.dumps((state, self.render().copy()))}
        if self.save_pixels:
            extra["palette"] = self._palette
        return extra
