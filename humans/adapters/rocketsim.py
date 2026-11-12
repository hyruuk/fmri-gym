"""RocketSim adapter (``rocketsim``) -- Rocket League's soccar, one car against a bot.

The game is ``rocketsim_gym`` (github.com/hyruuk/rocketsim-gym, pinned in pyproject.toml):
RocketSim's physics of Rocket League, a 1v1 match's rules (kickoff countdown, clock, goals,
overtime, demolitions) against a scripted ball-chaser, and frames drawn offscreen with OpenGL
from Rocket League's car cam and ball cam, its scoreboard, clock and boost meter included. The
subject drives the blue car.

Actions: the env's digital input (its default ``input_mode``) is a MultiBinary(12) of a
controller's buttons, so a phase's ``controls`` are button indices and held controls combine:
0 throttle, 1 reverse, 2 stick left, 3 stick right (steer on the ground, yaw in the air),
4 stick up (nose down), 5 stick down (nose up), 6 jump, 7 boost, 8 powerslide (in the air,
the stick left / right rolls), 9 ball cam (a press switches the camera), 10 air roll left,
11 air roll right.

``env_kwargs`` go to the env: ``match_seconds``, ``ticks_per_step``, ``render_width``,
``render_height``, ``camera`` (Rocket League's camera settings by name) and ``controller``
(dead zone and sensitivities).

Savestate: RocketSim's own state set drops Bullet's contact caches, so a restored arena drifts
at its next contact. The env is deterministic from ``reset(seed=)`` and the actions, so the
savestate is ``replay_state_gym.ReplayState``'s: the seed and the actions since.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from .base import EnvAdapter, FrameState


class RocketSimAdapter(EnvAdapter):
    name: str = "rocketsim"

    def _make(self, spec: dict) -> gym.Env:
        """A soccar match whose frames are drawn, behind a savestate by replay.

        :raises RuntimeError: no OpenGL 3.3 context opens to draw the frames.
        """
        import rocketsim_gym  # noqa: F401  (registers RocketSimSoccar-v0)
        from replay_state_gym import ReplayState
        return ReplayState(gym.make(spec["game"], render_mode="rgb_array",
                                    **spec.get("env_kwargs", {})))

    def native_fps(self) -> float:
        """Physics ticks per second over the ticks a step advances."""
        return self.env.unwrapped.metadata["render_fps"]

    def outcome(self, terminated: bool, truncated: bool) -> tuple[str, str]:
        # A match ends only with a winner: a tie at 0:00 goes to overtime.
        if not terminated:
            return super().outcome(terminated, truncated)
        blue, orange = self.last_info["blue_score"], self.last_info["orange_score"]
        result = "won" if blue > orange else "lost"
        return result, f"You {result} {blue} - {orange}"

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        # Every SAMPLE_FIELDS column every frame: the match, the ball, both cars, what happened
        # during the step and the controls each car obeyed.
        return FrameState(variables=self.env.unwrapped.sample())

    def restore(self, blob: bytes) -> None:
        self.env.set_state(blob)

    def block_extra(self) -> dict:
        """The block's end state, and what its frames depend on besides seeds and actions.

        :return: ``{"state": ..., "provenance": ...}``: the versions, the collision meshes'
            SHA-256s, the tick rates, the OpenGL renderer and the camera and controller
            settings; a block replays bit for bit only where they are the same.
        """
        return {"state": self.env.get_state(), "provenance": self.env.unwrapped.provenance()}
