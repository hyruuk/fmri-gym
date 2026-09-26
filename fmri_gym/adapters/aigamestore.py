"""AI GameStore adapter (aigamestore_gym) -- the ten browser games, lock-stepped.

``aigamestore_gym`` (``vendor/aigamestore/``) wraps each AI GameStore game -- an
LLM-generated p5.js / three.js page -- in a Gymnasium env by taking over the
page's clock: one ``step`` holds a set of keys and advances the game exactly
``frame_skip`` of its 60 Hz frames, then returns the canvas and the game's own
``getGameState()``. A model deliberating between steps and a subject at the
button box therefore meet the same game, and a block replays from
``episode_seeds`` + ``actions``. The action is ``MultiBinary`` over the game's
keys (``env.keys``), so held keys combine, as on a keyboard.

An episode is one level, named in the phase: ``"game": "game6/level3"`` plays
level 3 of game6 afresh every episode, and the episode ends on a win, a loss
or a level clear (the env's rule; see its docstring). A curriculum therefore
lists one game phase per level it wants played. ``"game": "game6"`` is level 1;
game4 has no levels and takes no ``/level``.

Keys: combo VALUES are the game's key names (``"LEFT"``, ``"SPACE"``, ...) and
combo keys are what the subject presses, so ``"keys": {"B1": "LEFT"}`` binds a
button box. The games paint control hints on the canvas; the env relabels them
from the same map, so the hint names the button the subject actually presses.

Timing: lock-stepped, one ``step`` per fmri-gym frame, so ``fps`` must equal
``60 / frame_skip`` (10 by default); ``_make`` refuses a config where they
disagree. Not supported: sound (the games have none) and savestates.

Phase fields: ``game`` ("game1".."game10", with an optional "/level<N>", or an
http(s):// URL to an index.html -- then ``game_keys`` lists the keys that game
listens for),
``frame_skip`` (default 6), ``headed``, ``browser_channel`` ("chrome" by
default; ``null`` for Playwright's bundled Chromium), ``games_dir``.
"""

from __future__ import annotations

from typing import Any

from .base import EnvAdapter, FrameState
from .keyspec import MultiKeySpec


class AIGameStoreAdapter(EnvAdapter):
    name: str = "aigamestore"

    def _make(self, spec: dict) -> Any:
        from aigamestore_gym import AIGameStoreEnv

        keys = spec.get("keys", {})
        # "game6/level3" -> game6, level 3; an episode is one level.
        game, _, level = spec["game"].partition("/level")
        env = AIGameStoreEnv(
            game,
            level=int(level) if level else None,
            keys=spec.get("game_keys"),
            frame_skip=int(spec.get("frame_skip", 6)),
            headless=not spec.get("headed", False),
            browser_channel=spec.get("browser_channel", "chrome"),
            # Hints name the pressed key where it differs from the game's own.
            key_labels={want: pressed for pressed, want in keys.items() if pressed != want},
            games_dir=spec.get("games_dir"),
        )
        unknown = set(keys.values()) - set(env.keys)
        fps = spec["fps"]
        if unknown or env.metadata["render_fps"] != fps:
            env.close()
        if unknown:
            raise ValueError(f"keys {sorted(unknown)} are not keys of {spec['game']}: {env.keys}")
        if env.metadata["render_fps"] != fps:
            raise ValueError(f"fps={fps} but the game steps at {env.metadata['render_fps']:g} Hz "
                             f"(60 / frame_skip {env.frame_skip}); set frame_skip so they match")
        return env

    def _keyspec(self) -> MultiKeySpec:
        # Combo values are the game's key names; button_map turns each into the
        # env's 0/1 vector, so a curriculum keymap stays written in key names.
        n = len(self.env.keys)
        button_map = {k: [int(i == j) for j in range(n)] for i, k in enumerate(self.env.keys)}
        button_map["NOOP"] = [0] * n
        return MultiKeySpec(combos={frozenset([k]): k for k in self.env.keys}, noop="NOOP",
                            button_map=button_map)

    def native_fps(self) -> float:
        return self.env.metadata["render_fps"]

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        # The scalar fields of the game's own state (score, level, gamePhase,
        # lives, ...), one state_* variable each.
        return FrameState(blob=None, variables={
            f"state_{name}": value for name, value in info["state"].items()})
