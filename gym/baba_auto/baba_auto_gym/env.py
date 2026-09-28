"""Baba Is Auto (utilForever/baba-is-auto) as a Gymnasium env.

baba-is-auto is a C++17 simulator of Baba Is You: it loads a text map, parses
the rules the word tiles spell out, and applies pushes, transformations, SINK,
HOT/MELT, DEFEAT, OPEN/SHUT and the rest of the ruleset (its ARCHITECTURE.md
is the contract). Its ``pyBaba`` module exposes ``Game`` alone: no gym env
(the ``Extensions/BabaRL`` ones speak ``gym`` 0.24, hard-code a map path and
open a window to render). This env holds one ``pyBaba.Game`` and presents the
Gymnasium contract in front of it; the rules stay in the simulator. The
maps and sprites are read from a utilForever/baba-is-auto checkout, ``repo=``
or fmri-gym's ``external/baba_auto`` (the one ``pyBaba`` was built from).

The picture is the one baba-is-auto's own GUI draws (``Extensions/BabaGUI``):
its 24x24 sprites, one tile per cell, KEKE turned to face where it walks, on
black; ``render()`` returns it as ``(24 * H, 24 * W, 3)`` uint8. A map that
uses an object the GUI has no sprite for is refused when the env is built.

The observation is ``pyBaba.Preprocess.StateToTensor``: ``(16, H, W)``
float32 in [0, 1], the feature planes baba-is-auto's RL examples train on.
Actions are ``Discrete(5)``: 0 = wait (a turn in which only MOVE objects act),
1 = up, 2 = down, 3 = left, 4 = right. The reward is the RL examples': +200 on
WON, -100 on LOST, -0.5 for every other turn; ``terminated`` on either end,
never ``truncated``. The seed goes to ``Game.SetRandomSeed`` (directionless
EMPTY only; the maps here are deterministic). No savestate: an episode replays
from its seed and actions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import pyBaba

TILE = 24

_DIRECTIONS = (pyBaba.Direction.NONE, pyBaba.Direction.UP, pyBaba.Direction.DOWN,
               pyBaba.Direction.LEFT, pyBaba.Direction.RIGHT)
#: The GUI turns KEKE's sprite (drawn facing right) to face its direction.
_ROTATION = {pyBaba.Direction.NONE: 0, pyBaba.Direction.RIGHT: 0, pyBaba.Direction.UP: 90,
             pyBaba.Direction.LEFT: 180, pyBaba.Direction.DOWN: -90}
_REWARD = {pyBaba.PlayState.WON: 200.0, pyBaba.PlayState.LOST: -100.0}


#: fmri-gym's ``external/baba_auto``; this file is ``gym/baba_auto/baba_auto_gym/env.py``.
_DEFAULT_REPO = Path(__file__).resolve().parents[3] / "external" / "baba_auto"
_CLONE = ("git clone https://github.com/utilForever/baba-is-auto.git external/baba_auto && "
          "git -C external/baba_auto checkout <commit>  (the README pins the commit)")


def _repo(repo: str | None) -> Path:
    """The baba-is-auto checkout: ``repo``, or fmri-gym's ``external/baba_auto``.

    :raises RuntimeError: nothing at the resolved path.
    """
    path = Path(repo or _DEFAULT_REPO)
    if not (path / "Resources" / "Maps").is_dir():
        raise RuntimeError(f"no baba-is-auto checkout at {path}; {_CLONE}, or pass repo=")
    return path


def map_path(game: str, repo: str | None = None) -> Path:
    """The map file ``game`` names.

    :param game: a map name under the checkout's ``Resources/Maps``
        (``baba_is_you``, ``out_of_reach``, ...) or the path of a map file.
    :param repo: the baba-is-auto checkout; default fmri-gym's ``external/baba_auto``.
    :raises FileNotFoundError: no such map.
    """
    if Path(game).is_file():
        return Path(game)
    maps = _repo(repo) / "Resources" / "Maps"
    path = maps / f"{game}.txt"
    if not path.is_file():
        names = ", ".join(sorted(p.stem for p in maps.glob("*.txt")))
        raise FileNotFoundError(f"no map {game!r} in {maps}; the maps are: {names}")
    return path


def _load_sprites(sprites_dir: Path) -> dict[Any, Any]:
    """``{ObjectType: Surface}`` from the GUI's ``sprites/icon`` and ``sprites/text`` GIFs."""
    import pygame

    sprites = {}
    for kind, prefix in (("icon", "ICON_"), ("text", "")):
        for gif in (sprites_dir / kind).glob("*.gif"):
            sprites[getattr(pyBaba.ObjectType, prefix + gif.stem)] = pygame.image.load(str(gif))
    return sprites


class BabaAutoEnv(gym.Env):
    """One Baba Is Auto map (a Baba Is You level).

    :param game: a map name under the checkout's ``Resources/Maps``, or a
        path to a map file.
    :param repo: the baba-is-auto checkout; default fmri-gym's ``external/baba_auto``.
    :raises ValueError: the map has an object the GUI has no sprite for.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, game: str = "baba_is_you", *, repo: str | None = None) -> None:
        self.game = pyBaba.Game(str(map_path(game, repo)))
        self.sprites_dir = _repo(repo) / "Extensions" / "BabaGUI" / "sprites"
        self.sprites = _load_sprites(self.sprites_dir)
        grid = self.game.GetMap()
        self.width, self.height = grid.GetWidth(), grid.GetHeight()
        self._check_sprites(grid)
        self.action_space = spaces.Discrete(len(_DIRECTIONS))
        self.observation_space = spaces.Box(
            0.0, 1.0, (pyBaba.Preprocess.TENSOR_DIM, self.height, self.width), np.float32)

    def _check_sprites(self, grid: Any) -> None:
        """Refuse a map with an object the picture could not show."""
        types = {t for y in range(self.height) for x in range(self.width)
                 for t in grid.At(x, y).GetTypes()}
        missing = sorted(str(t) for t in types - set(self.sprites) - {pyBaba.ObjectType.ICON_EMPTY})
        if missing:
            raise ValueError(f"the map uses {', '.join(missing)}, which the baba-is-auto GUI "
                             f"has no sprite for under {self.sprites_dir}")

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        """Put the map back as loaded.

        :param seed: seeds the simulator's RNG (directionless EMPTY behaviour).
        :param options: unused.
        :return: ``(obs, info)``.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.game.SetRandomSeed(seed & 0xFFFFFFFF)
        self.game.Reset()
        return self._obs(), self._info()

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Play one turn.

        :param action: 0 = wait, 1 = up, 2 = down, 3 = left, 4 = right.
        :return: ``(obs, reward, terminated, truncated, info)``.
        """
        self.game.MovePlayer(_DIRECTIONS[int(action)])
        state = self.game.GetPlayState()
        return self._obs(), _REWARD.get(state, -0.5), state in _REWARD, False, self._info()

    def _obs(self) -> np.ndarray:
        return np.asarray(pyBaba.Preprocess.StateToTensor(self.game), dtype=np.float32).reshape(
            self.observation_space.shape)

    def _info(self) -> dict:
        return {"play_state": self.game.GetPlayState().name}

    def render(self) -> np.ndarray:
        """The grid as the GUI draws it, ``(24 * H, 24 * W, 3)`` uint8."""
        import pygame

        grid = self.game.GetMap()
        surface = pygame.Surface((TILE * self.width, TILE * self.height))
        for y in range(self.height):
            for x in range(self.width):
                self._draw_cell(surface, grid.At(x, y), x, y)
        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))

    def _draw_cell(self, surface: Any, cell: Any, x: int, y: int) -> None:
        import pygame

        for instance in cell.GetInstances():
            if instance.type == pyBaba.ObjectType.ICON_EMPTY:
                continue
            image = self.sprites[instance.type]
            if instance.type == pyBaba.ObjectType.ICON_KEKE:
                image = pygame.transform.rotate(image, _ROTATION[instance.direction])
            surface.blit(image, (x * TILE, y * TILE))
