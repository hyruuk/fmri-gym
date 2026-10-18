"""VGDL games (the ``dbp`` fork of ccolas/language_and_experience) as a standard Gymnasium env.

The fork's ``VGDLEnv`` is a ``gymnasium.Env`` already, ported from old gym,
but with habits of its own: it is built from game and level FILES, its
``reset`` and ``step`` take a ``with_img`` flag and no seed (seeding is
``game.set_seed``), ``render`` calls ``pygame.display.set_mode`` (opening a
window, or resizing the one the process has to the game's size), and
``close`` quits pygame altogether. This wrapper takes a game name, a level and
a checkout, seeds on ``reset(seed=)``, draws on an offscreen surface, and
never touches ``pygame.display``.

Games live at ``<repo>/games/<game>_v0/<game>.txt`` and ``<game>_lvl<level>.txt``
(``aliens``, ``beesAndBirds``, ``avoidGeorge``, ``jaws``, ``missile_command``,
``plaqueAttack``, ``portals``, ``preconditions``, ``pushBoulders``,
``relational``); ``repo`` defaults to fmri-gym's ``external/vgdl`` and is put on ``sys.path``
(the fork's package is ``src.vgdl``). The observation is the fork's "objects"
observation; ``info`` carries the symbolic per-cell ``state``, the collision
``events_triggered``, ``won`` and ``lose``. Actions are the fork's fixed order:
0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT, 4 = NO_OP, 5 = SPACE.
:meth:`get_state` / :meth:`set_state` are the game's exact savestate.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

#: fmri-gym's ``external/vgdl``; this file is ``gym/vgdl/vgdl_gym/env.py``.
_DEFAULT_REPO = Path(__file__).resolve().parents[3] / "external" / "vgdl"
_CLONE = "git submodule update --init external/vgdl  (a submodule of fmri-gym)"


class VGDLEnv(gym.Wrapper):
    """One VGDL game at one level. Wraps around the fork's ``VGDLEnv`` env, 
    which has a non standard API and undesirably modifies the pygame window.

    :param game: game name, the ``<game>_v0`` directory under ``games/``.
    :param level: level index, ``<game>_lvl<level>.txt``.
    :param repo: the language_and_experience checkout. Defaults to
        ``external/vgdl`` under the fmri-gym root.
    :param block_size: pixels per grid cell in the rendered frame.
    :raises RuntimeError: no checkout at the resolved path.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, game: str, *, level: int = 0, repo: str | None = None,
                 block_size: int = 25) -> None:
        repo = repo or str(_DEFAULT_REPO)
        if not (Path(repo) / "games").is_dir():
            raise RuntimeError(f"no language_and_experience checkout at {repo}; {_CLONE}, "
                               "or pass repo=")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from src.vgdl.interfaces.gym.env import VGDLEnv as ForkEnv
        from src.vgdl.render import PygameRenderer

        files = Path(repo) / "games" / f"{game}_v0"
        env = ForkEnv(game_file=str(files / f"{game}.txt"),
                      level_file=str(files / f"{game}_lvl{level}.txt"),
                      obs_type="objects", block_size=block_size)
        env.renderer = _offscreen(PygameRenderer(env.game, block_size))
        super().__init__(env)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict]:
        """Restart the level.

        :param seed: seeds the game's RNG (sprite spawns, enemy moves).
        :param options: unused.
        :return: ``(obs, info)``.
        """
        if seed is not None:
            self.env.game.set_seed(int(seed))
        _resume_ids(self.env.game.sprite_registry, None)   # a restored game's ids end with it
        return self.env.reset(with_img=False)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Advance one game tick.

        :param action: an index into the fork's six actions.
        :return: ``(obs, reward, terminated, truncated, info)``.
        """
        return self.env.step(int(action), with_img=False)

    def render(self) -> np.ndarray:
        """Draw the board on the offscreen surface and return it, ``(H, W, 3)`` uint8."""
        renderer = self.env.renderer
        renderer.draw_all()
        return renderer.get_image()

    def close(self) -> None:
        """Nothing to close: the offscreen surface is a plain object, and the fork's
        ``close`` would quit pygame for the whole process."""

    def get_state(self) -> Any:
        """The game's exact, picklable state; :meth:`set_state` takes it back.

        The fork's state is its live sprites, and two things the game then
        goes on with are not in it. Its random generator: a restored game
        would draw its spawns and chance effects from wherever the generator
        happened to stand, not from where it stood at this frame. And the id a
        sprite born later gets (``sam.2``), which comes from a count over the
        live *and dead* sprites of its kind, the dead being gone from the
        state. The state carries both, so play from a restored state is the
        recorded game's.
        """
        state = self.env.get_state(return_orientation=True)
        registry = self.env.game.sprite_registry
        state["rng"] = self.env.game.random_generator.getstate()
        state["next_ids"] = {key: int(registry.generate_id_number(key))
                             for key in registry.sprite_keys}
        return state

    def set_state(self, state: Any) -> None:
        """Restore a state from :meth:`get_state`, random generator and later ids included."""
        self.env.set_state(state)
        self.env.game.random_generator.setstate(state["rng"])
        _one_of_each(self.env.game.sprite_registry)
        _resume_ids(self.env.game.sprite_registry, state["next_ids"])


def _one_of_each(registry: Any) -> None:
    """Leave each sprite id one live instance in ``registry``.

    The fork's restore carries the walls of the registry it replaces over
    into the new one and then rebuilds every sprite of the state, walls
    included, so each wall is live twice and a collision with one fires two
    events. The instance kept is the one the registry's id table names.
    """
    for key, sprites in registry._live_sprites_by_key.items():
        ids = dict.fromkeys(s.id for s in sprites)
        sprites[:] = [registry._sprite_by_id[i] for i in ids]


def _resume_ids(registry: Any, next_ids: dict[str, int] | None) -> None:
    """Make ``registry`` number new sprites from ``next_ids`` on, never below; ``None`` lifts it.

    The registry a restore builds is fresh, so the fork's own count (over the
    sprites it holds) is what it would start from; this keeps whichever is
    higher, which is the recorded game's count once its dead are gone. The
    floor is the instance's own attribute, so lifting it is dropping that and
    the class's method shows again; a reset lifts it, since a new episode
    starts its count from one.
    """
    registry.__dict__.pop("generate_id_number", None)
    if next_ids is None:
        return
    own = registry.generate_id_number
    registry.generate_id_number = lambda key: max(own(key), next_ids[key])


def _offscreen(renderer: Any) -> Any:
    """Give the fork's renderer a plain Surface to draw on instead of a window.

    ``PygameRenderer.init_screen`` would call ``pygame.display.set_mode``; this
    sets the same attributes by hand, so ``draw_all`` and ``get_image`` work
    and a window the process already has is never touched. The fork's sprite
    cache converts each sprite to the display's pixel format
    (``Surface.convert_alpha``), which pygame refuses without a display
    surface, so a process with none -- a model playing, a test -- gets a
    hidden 1x1 window to provide that format.
    """
    import pygame

    if pygame.display.get_surface() is None:
        pygame.display.init()
        pygame.display.set_mode((1, 1), pygame.HIDDEN)
    renderer.headless = True
    renderer.screen = pygame.Surface(renderer.screen_dims)
    renderer.screen.fill((255, 255, 255))
    renderer.background = renderer.screen.copy()
    return renderer
