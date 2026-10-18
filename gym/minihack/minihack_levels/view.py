"""``pixel_crop`` as a window fixed on the level's map, for any MiniHack env.

MiniHack's own ``pixel_crop`` is a window centred on the agent, so on screen
the agent stands still and the dungeon slides under it. On a small fixed
level the natural picture is the opposite: the level holds still and the
agent moves. The whole-terminal ``pixel`` observation would give that, but a
5x5 room is a few percent of its 21x79 tiles, so :class:`MapView` cuts
``pixel`` down to the level's map and hands that back as ``pixel_crop``, the
same size every step.

Where the map is comes from the des-file MiniHack compiled for the env: the
``MAP`` block gives its size, and NetHack's ``sp_lev.c`` places a
``GEOMETRY:center,center`` map at a fixed, computable spot (:func:`map_origin`).
MiniHack keeps the des-file it was given as a string at ``mylevel.des`` in the
env's NetHack directory; that is where it is read from. An env built from a
``.des`` file on disk leaves no such copy, and so does one whose des-file has
no ``MAP`` (``RANDOM_CORRIDORS`` levels): both get the whole terminal, which
is the only fixed view there is for a level that can be anywhere.
"""

from __future__ import annotations

import os
import re
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from minihack.base import N_TILE_PIXEL
from nle import nethack

#: glyph rows and columns of the NetHack map NLE exposes (``nethack.ROWNO``
#: rows; the terminal is 80 columns but NLE drops column 0).
MAP_SHAPE = tuple(nethack.OBSERVATION_DESC["glyphs"]["shape"])

# sp_lev.c's x_maze_max and y_maze_max: (COLNO - 1) & ~1 and (ROWNO - 1) & ~1.
_X_MAZE_MAX = (nethack.COLNO - 1) & ~1
_Y_MAZE_MAX = (nethack.ROWNO - 1) & ~1


def map_origin(height: int, width: int) -> tuple[int, int]:
    """Where NetHack puts a ``GEOMETRY:center,center`` map of this size.

    Follows ``sp_lev.c``: the start is ``2 + (maze_max - 2 - size) / 2`` on
    each axis, then bumped to odd, which is the step a plain "centre it"
    formula misses. The column comes back in glyph coordinates, where NetHack's
    ``x = 1`` is column 0.

    :param height: the map's rows.
    :param width: the map's columns (its longest line).
    :return: ``(row, col)`` of the map's top-left tile in ``glyphs``.
    """
    x = 2 + (_X_MAZE_MAX - 2 - width) // 2
    y = 2 + (_Y_MAZE_MAX - 2 - height) // 2
    x += x % 2 == 0
    y += y % 2 == 0
    return y, x - 1


def map_extent(des: str | None) -> tuple[int, int, int, int]:
    """The ``(row, col, height, width)`` in ``glyphs`` of a des-file's map.

    :param des: the des-file's text, or ``None`` for an env without one.
    :return: the map's rectangle; the whole map when there is no ``MAP`` block.
    :raises ValueError: the map is placed with a geometry other than
        ``center,center``, the only one :func:`map_origin` knows.
    """
    block = re.search(r"^MAP\n(.*?)^ENDMAP", des or "", re.S | re.M)
    if block is None:
        return 0, 0, MAP_SHAPE[0], MAP_SHAPE[1]
    geometry = re.search(r"^GEOMETRY:(\S+)", des, re.M)
    if geometry is None or geometry.group(1) != "center,center":
        raise ValueError(f"MapView knows where NetHack puts a GEOMETRY:center,center map, "
                         f"not {geometry.group(1) if geometry else 'an unplaced'!r} one")
    rows = block.group(1).splitlines()
    height, width = len(rows), max(len(r) for r in rows)
    row, col = map_origin(height, width)
    return row, col, height, width


class MapView(gym.ObservationWrapper):
    """``pixel_crop`` is the level's map cut from ``pixel``, fixed across the episode.

    The env must observe ``pixel``; ``pixel_crop`` in the result is
    ``(16 * height, 16 * width, 3)`` for the map's size in tiles. Every other
    observation passes through untouched.

    :param env: a MiniHack env observing ``pixel``.
    :raises ValueError: the env does not observe ``pixel``, or its map is not
        centred (see :func:`map_extent`).
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if "pixel" not in env.observation_space.spaces:
            raise ValueError("MapView cuts pixel_crop from pixel: add 'pixel' to observation_keys")
        row, col, height, width = map_extent(self._des_text(env))
        self._rows = slice(row * N_TILE_PIXEL, (row + height) * N_TILE_PIXEL)
        self._cols = slice(col * N_TILE_PIXEL, (col + width) * N_TILE_PIXEL)
        self.observation_space = spaces.Dict({
            **env.observation_space.spaces,
            "pixel_crop": spaces.Box(0, 255, (height * N_TILE_PIXEL, width * N_TILE_PIXEL, 3),
                                     np.uint8),
        })

    @staticmethod
    def _des_text(env: gym.Env) -> str | None:
        # MiniHack writes a des-file given as a string to mylevel.des in the
        # env's own NetHack directory before compiling it; an env built from a
        # .des on disk never writes one.
        path = os.path.join(env.unwrapped.nethack._vardir, "mylevel.des")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return f.read()

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Return ``observation`` with ``pixel_crop`` replaced by the map's pixels.

        :param observation: the wrapped env's observation dict.
        :return: a new dict; ``pixel_crop`` is a copy, the rest the same arrays.
        """
        return {**observation, "pixel_crop": observation["pixel"][self._rows, self._cols].copy()}
