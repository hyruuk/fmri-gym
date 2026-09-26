"""AI GameStore games as a Gymnasium environment.

AI GameStore (https://aigamestore.org) is a benchmark of ten LLM-generated
browser games -- plain HTML + JavaScript, nine on p5.js and one on three.js --
built to compare vision-language models with human players. The paper's own
harness drives the page in real time with Playwright: it pauses the game while
the model thinks, then holds keys for 0.2 s segments on the wall clock. That
has no ``step()`` and no seed, so it cannot be trained against or replayed.

This env keeps the browser (the games are the browser) but takes over its
clock. ``lockstep.js`` runs in the page before the game and replaces
``requestAnimationFrame``, ``setTimeout``, ``performance.now``, ``Date`` and
``Math.random``; one :meth:`step` then dispatches key events, advances the game
by exactly ``frame_skip`` of its 60 Hz frames and returns the canvas pixels and
the game's own ``getGameState()``. Time does not pass between steps, so a model
that deliberates for a minute and a human at the keyboard meet the same game,
and ``reset(seed=...)`` plus the action sequence replays an episode exactly.

Observation: the game canvas as an RGB ``Box`` (each game has its own size,
e.g. 600x400). Action: ``MultiBinary(len(keys))`` -- which of the game's keys are
held this step; see :data:`GAME_KEYS`. Reward: the change in the game's
``score``. An episode ends when the game returns to its START screen (or
reports ``ENDED``); level clears and game overs do not end it, they move on by
themselves after 3 s. ``info["state"]`` is the scalar part of the game state
(``gamePhase``, ``score``, ``currentLevel``, ...).

Not supported: sound (the games have none) and savestates (replay from seed).
Needs a Chromium-based browser (the system Chrome by default; pass
``browser_channel=None`` for Playwright's bundled Chromium after
``playwright install chromium``). The games load p5 / three.js from a CDN, so
the first run needs network access.
"""

from __future__ import annotations

import base64
import contextlib
import functools
import http.server
import io
import socketserver
import threading
import urllib.parse
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

_HERE = Path(__file__).resolve().parent
_GAMES_DIR = _HERE.parent          # game1/ .. game10/ sit beside this package
_LOCKSTEP_JS = _HERE / "lockstep.js"

#: The keys each vendored game listens for during play (its ``input.js``);
#: this is the env's action vocabulary. Enter / Escape / R start, pause and
#: restart and are the env's business, not the agent's.
GAME_KEYS: dict[str, list[str]] = {
    "game1": ["LEFT", "RIGHT", "SPACE", "Z"],
    "game2": ["LEFT", "RIGHT", "UP", "DOWN", "SPACE", "Z"],
    "game3": ["W", "A", "S", "D", "LEFT", "RIGHT", "UP", "DOWN", "SPACE", "Z", "LSHIFT"],
    "game4": ["SPACE", "UP"],
    "game5": ["LEFT", "RIGHT", "UP"],
    "game6": ["LEFT", "RIGHT", "UP", "SPACE", "Z"],
    "game7": ["LEFT", "RIGHT", "UP", "DOWN", "Z", "SPACE", "LSHIFT", "1", "2", "3", "4", "5"],
    "game8": ["LEFT", "RIGHT", "SPACE", "Z"],
    "game9": ["LEFT", "RIGHT", "UP", "DOWN", "Z", "SPACE", "LSHIFT"],
    "game10": ["LEFT", "RIGHT", "UP", "DOWN", "LSHIFT", "SPACE", "Z"],
}

# Headless Chrome falls back to SwiftShader (software GL) unless told to use
# the GPU; the three.js game then takes ~130 ms a frame instead of ~2 ms.
_BROWSER_ARGS = ["--use-gl=angle", "--use-angle=gl", "--ignore-gpu-blocklist"]
_BOOT_FRAMES = 600                 # 10 s of game time for p5 to set up


class AIGameStoreEnv(gym.Env):
    """One AI GameStore game in a lock-stepped browser page.

    :param game: ``"game1"`` .. ``"game10"`` (served from ``games_dir``), or the
        ``http(s)://`` URL of a game's ``index.html``.
    :param keys: the game's keys, in action-vector order. Defaults to
        :data:`GAME_KEYS`; required for a game given by URL.
    :param frame_skip: game frames (at 60 Hz) per :meth:`step`.
    :param headless: run the browser without a window.
    :param browser_channel: Playwright browser channel (``"chrome"``), or
        ``None`` for the bundled Chromium.
    :param key_labels: ``{game key: label}`` for the on-canvas control hints, for
        a host whose player presses something other than the named key.
    :param games_dir: directory holding ``game1/`` .. ``game10/``.
    :param start_key: tapped on reset to leave the START screen.
    :param render_mode: only ``"rgb_array"``.
    """

    metadata: ClassVar[dict] = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        game: str = "game1",
        *,
        keys: list[str] | None = None,
        frame_skip: int = 6,
        headless: bool = True,
        browser_channel: str | None = "chrome",
        key_labels: dict[str, str] | None = None,
        games_dir: str | Path | None = None,
        start_key: str = "RETURN",
        render_mode: str | None = "rgb_array",
    ) -> None:
        from playwright.sync_api import sync_playwright

        if keys is None and game not in GAME_KEYS:
            raise ValueError(f"{game!r} is not one of {sorted(GAME_KEYS)}; pass keys=[...] for it")
        self.game = game
        self.keys = list(GAME_KEYS[game] if keys is None else keys)
        self.frame_skip = int(frame_skip)
        self.start_key = start_key
        self.render_mode = render_mode
        self.metadata = {**self.metadata, "render_fps": 60 / self.frame_skip}

        self._server, self._url = None, game
        if not game.startswith(("http://", "https://")):
            self._server, base = _serve(Path(games_dir or _GAMES_DIR))
            self._url = f"{base}/{game}/index.html"
        labels = {f"label_{k}": v for k, v in (key_labels or {}).items()}
        self._url += ("&" if "?" in self._url else "?") + urllib.parse.urlencode(labels)

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=headless, args=_BROWSER_ARGS,
            **({"channel": browser_channel} if browser_channel else {}))
        self._page = self._browser.new_page(viewport={"width": 900, "height": 700})
        self._page.add_init_script(path=str(_LOCKSTEP_JS))

        self._frame: np.ndarray | None = None
        self._score = 0.0
        self._phase = ""
        self._load(seed=0)
        unknown = set(self.keys) - set(self._page.evaluate("() => window.__aigs.keys"))
        if unknown:
            self.close()
            raise ValueError(f"keys {sorted(unknown)} are not in lockstep.js's key table")
        frame, _ = self._step([], 0)
        self.observation_space = spaces.Box(0, 255, frame.shape, np.uint8)
        self.action_space = spaces.MultiBinary(len(self.keys))

    # ------------------------------------------------------------------ gym
    def reset(self, *, seed: int | None = None, options: dict | None = None
              ) -> tuple[np.ndarray, dict]:
        """Reload the game with ``seed`` and leave its START screen.

        :param seed: seeds the page's ``Math.random``; ``None`` draws one.
        :param options: unused.
        :return: ``(frame, {"state": ...})`` -- the first PLAYING frame.
        """
        super().reset(seed=seed)
        self._load(seed=int(self.np_random.integers(1, 2**31)) if seed is None else seed)
        frame, state = self._step([self.start_key], 1)
        self._score = float(state.get("score", 0.0))
        self._phase = str(state.get("gamePhase", ""))
        return frame, {"state": state}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Hold the keys ``action`` marks and advance ``frame_skip`` frames.

        :param action: 0/1 per entry of :attr:`keys`.
        :return: ``(frame, score delta, terminated, False, {"state": ...})``.
        """
        names = [k for k, on in zip(self.keys, action) if on]
        frame, state = self._step(names, self.frame_skip)
        score = float(state.get("score", self._score))
        phase = str(state.get("gamePhase", self._phase))
        reward, self._score = score - self._score, score
        terminated = (phase == "START" and self._phase != "START") or phase == "ENDED"
        self._phase = phase
        return frame, reward, terminated, False, {"state": state}

    def render(self) -> np.ndarray:
        """Return the canvas as it stood after the last step, RGB ``(H, W, 3)``."""
        return self._frame

    def close(self) -> None:
        """Tear down the browser, Playwright, and the file server.

        Each is attempted even if an earlier one raises, so a wedged browser
        cannot leak the server thread and its port.
        """
        closers = [self._browser.close, self._playwright.stop]
        if self._server:
            closers.append(self._server.shutdown)
        for shutdown in closers:
            with contextlib.suppress(Exception):
                shutdown()

    # -------------------------------------------------------------- helpers
    def _load(self, seed: int) -> None:
        """Navigate to the game with ``seed`` and tick it up to its START screen."""
        self._page.goto(f"{self._url}&seed={seed}")
        self._page.evaluate("n => window.__aigs.boot(n)", _BOOT_FRAMES)

    def _step(self, names: list[str], frames: int) -> tuple[np.ndarray, dict]:
        """Hold exactly ``names``, tick ``frames``, and read canvas + state."""
        out = self._page.evaluate("([n, f]) => window.__aigs.step(n, f)", [names, frames])
        self._frame = _decode_png(out["png"])
        return self._frame, out["state"]


def _decode_png(data_url: str) -> np.ndarray:
    """RGB array from a ``data:image/png;base64,...`` URL."""
    from PIL import Image

    png = base64.b64decode(data_url.split(",", 1)[1])
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """``SimpleHTTPRequestHandler`` without the per-request stderr line."""

    def log_message(self, *args: Any) -> None:
        """Drop the request log."""


def _serve(directory: Path) -> tuple[socketserver.TCPServer, str]:
    """Serve ``directory`` over HTTP on a free local port, on a daemon thread.

    The games are ES modules, which browsers refuse to load over ``file://``.

    :return: ``(server, base_url)``; the caller shuts the server down.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"games directory {directory} does not exist")
    handler = functools.partial(_QuietHandler, directory=str(directory))
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"
