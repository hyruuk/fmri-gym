"""The pad's two tools: the calibration, and the scope it runs in.

``fmri-pad-scope`` shows what a controller sends and what :mod:`fmri_gym.pad`
makes of it, and its buttons calibrate and tune it; ``fmri-pad-calibrate`` opens
it straight into a calibration. The rig check runs the same calibration in the
window it has open (:func:`calibrate`). Every step of a calibration waits for the
controller; it measures what a run needs (each stick's rest, travel and how far
its centre moves) and confirms that each control is the one it is named.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import time
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pygame

from . import pad as pads
from .pad import (CONFIG_ENV, DPAD, RIGHT_AXES, RIGHT_KEY, RIGHT_STICK, STICK,
                  STICK_AXES, STICKS, Pad, Tuning)


# ------------------------------------------------------------ the two tools
# Shared by the calibration task and the scope, which are the same program seen
# before and after: one measures the pad, the other watches what it measured.
SURFACE, PANEL, GRID = (18, 21, 28), (24, 28, 37), (35, 41, 54)
INK, INK2, INK3 = (231, 234, 240), (154, 164, 178), (91, 100, 114)
X_HUE, Y_HUE = (79, 143, 247), (240, 136, 62)
LIVE, WARN = (63, 185, 80), (210, 153, 34)
#: How each control is named and coloured on screen: the face buttons as they
#: are printed on the pad, in the colour they are printed in, so a cue can be
#: matched by colour without reading it.
FACE: dict[str, tuple[int, int, int]] = {
    "Y": (226, 192, 68), "B": (230, 86, 86), "X": (79, 143, 247), "A": (63, 185, 80),
}


def _arrow(surface, direction: str, at: tuple[int, int], size: int, colour) -> None:
    """Draw a direction as the arrow it is, rather than spelling it out."""
    dx, dy = {"LEFT": (-1, 0), "RIGHT": (1, 0), "UP": (0, -1), "DOWN": (0, 1)}[direction]
    px, py = -dy, dx                                    # across the shaft
    cx, cy = at
    tip = (cx + dx * size, cy + dy * size)
    base = (cx - dx * size * 0.4, cy - dy * size * 0.4)
    neck = (cx + dx * size * 0.15, cy + dy * size * 0.15)
    pygame.draw.polygon(surface, colour, [
        tip,
        (neck[0] + px * size * 0.55, neck[1] + py * size * 0.55),
        (neck[0] + px * size * 0.2, neck[1] + py * size * 0.2),
        (base[0] + px * size * 0.2, base[1] + py * size * 0.2),
        (base[0] - px * size * 0.2, base[1] - py * size * 0.2),
        (neck[0] - px * size * 0.2, neck[1] - py * size * 0.2),
        (neck[0] - px * size * 0.55, neck[1] - py * size * 0.55),
    ])


#: Each stick direction as (its axis, the sign a push gives it).
DIRECTIONS = {"LEFT": (0, -1), "RIGHT": (0, 1), "UP": (1, -1), "DOWN": (1, 1),
              "RS_LEFT": (2, -1), "RS_RIGHT": (2, 1), "RS_UP": (3, -1), "RS_DOWN": (3, 1)}
#: The order the calibration asks for the buttons: round the face the way they
#: sit under a thumb, then the rest; and for each stick's directions.
CUE_BUTTONS = ("A", "X", "Y", "B", "LT", "RT", "LB", "RB", "START", "BACK", *DPAD)
CUE_DIRECTIONS = ("DOWN", "LEFT", "UP", "RIGHT")
#: Each control is asked for this many times in a row, so one slip is caught.
TWICE = 2
#: A stick is at rest once each axis has stayed within this, raw, for this long.
STILL_S, STILL_RANGE = 1.0, 0.02
#: A roll is done when each of this many directions round the edge has been
#: pushed this far, as a fraction of the travel seen that way, with no axis
#: reaching further since; and each side has travelled at least this far, raw.
ROLL_BINS, ROLL_REACH, ROLL_MIN = 12, 0.9, 0.3
#: A cued push counts past this fraction of its travel; the next waits for the stick under the second.
PUSH, RELEASE = 0.5, 0.2
#: The thresholds a calibration writes, in fractions of travel; raised above ten times the dither.
CAL_ON, CAL_OFF = 0.5, 0.3


class _Quit(Exception):
    """ESC, Cancel, or the window closed: the calibration stops, and writes nothing."""


class _Task:
    """The calibration's screen and its recording: one row per frame.

    It draws on ``screen`` and shows it with ``present``. Given ``to_canvas`` (a
    window position -> ``screen``'s), it draws a Cancel button that clicks stop it.
    """

    def __init__(self, pad: Pad, screen, present=None, to_canvas=None) -> None:
        self.pad, self.screen = pad, screen
        self.present, self.to_canvas = present or pygame.display.flip, to_canvas
        w, self.h = screen.get_size()
        self.mx, scale = w // 2, min(2.0, max(0.8, self.h / 600))
        font = "Menlo,Monaco,monospace"
        self.big, self.mid, self.sml = (pygame.font.SysFont(font, int(n * scale))
                                        for n in (62, 25, 15))
        self.cancel = None if to_canvas is None else pygame.Rect(w - 150, self.h - 60, 130, 40)
        self.clock = pygame.time.Clock()
        self.buttons = [c for c in CUE_BUTTONS if c in pad.controls()]
        self.samples: list[dict] = []
        self.t0 = time.perf_counter()

    def frame(self, phase: str, cue: str) -> tuple[dict[int, float], set[str]]:
        """Wait for the next frame and record the pad's reading.

        :raises _Quit: on ESC, a click on Cancel, or the window closing (which is
            left on the queue for the window's owner, when it has a Cancel).
        """
        self.clock.tick(60)
        for e in pygame.event.get((pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN)):
            if e.type == pygame.QUIT:
                if self.cancel:
                    pygame.event.post(pygame.event.Event(pygame.QUIT))
                raise _Quit
            if e.type == pygame.KEYDOWN and not getattr(e, "pad", False) \
                    and e.key == pygame.K_ESCAPE:
                raise _Quit
            if e.type == pygame.MOUSEBUTTONDOWN and self.cancel \
                    and self.cancel.collidepoint(self.to_canvas(e.pos)):
                raise _Quit
        pygame.event.clear(pads.PAD_EVENTS)         # read by polling here, not queued for a run
        ax, bt = self.pad.read()
        row = {"t": round(time.perf_counter() - self.t0, 4), "phase": phase, "cue": cue}
        row.update({f"ax{i}": round(v, 5) for i, v in sorted(ax.items())})
        row.update({f"b_{c}": int(c in bt) for c in self.buttons})
        self.samples.append(row)
        return ax, bt

    def draw(self, lines: list[str], colour=INK, sub: str = "", arrow: str | None = None,
             ring: tuple[int, set[int]] | None = None) -> None:
        """One screen: a word or two, an arrow or the roll's ring of directions, a line under."""
        self.screen.fill(SURFACE)
        if arrow:
            _arrow(self.screen, arrow, (self.mx, 235), 74, colour)
        if ring:
            n, lit = ring
            for b in range(n):
                angle = (b + 0.5) / n * 2 * math.pi - math.pi
                at = (self.mx + math.cos(angle) * 110, 235 + math.sin(angle) * 110)
                pygame.draw.circle(self.screen, LIVE if b in lit else GRID, at, 12)
        top = 360 if ring else 300 if arrow else 235
        for k, line in enumerate(lines):
            surf = (self.big if k == 0 else self.mid).render(line, True, colour if k == 0 else INK2)
            self.screen.blit(surf, surf.get_rect(center=(self.mx, top + k * 66)))
        s = self.sml.render(sub or ("Cancel or ESC stops; nothing is written" if self.cancel
                                    else "ESC stops; nothing is written"), True, INK2)
        self.screen.blit(s, s.get_rect(center=(self.mx, self.h - 40)))
        if self.cancel:
            _draw_button(self.screen, self.sml, self.cancel, "Cancel", hover=False)
        self.present()


def _hold_still(task: _Task, stick: tuple[int, int], phase: str, cue: str,
                words: list[str]) -> dict[int, float]:
    """Wait until the stick has been still for :data:`STILL_S`. Returns where it rests."""
    window: deque = deque()
    while True:
        task.draw(words, INK2)
        ax, _ = task.frame(phase, cue)
        now = time.perf_counter()
        window.append((now, ax))
        while now - window[0][0] > STILL_S:
            window.popleft()
        if now - window[0][0] < 0.95 * STILL_S:
            continue
        spans = [max(a[i] for _, a in window) - min(a[i] for _, a in window) for i in stick]
        if max(spans) <= STILL_RANGE:
            return {i: statistics.median(a[i] for _, a in window) for i in stick}


def _roll(task: _Task, stick: tuple[int, int], cue: str, rest: dict[int, float],
          which: str) -> dict[int, tuple[float, float]]:
    """Wait until the stick has been rolled to its stop all the way round. Returns each axis's (min, max)."""
    lo, hi, lit = dict(rest), dict(rest), set()
    while True:
        task.draw([f"roll {which}", "round its edge"], ring=(ROLL_BINS, lit),
                  sub="push it to its stop, all the way round, until every dot is lit")
        ax, _ = task.frame("roll", cue)
        for i in stick:
            if ax[i] < lo[i] - 0.01 or ax[i] > hi[i] + 0.01:
                lit.clear()                     # further than before: round again at it
            lo[i], hi[i] = min(lo[i], ax[i]), max(hi[i], ax[i])
        n = [_fraction(ax[i], rest[i], lo[i], hi[i]) for i in stick]
        if math.hypot(*n) >= ROLL_REACH:
            lit.add(int((math.atan2(n[1], n[0]) + math.pi) / (2 * math.pi) * ROLL_BINS) % ROLL_BINS)
        if len(lit) == ROLL_BINS and all(min(hi[i] - rest[i], rest[i] - lo[i]) >= ROLL_MIN
                                         for i in stick):
            return {i: (lo[i], hi[i]) for i in stick}


def _fraction(value: float, rest: float, lo: float, hi: float) -> float:
    """A reading as a fraction of the travel seen on its side of rest, -1..1."""
    span = (hi - rest) if value >= rest else (rest - lo)
    return 0.0 if span <= 1e-6 else max(-1.0, min(1.0, (value - rest) / span))


def _ask(task: _Task, cue: str, words: str, colour, arrow: str | None, answer, idle) -> int:
    """Ask for ``cue`` until it comes :data:`TWICE` in a row. Returns the wrong answers.

    :param answer: the reading -> the control it answers, or None.
    :param idle: the reading -> whether the pad is let go of, which a next answer waits for.
    """
    wrong, row, said = 0, 0, ""
    while row < TWICE:
        task.draw([words, f"{row + 1} of {TWICE}"], colour, said, arrow)
        while not idle(*task.frame("idle", cue)):
            pass
        got = None
        while got is None:
            task.draw([words, f"{row + 1} of {TWICE}"], colour, said, arrow)
            got = answer(*task.frame("ask", cue))
        if got == cue:
            row, said = row + 1, ""
        else:
            wrong, row, said = wrong + 1, 0, f"that was {got}: again, {TWICE} in a row"
    return wrong


def _calibrate_stick(task: _Task, stick: tuple[int, int], which: str) -> int:
    """Rest, travel and directions of one stick. Returns the wrong answers."""
    cue = "left" if stick == STICK_AXES else "right"
    cold = _hold_still(task, stick, "cold", cue, ["hands off"])
    travel = _roll(task, stick, cue, cold, which)
    rest = _hold_still(task, stick, "rest", cue, ["let go", "hands off"])

    def pushes(ax: dict[int, float]) -> dict[str, float]:
        return {d: _fraction(ax[a], rest[a], *travel[a]) * s
                for d, (a, s) in DIRECTIONS.items() if a in stick}

    def answer(ax: dict[int, float], _bt: set[str]) -> str | None:
        best = max(pushes(ax).items(), key=lambda kv: kv[1])
        return best[0] if best[1] >= PUSH else None

    def idle(ax: dict[int, float], _bt: set[str]) -> bool:
        return all(abs(v) < RELEASE for v in pushes(ax).values())

    names = dict(zip(STICK, STICKS[stick]))
    return sum(_ask(task, names[d], f"push {which}", INK, d, answer, idle)
               for d in CUE_DIRECTIONS)


def _ask_buttons(task: _Task) -> int:
    """Each button the pad has, asked for twice in a row. Returns the wrong answers."""
    return sum(_ask(task, name, f"press  {name}", FACE.get(name, INK), None,
                    lambda _ax, bt: min(bt) if bt else None, lambda _ax, bt: not bt)
               for name in task.buttons)


def pad_sticks(pad: Pad) -> list[tuple[int, int]]:
    """The sticks a pad has."""
    return [s for s, directions in STICKS.items() if directions[0] in pad.controls()]


def run(screen=None, sticks: list[tuple[int, int]] | None = None, buttons: bool = True,
        present=None, to_canvas=None) -> tuple[list[dict], str, int, int]:
    """Measure the pad's sticks and confirm its controls, waiting on each step.

    :param screen: draw on this surface instead of opening a window, so the rig
        check and the scope can run the calibration in the window they have open.
    :param sticks: the sticks to measure; None for every stick the pad has.
    :param buttons: whether to confirm each button too.
    :param present: shows ``screen`` (default: ``pygame.display.flip``).
    :param to_canvas: a window position -> ``screen``'s; given, the screen has a Cancel button.
    :return: (every frame recorded, the pad's name, controls confirmed, wrong answers).
    :raises SystemExit: with no controller plugged in.
    :raises _Quit: on ESC or Cancel.
    """
    own = screen is None
    if own:
        pygame.init()
        screen = pygame.display.set_mode((900, 600))
        pygame.display.set_caption("fmri-gym pad calibration")
    try:
        pad = pads.find()
        if pad is None:
            raise SystemExit("no controller found: plug it in and run this again")
        task, has = _Task(pad, screen, present, to_canvas), pad_sticks(pad)
        wrong, confirmed = 0, 0
        for stick in (has if sticks is None else [s for s in sticks if s in has]):
            which = ("the left stick" if stick == STICK_AXES else "the right stick") \
                if len(has) > 1 else "the stick"
            wrong += _calibrate_stick(task, stick, which)
            confirmed += len(CUE_DIRECTIONS)
        if buttons:
            wrong += _ask_buttons(task)
            confirmed += len(task.buttons)
        task.draw(["done"], LIVE)
        return task.samples, pad.name, confirmed, wrong
    finally:
        if own:
            pygame.quit()


def _still_at(rows: list[dict], i: int) -> tuple[float, float]:
    """Where axis ``i`` rested over the last :data:`STILL_S` of ``rows``, and how much it dithered."""
    if not rows:
        return 0.0, 0.0
    vals = [r[f"ax{i}"] for r in rows if r["t"] >= rows[-1]["t"] - STILL_S]
    return statistics.median(vals), (max(vals) - min(vals)) / 2


def fit(samples: list[dict]) -> dict:
    """Turn a calibration's recording into its file: each measured stick's rest, travel and thresholds.

    :param samples: every frame :func:`run` recorded (raw readings).
    :return: the calibration, ready to write; ``"device"`` is the caller's to fill in.
    """
    blob: dict = {"_comment": "written by fmri-pad-calibrate; rerun it if the controller changes",
                  "device": "",
                  "measured": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "axes": {}}
    for stick, cue in ((STICK_AXES, "left"), (RIGHT_AXES, "right")):
        rows = [s for s in samples if s["cue"] == cue]
        if not rows:
            continue
        dither = 0.0
        for i in stick:
            cold, _ = _still_at([r for r in rows if r["phase"] == "cold"], i)
            rest, noise = _still_at([r for r in rows if r["phase"] == "rest"], i)
            col = [r[f"ax{i}"] for r in rows]
            lo, hi = min(col), max(col)
            blob["axes"][str(i)] = {"rest": round(rest, 5), "min": round(lo, 5),
                                    "max": round(hi, 5), "noise": round(noise, 6),
                                    "drift": round(abs(rest - cold), 5)}
            dither = max(dither, noise / max(min(hi - rest, rest - lo), 1e-3))
        on = round(min(max(CAL_ON, 10 * dither), 0.65), 3)
        settings = replace(Tuning(), on=on, off=CAL_OFF if on == CAL_ON else round(0.6 * on, 3))
        if stick == STICK_AXES:
            blob.update(settings.record())
        else:
            blob[RIGHT_KEY] = settings.record()
    return blob


#: A calibration's top-level keys that are the left stick's settings.
_LEFT_KEYS = ("axis_on", "axis_off", "axis_diagonal", "stick_dwell_s", "sectors")


def _merged(new: dict, base: dict | None, measured: list[tuple[int, int]]) -> dict:
    """A calibration of the sticks measured, keeping the others' from ``base`` (the one in use)."""
    if base is None:
        return new
    kept = [i for s in STICKS if s not in measured for i in s]
    out = {**new, "axes": {**{k: v for k, v in (base.get("axes") or {}).items()
                              if int(k) in kept}, **new["axes"]}}
    if STICK_AXES not in measured:
        out.update({k: base[k] for k in _LEFT_KEYS if k in base})
    if RIGHT_AXES not in measured and RIGHT_KEY in base:
        out[RIGHT_KEY] = base[RIGHT_KEY]
    return out


def _slug(device: str) -> str:
    """A device's name as a file name's start."""
    return re.sub(r"[^A-Za-z0-9]+", "-", device).strip("-") or "pad"


def save(blob: dict, samples: list[dict]) -> Path:
    """Write a calibration and the recording it came from, side by side. Returns the calibration's path.

    The calibration is this site's, not this session's: every run on the rig
    reads it, so it lives beside the other per-machine config rather than in the
    checkout or the session's data, one file per calibration so none is lost.
    """
    override = os.environ.get(CONFIG_ENV)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = (Path(override).expanduser() if override
           else pads.pads_dir() / f"{_slug(blob['device'])}_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2) + "\n")
    with out.with_suffix(".csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(samples[0]))
        w.writeheader()
        w.writerows(samples)
    return out


def calibrate(screen=None, extra: dict | None = None, sticks: list[tuple[int, int]] | None = None,
              buttons: bool = True, present=None, to_canvas=None) -> dict:
    """Calibrate the pad and write its calibration file. Returns what was measured.

    :param screen: draw here (the rig check's window, the scope's) rather than in a window of its own.
    :param extra: merged into the file (the rig check records which site and rig measured it).
    :param sticks: the sticks to measure, None for all; a stick not measured
        keeps its numbers from the calibration in use, in the new file.
    :param buttons, present, to_canvas: as :func:`run` takes them.
    """
    base = None if pads.calibration_path() is None else pads.read_calibration(pads.calibration_path())
    try:
        samples, device, confirmed, wrong = run(screen, sticks, buttons, present, to_canvas)
    except _Quit:
        return {"measured": False, "why": "stopped before it finished; nothing written"}
    new = fit(samples)
    measured = [s for s, cue in ((STICK_AXES, "left"), (RIGHT_AXES, "right"))
                if any(r["cue"] == cue for r in samples)]
    blob = {**_merged(new, base, measured), "device": device, **(extra or {})}
    path = save(blob, samples)
    pads.load_calibration()                      # what follows plays calibrated
    return {"measured": True, "device": device, "config": str(path),
            "samples": str(path.with_suffix(".csv")), "confirmed": confirmed,
            "corrections": wrong, "axis_on": blob.get("axis_on", Tuning().on),
            "axis_off": blob.get("axis_off", Tuning().off),
            "stick_dwell_s": blob.get("stick_dwell_s", Tuning().dwell), "axes": blob["axes"],
            **({RIGHT_KEY: blob[RIGHT_KEY]} if RIGHT_KEY in blob else {})}


def check_buttons(screen, present=None, to_canvas=None) -> dict:
    """Ask for each button twice, writing nothing. Returns how it went."""
    try:
        _, _, confirmed, wrong = run(screen, [], True, present, to_canvas)
    except _Quit:
        return {"done": False}
    return {"done": True, "confirmed": confirmed, "corrections": wrong}


def calibrate_main() -> None:
    """``fmri-pad-calibrate``: the scope, opened straight into a calibration of everything."""
    _scope_from_args("Calibrate this rig's controller, in the scope: each stick's rest and "
                     "travel, and a check that each control is the one it is named. Every step "
                     "waits for the controller.", start="all")


#: The scope is drawn at this size and shown scaled to its window, which can be any size from the second.
W, H = 1020, 880
MIN_W, MIN_H = 640, 552
WINDOW_S, TRAIL_S = 8.0, 0.8
#: The arrow each stick direction is drawn as.
_ARROW_OF = {**{d: d for d in STICK}, **dict(zip(RIGHT_STICK, STICK))}
#: The scope's buttons, (label, action), a row each; the tuning row also shows each value.
_ACTIONS = (("Calibrate all", "all"), ("Calibrate left stick", "left"),
            ("Calibrate right stick", "right"), ("Check buttons", "buttons"),
            ("Save", "save"), ("Reload", "reload"), ("Clear", "clear"), ("Quit", "quit"))
_SHOWN = (("Left stick", "show-left"), ("Right stick", "show-right"))
#: The tuning row's values, each between a - and a + button: (label, its actions' start).
_TUNED = (("on", "on"), ("off", "off"), ("dwell", "dwell"))
_VALUE_W = 58                   # the room for a value between its - and +


class _Canvas:
    """A picture of a fixed size, shown scaled to fit a window the user can resize."""

    def __init__(self, caption: str) -> None:
        pygame.init()
        desk = pygame.display.get_desktop_sizes()[0]
        scale = min(1.0, 0.9 * desk[0] / W, 0.9 * desk[1] / H)
        self.window = pygame.Window(caption, (int(W * scale), int(H * scale)), resizable=True)
        self.window.minimum_size = (MIN_W, MIN_H)
        self.surface = pygame.Surface((W, H))

    def _fit(self) -> tuple[float, int, int]:
        """The picture's scale in the window, and where its top left corner sits."""
        ww, wh = self.window.size
        s = min(ww / W, wh / H)
        return s, (ww - int(W * s)) // 2, (wh - int(H * s)) // 2

    def present(self) -> None:
        """Show the picture, scaled, with bars where the window's shape is not its own."""
        s, x, y = self._fit()
        out = self.window.get_surface()
        out.fill((0, 0, 0))
        out.blit(pygame.transform.smoothscale(self.surface, (int(W * s), int(H * s))), (x, y))
        self.window.flip()

    def to_canvas(self, pos: tuple[int, int]) -> tuple[int, int]:
        """A position in the window as one in the picture."""
        s, x, y = self._fit()
        return int((pos[0] - x) / s), int((pos[1] - y) / s)


_PLOT_Y = 280                   # the top of the plot and of the stick's square


def _wrapped(text: str, font, width: int, lines: int) -> list[str]:
    """``text`` broken at spaces into at most ``lines`` lines of ``width`` pixels, the last cut."""
    out, line = [], ""
    for word in text.split(" "):
        if line and font.size(f"{line} {word}")[0] > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    out.append(line)
    if len(out) > lines:
        out = out[:lines]
        while font.size(out[-1] + " ...")[0] > width:
            out[-1] = out[-1][:-1]
        out[-1] += " ..."
    return out


def _draw_button(surface, font, rect: pygame.Rect, label: str, hover: bool,
                 enabled: bool = True, on: bool = False) -> None:
    """A button: its label in a box, lit when on, brighter under the mouse, dim when it can do nothing."""
    fill = LIVE if on else (GRID if hover and enabled else PANEL)
    pygame.draw.rect(surface, fill, rect, border_radius=5)
    pygame.draw.rect(surface, INK3 if enabled else GRID, rect, 1, border_radius=5)
    text = font.render(label, True, SURFACE if on else (INK if enabled else INK3))
    surface.blit(text, text.get_rect(center=rect.center))


def _layout() -> list[tuple[str, str, pygame.Rect]]:
    """Where each button sits: (label, action, rect), the actions' row above the tuning's.

    In the tuning row, each value of :data:`_TUNED` sits in the gap between its - and +.
    """
    out, x = [], 20
    for label, action in _ACTIONS:
        w = 22 + 7 * len(label)
        out.append((label, action, pygame.Rect(x, 10, w, 30)))
        x += w + 6
    x = 20
    for label, action in _SHOWN:
        w = 22 + 7 * len(label)
        out.append((label, action, pygame.Rect(x, 48, w, 26)))
        x += w + 6
    for label, action in _TUNED:
        x += 20 + 7 * len(label)                     # its name, before the -
        out.append(("-", f"{action}-", pygame.Rect(x, 48, 26, 26)))
        out.append(("+", f"{action}+", pygame.Rect(x + 26 + _VALUE_W, 48, 26, 26)))
        x += 52 + _VALUE_W
    out.append(("Diagonals", "diag", pygame.Rect(x + 24, 48, 22 + 7 * 9, 26)))
    return out


def _draw_tiles(screen, fonts, pad: Pad | None, down: set[str], keys: frozenset, by: int) -> None:
    """One tile per button the pad has: its name, where it comes from, and its key."""
    f_big, f_sm = fonts
    names = [c for c in pad.controls() if c not in pads.STICK_DIRECTIONS] if pad else []
    cols, bx, gap = 7, 20, 4
    bw = (W - 2 * bx - (cols - 1) * gap) // cols
    for k, name in enumerate(names):
        x, y = bx + (k % cols) * (bw + gap), by + (k // cols) * (62 + gap)
        key, is_down, hue = pads.binding(name), name in down, FACE.get(name, INK2)
        pygame.draw.rect(screen, hue if is_down else PANEL, (x, y, bw, 62), border_radius=4)
        pygame.draw.rect(screen, hue if is_down else GRID, (x, y, bw, 62), 1, border_radius=4)
        ink = SURFACE if is_down else INK2
        screen.blit(f_big.render(name, True, SURFACE if is_down else hue), (x + 8, y + 6))
        screen.blit(f_sm.render(pad.source(name), True, ink), (x + 8, y + 42))
        screen.blit(f_big.render(key or "-", True, ink if key else INK3), (x + bw - 60, y + 36))
        if is_down and key in keys:
            pygame.draw.circle(screen, SURFACE, (x + bw - 10, y + 10), 4)


def _vy(v: float, cy: int, half: int) -> int:
    """A calibrated value as a y pixel on the scrolling plot."""
    return int(cy - max(-1.0, min(1.0, v)) * half)


def _tx(t: float, now: float, px: int, pw: int) -> int:
    """A timestamp as an x pixel, newest at the right edge."""
    return int(px + pw - (now - t) / WINDOW_S * pw)


def _qpos(vx: float, vz: float, mid: tuple[int, int], q: int) -> tuple[int, int]:
    """A calibrated x/y pair as a point in the square stick plot."""
    return (int(mid[0] + max(-1, min(1, vx)) * q // 2),
            int(mid[1] + max(-1, min(1, vz)) * q // 2))


def _save_thresholds(stick: tuple[int, int]) -> str:
    """Write one stick's live thresholds into the calibration in use, keeping the rest of it.

    Only into a calibration of the pad plugged in: with none, there is nothing to save into.
    """
    path = pads.calibration_path()
    blob = None if path is None else pads.read_calibration(path)
    if blob is None:
        return "NOT saved: this pad has no calibration to save into (Calibrate first)"
    record = pads.tunings[stick].record()
    if stick == STICK_AXES:
        blob.update(record)
    else:
        blob[RIGHT_KEY] = {**(blob.get(RIGHT_KEY) or {}), **record}
    path.write_text(json.dumps(blob, indent=2) + "\n")
    return (f"saved {record['axis_on']} / {record['axis_off']} / "
            f"diag {record['axis_diagonal']} -> {path}")


def _watched() -> dict[str, float]:
    """What the scope reloads on a change to: this file, the calibrations, the one in use."""
    paths = [__file__, str(pads.pads_dir()), *([str(pads.calibration_path())] if pads.calibration_path() else [])]
    return {p: (os.path.getmtime(p) if os.path.exists(p) else 0.0) for p in paths}


def _retuned(t: Tuning, action: str, step: float = 0.01) -> Tuning:
    """A stick's settings after one of the tuning row's actions."""
    if action == "on+":
        return replace(t, on=round(min(0.99, t.on + step), 3))
    if action == "on-":
        return replace(t, on=round(max(0.01, t.on - step), 3))
    if action == "off+":
        return replace(t, off=round(min(0.99, t.off + step), 3))
    if action == "off-":
        return replace(t, off=round(max(0.01, t.off - step), 3))
    if action == "dwell-":
        return replace(t, dwell=round(max(0.0, t.dwell - 0.01), 3))
    if action == "dwell+":
        return replace(t, dwell=round(min(0.3, t.dwell + 0.01), 3))
    return replace(t, diagonal=0.7 if t.diagonal >= 1.0 else 1.0)    # diagonals on, or off


def _draw_square(screen, fonts, stick: tuple[int, int], raw: dict[int, float],
                 controls: frozenset, hist: deque, now: float, at: tuple[int, int, int]) -> None:
    """The stick shown, where it is and the rule deciding it; the other stick a small dot."""
    f, f_sm = fonts
    qx, qy, q = at
    tuning, directions = pads.tunings[stick], STICKS[stick]
    sx, sy = (pads.normalise(i, raw.get(i, 0.0)) for i in stick)
    pygame.draw.rect(screen, PANEL, (qx, qy, q, q))
    mid = (qx + q // 2, qy + q // 2)
    pygame.draw.line(screen, GRID, (qx, mid[1]), (qx + q, mid[1]))
    pygame.draw.line(screen, GRID, (mid[0], qy), (mid[0], qy + q))
    # Draw the rule that is actually deciding, so the picture cannot imply a
    # different one: with the classifier that is a distance from rest and a
    # set of directions, not a pair of per-axis boxes.
    classifying = tuning.diagonal >= 1.0
    for frac, col in ((tuning.on, (120, 130, 150)), (tuning.off, (70, 78, 96))):
        r = int(frac * q // 2)
        if classifying:
            pygame.draw.circle(screen, col, mid, r, 1)
        else:
            pygame.draw.rect(screen, col, (mid[0] - r, mid[1] - r, r * 2, r * 2), 1)
    if classifying:
        ordered = sorted(zip(directions, STICK), key=lambda d: tuning.sectors[d[1]])
        for i, (name, base) in enumerate(ordered):
            angle, reach = tuning.sectors[base], q // 2 - 4
            end = (mid[0] + math.cos(angle) * reach, mid[1] + math.sin(angle) * reach)
            pygame.draw.line(screen, (58, 66, 84), mid, end, 1)
            _arrow(screen, base, (int(end[0]), int(end[1])), 9,
                   LIVE if name in controls else INK3)
            nxt = tuning.sectors[ordered[(i + 1) % len(ordered)][1]]
            edge = angle + ((nxt - angle) % (2 * math.pi)) / 2   # the boundary between them
            pygame.draw.line(screen, (44, 50, 64), mid,
                             (mid[0] + math.cos(edge) * reach, mid[1] + math.sin(edge) * reach), 1)
    for i, h in enumerate([h for h in hist if now - h[0] < TRAIL_S]):
        k = 40 + i // 2
        pygame.draw.circle(screen, (k, k + 6, k + 16), _qpos(h[1], h[2], mid, q), 2)
    other = next((s for s in STICKS if s != stick and s[0] in raw), None)
    if other is not None:
        ox, oy = (pads.normalise(i, raw.get(i, 0.0)) for i in other)
        pygame.draw.circle(screen, INK3, _qpos(ox, oy, mid, q), 4)
    cur = _qpos(sx, sy, mid, q)
    pygame.draw.circle(screen, SURFACE, cur, 9)
    pygame.draw.circle(screen, LIVE if controls & set(directions) else X_HUE, cur, 6)
    name = "right stick" if stick == RIGHT_AXES else "left stick" if other else "stick"
    screen.blit(f.render(f"{name} position", True, INK2), (qx + 8, qy + 6))
    rows = [(f"{tuning.rule()}   pushed {math.hypot(sx, sy):.2f}", INK2),
            (f"calibrated {sx:+.3f} {sy:+.3f}", INK),
            (f"raw        {raw.get(stick[0], 0.0):+.3f} {raw.get(stick[1], 0.0):+.3f}", INK3),
            ("centre     " + " ".join(f"{pads.centre(i):+.3f}" for i in stick)
             + ("  (tracked while idle)" if pads.is_calibrated(stick[0]) else "  (uncalibrated)"), INK3)]
    for k, (text, colour) in enumerate(rows):
        screen.blit(f_sm.render(text, True, colour), (qx + 8, qy + q - 60 + k * 14))


class _Scope:
    """``fmri-pad-scope``: what the controller sends, what is made of it, and the
    buttons that calibrate and tune it."""

    def __init__(self, rig: str | None) -> None:
        self.canvas = _Canvas("fmri-gym pad scope")
        font = "Menlo,Monaco,monospace"
        self.f_big, self.f, self.f_sm = (pygame.font.SysFont(font, n) for n in (17, 13, 11))
        self.rig = rig
        bindings = {}
        if rig:
            from .config.rig import RigConfig       # the rig module imports this one
            bindings = RigConfig.open(rig).bindings
        self.bindings = bindings
        self.status, self.watched = pads.init(True, bindings), _watched()
        self.flash, self.flash_until = "", 0.0
        self.hist: deque = deque()
        self.log: deque = deque(maxlen=11)
        self.prev: frozenset = frozenset()
        self.stick, self.running, self.t0 = STICK_AXES, True, time.perf_counter()

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def say(self, text: str, seconds: float = 2.5) -> None:
        self.flash, self.flash_until = text, self.now() + seconds

    def enabled(self, action: str) -> bool:
        """Whether a button can do anything now."""
        pad = pads.first()
        if action in ("all", "left", "buttons"):
            return pad is not None
        if action in ("right", "show-right"):
            return pad is not None and RIGHT_AXES in pad_sticks(pad)
        if action == "save":
            return pads.calibration_path() is not None
        return True

    def do(self, action: str) -> None:
        """What a button does."""
        if not self.enabled(action):
            return
        if action == "quit":
            self.running = False
        elif action in ("show-left", "show-right"):
            self.stick = RIGHT_AXES if action == "show-right" else STICK_AXES
            self.hist.clear()
        elif action in ("all", "left", "right"):
            self.calibrate(action)
        elif action == "buttons":
            out = check_buttons(self.canvas.surface, self.canvas.present, self.canvas.to_canvas)
            self.say(f"{out['confirmed']} buttons confirmed, {out['corrections']} wrong presses "
                     "asked again" if out["done"] else "button check stopped", 4.0)
        elif action == "save":
            self.say(_save_thresholds(self.stick))
            self.watched = _watched()
        elif action == "reload":
            self.reload("reloaded")
        elif action == "clear":
            self.hist.clear()
            self.log.clear()
        else:
            pads.tunings[self.stick] = _retuned(pads.tunings[self.stick], action)

    def calibrate(self, action: str) -> None:
        sticks = {"all": None, "left": [STICK_AXES], "right": [RIGHT_AXES]}[action]
        out = calibrate(self.canvas.surface, sticks=sticks, buttons=action == "all",
                        present=self.canvas.present, to_canvas=self.canvas.to_canvas)
        self.reload(f"calibrated -> {out['config']}" if out["measured"]
                    else f"NOT calibrated: {out['why']}")

    def reload(self, said: str) -> None:
        self.status, self.watched = pads.init(True, self.bindings), _watched()
        self.hist.clear()
        self.say(said, 4.0)

    def events(self) -> None:
        """Clicks on the buttons, and ESC or the window's close to quit.

        Nothing else on the keyboard does anything: a program that turns a pad
        into keystrokes (a gamepad mapper) would otherwise retune or save as the
        pad is pushed, and its keys cannot be told from a person's.
        """
        for e in pygame.event.get((pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN)):
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.KEYDOWN and not getattr(e, "pad", False) \
                    and e.key == pygame.K_ESCAPE:
                self.running = False
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                at = self.canvas.to_canvas(e.pos)
                for _, action, rect in _layout():
                    if rect.collidepoint(at):
                        self.do(action)

    def loop(self, start: str | None) -> None:
        clock = pygame.time.Clock()
        if start:
            self.do(start)
        while self.running:
            self.events()
            if self.now() % 1 < 0.02 and self.watched != _watched():
                self.reload("a calibration changed - reloaded")
            pads.pump()
            self.draw()
            clock.tick(60)
        pygame.quit()

    def draw(self) -> None:
        now, pad = self.now(), pads.first()
        pressed, controls = pads.held(), pads.held_controls()
        raw, down = pad.read() if pad else ({}, set())
        sx, sy = (pads.normalise(i, raw.get(i, 0.0)) for i in self.stick)
        for name in sorted(controls - self.prev):
            key = pads.binding(name) or ("no key bound" if self.rig else "no --rig given")
            self.log.appendleft((now, f"{name:<10} pressed   -> {key}", LIVE))
        for name in sorted(self.prev - controls):
            self.log.appendleft((now, f"{name:<10} released", INK3))
        self.prev = controls
        self.hist.append((now, sx, sy, controls))
        while self.hist and now - self.hist[0][0] > WINDOW_S:
            self.hist.popleft()

        screen = self.canvas.surface
        screen.fill(SURFACE)
        self.draw_toolbar(screen)
        screen.blit(self.f_sm.render(_wrapped(self.status, self.f_sm, W - 40, 1)[0], True,
                                     INK2), (20, 84))
        cal = pads.calibration_status()
        screen.blit(self.f_sm.render(_wrapped(cal, self.f_sm, W - 40, 1)[0], True,
                                     WARN if cal.startswith("UNCAL") else INK2), (20, 98))
        if self.flash and now < self.flash_until:
            screen.blit(self.f.render(self.flash, True, WARN if self.flash.startswith("NOT")
                                      else LIVE), (20, 114))
        _draw_tiles(screen, (self.f_big, self.f_sm), pad, down, pressed, 136)
        lanes = self.draw_plot(screen, pad, controls, now)
        _draw_square(screen, (self.f, self.f_sm), self.stick, raw, controls, self.hist, now,
                     (676, _PLOT_Y, 310))
        ly = _PLOT_Y + 200 + 10 + len(lanes) * 16 + 14
        screen.blit(self.f.render("events", True, INK2), (20, ly))
        for i, (t, text, colour) in enumerate(self.log):
            screen.blit(self.f_sm.render(f"{t:7.2f}s  {text}", True, colour), (20, ly + 20 + i * 14))
        screen.blit(self.f_sm.render("the buttons along the top do everything; ESC quits",
                                     True, INK3), (20, H - 22))
        if pad is None:
            screen.blit(self.f_big.render("no controller plugged in", True, WARN),
                        (684, _PLOT_Y + 150))
        self.canvas.present()

    def draw_toolbar(self, screen) -> None:
        mouse = self.canvas.to_canvas(pygame.mouse.get_pos())
        tuning = pads.tunings[self.stick]
        values = {"on": f"{tuning.on:.2f}", "off": f"{tuning.off:.2f}",
                  "dwell": f"{tuning.dwell * 1000:.0f} ms"}
        for label, action, rect in _layout():
            on = (action == "show-left" and self.stick == STICK_AXES
                  or action == "show-right" and self.stick == RIGHT_AXES
                  or action == "diag" and tuning.diagonal < 1.0)
            _draw_button(screen, self.f_sm, rect, label, rect.collidepoint(mouse),
                         self.enabled(action), on)
            if action[:-1] in values and action.endswith("-"):   # its name, and its value after
                name = self.f_sm.render(action[:-1], True, INK2)
                screen.blit(name, name.get_rect(midright=(rect.x - 6, rect.centery)))
                value = self.f.render(values[action[:-1]], True, INK)
                screen.blit(value, value.get_rect(center=(rect.right + _VALUE_W // 2,
                                                          rect.centery)))

    def draw_plot(self, screen, pad: Pad | None, controls: frozenset, now: float) -> list[str]:
        """The shown stick's two axes over time, and a lane per stick direction. Returns the lanes."""
        tuning, directions = pads.tunings[self.stick], STICKS[self.stick]
        on, off, hist = tuning.on, tuning.off, self.hist
        which = ("right stick" if self.stick == RIGHT_AXES else "left stick") \
            if self.enabled("show-right") else "stick"
        px, py, pw, ph = 20, _PLOT_Y, 630, 200
        cy, half = py + ph // 2, ph // 2
        pygame.draw.rect(screen, PANEL, (px, py, pw, ph))
        for lvl in (-1.0, -0.5, 0.5, 1.0):
            pygame.draw.line(screen, GRID, (px, _vy(lvl, cy, half)), (px + pw, _vy(lvl, cy, half)))
            screen.blit(self.f_sm.render(f"{lvl:+.1f}", True, INK3), (px + 3, _vy(lvl, cy, half) - 12))
        pygame.draw.line(screen, (60, 68, 86), (px, cy), (px + pw, cy))
        for sgn in (1, -1):
            pygame.draw.line(screen, (120, 130, 150), (px, _vy(sgn * on, cy, half)),
                             (px + pw, _vy(sgn * on, cy, half)))
            for seg in range(px, px + pw, 10):
                pygame.draw.line(screen, (80, 88, 106), (seg, _vy(sgn * off, cy, half)),
                                 (seg + 5, _vy(sgn * off, cy, half)))
        if len(hist) > 1:
            for idx, hue in ((1, X_HUE), (2, Y_HUE)):
                pygame.draw.lines(screen, hue, False,
                                  [(_tx(h[0], now, px, pw), _vy(h[idx], cy, half)) for h in hist], 2)
            for a, b in zip(hist, list(hist)[1:]):
                if (b[3] - a[3]) & set(directions):
                    x = _tx(b[0], now, px, pw)
                    pygame.draw.line(screen, LIVE, (x, py), (x, py + ph), 1)
        screen.blit(self.f.render(f"{which} x", True, X_HUE), (px + 8, py + 6))
        screen.blit(self.f.render(f"{which} y", True, Y_HUE), (px + 150, py + 6))
        screen.blit(self.f_sm.render("calibrated units", True, INK3), (px + pw - 100, py + 6))
        ry = py + ph + 10
        lanes = [d for d in (pad.controls() if pad else STICK) if d in pads.STICK_DIRECTIONS]
        for r, name in enumerate(lanes):
            y = ry + r * 16
            _arrow(screen, _ARROW_OF[name], (px + 8, y + 7), 7, LIVE if name in controls else INK3)
            if name in RIGHT_STICK:
                screen.blit(self.f_sm.render("RS", True, INK3), (px + 20, y))
            pygame.draw.rect(screen, PANEL, (px + 46, y + 2, pw - 46, 10))
            for a, b in zip(hist, list(hist)[1:]):
                if name in b[3]:
                    x1, x2 = _tx(a[0], now, px, pw), _tx(b[0], now, px, pw)
                    pygame.draw.rect(screen, LIVE, (max(x1, px + 46), y + 2, max(1, x2 - x1), 10))
        return lanes


def _scope_from_args(description: str, start: str | None) -> None:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--rig", default=None, metavar="RIG.json",
                   help="the rig file whose bindings to show (configs/rigs/<site>/<rig>.json)")
    _Scope(p.parse_args().rig).loop(start)


def scope_main() -> None:
    """``fmri-pad-scope``."""
    _scope_from_args("Watch what the controller sends and what fmri-gym makes of it; "
                     "calibrate and tune it with the buttons along the top.", start=None)
