"""The controller: a gamepad or button box whose buttons become keypresses.

Some sites' response device is a game controller on an fMRI interface (a
Current Designs 932, say) that reports HID *joystick* buttons rather than
keystrokes. Nothing a phase's ``keys`` names is ever pressed there, so the game
does not move and the run records no responses, with nothing on screen to say
why. This module presses the keyboard key each control is bound to, so the run
sees a keyboard either way and a curriculum's ``keys`` stay in keyboard keys,
playable at the desk and in the scanner alike. :func:`pump` moves the events
across; it registers with :func:`fmri_gym.keys.get_events`, which runs it
before every read of the queue.

The controls are a gamepad's (:data:`CONTROLS`: each stick's and the D-pad's
four directions, Y B X A, LT RT, LB RB, START BACK), and a pad has the ones it
has. Which raw button or axis is which control is its :class:`Layout`: SDL's,
for a gamepad SDL knows (an Xbox controller, whose raw numbers differ by OS and
connection), or the Current Designs pad's for any other joystick. Which key each
control presses is the rig's (its file's ``"bindings"``, handed to :func:`init`),
one map for every game played on the rig. A control with no binding presses
nothing; a controller that types keys itself needs none. What is *not* the rig
file's is each stick, which rests off centre, travels further one way than the
other, and moves its centre between sessions; that is measured per rig by
``fmri-pad-calibrate`` (below), one file per calibration in :func:`pads_dir`, of
which a run reads the newest made on the pad plugged in, and checked in
``fmri-pad-scope``.

It turns itself on when a pad is plugged in, and a rig file's ``"has_pad": false`` leaves it off.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import pygame
from pygame._sdl2 import controller

from .keys import KEYCODE_BY_NAME, register_event_source, register_held_source

#: The directions of the left stick, the right stick and the D-pad, each up, down, left, right.
STICK = ("UP", "DOWN", "LEFT", "RIGHT")
RIGHT_STICK = ("RS_UP", "RS_DOWN", "RS_LEFT", "RS_RIGHT")
DPAD = ("DPAD_UP", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT")
#: Every control a rig file's ``"bindings"`` can bind to a key; a pad has the ones it has.
CONTROLS = (*STICK, "Y", "B", "X", "LT", "A", "RT", "LB", "RB", "START", "BACK",
            *DPAD, *RIGHT_STICK)
#: The sticks, by their axes: 0 and 1 the left stick, 2 and 3 the right (SDL's
#: numbering of a gamepad's), each read up and left as negative.
STICK_AXES, RIGHT_AXES = (0, 1), (2, 3)
STICKS: dict[tuple[int, int], tuple[str, ...]] = {STICK_AXES: STICK, RIGHT_AXES: RIGHT_STICK}
_DIRECTIONS = frozenset(STICK + RIGHT_STICK)


@dataclass(frozen=True)
class Layout:
    """Which of a joystick's numbered buttons and axes is which control."""

    name: str
    buttons: dict[int, str]
    axes: dict[int, int]            # its axis -> the stick axis it is (:data:`STICKS`)


#: The Current Designs pad. SDL numbers buttons from 0, one less than the HID
#: report does. It also reports two axes pinned at -1.0, which are no control.
CURRENT_DESIGNS = Layout("Current Designs", {0: "Y", 1: "B", 2: "X", 3: "LT", 4: "A", 5: "RT"},
                         {0: 0, 1: 1})
#: A gamepad SDL knows the layout of (an Xbox controller, say), by SDL's names
#: for its controls: its raw numbers differ by pad, OS and connection.
GAMEPAD = Layout("SDL's gamepad layout", {
    pygame.CONTROLLER_BUTTON_Y: "Y", pygame.CONTROLLER_BUTTON_B: "B",
    pygame.CONTROLLER_BUTTON_X: "X", pygame.CONTROLLER_BUTTON_A: "A",
    pygame.CONTROLLER_BUTTON_LEFTSHOULDER: "LB", pygame.CONTROLLER_BUTTON_RIGHTSHOULDER: "RB",
    pygame.CONTROLLER_BUTTON_START: "START", pygame.CONTROLLER_BUTTON_BACK: "BACK",
    pygame.CONTROLLER_BUTTON_DPAD_UP: "DPAD_UP", pygame.CONTROLLER_BUTTON_DPAD_DOWN: "DPAD_DOWN",
    pygame.CONTROLLER_BUTTON_DPAD_LEFT: "DPAD_LEFT",
    pygame.CONTROLLER_BUTTON_DPAD_RIGHT: "DPAD_RIGHT",
}, {
    pygame.CONTROLLER_AXIS_LEFTX: 0, pygame.CONTROLLER_AXIS_LEFTY: 1,
    pygame.CONTROLLER_AXIS_RIGHTX: 2, pygame.CONTROLLER_AXIS_RIGHTY: 3,
})
#: A gamepad's analog triggers, which press their control past :data:`_TRIGGER_ON`.
GAMEPAD_TRIGGERS: dict[int, str] = {
    pygame.CONTROLLER_AXIS_TRIGGERLEFT: "LT", pygame.CONTROLLER_AXIS_TRIGGERRIGHT: "RT",
}
#: The controls a gamepad has, by the names in its SDL mapping.
_MAPPED: dict[str, tuple[str, ...]] = {
    "y": ("Y",), "b": ("B",), "x": ("X",), "a": ("A",), "lefttrigger": ("LT",),
    "righttrigger": ("RT",), "leftshoulder": ("LB",), "rightshoulder": ("RB",),
    "start": ("START",), "back": ("BACK",), "dpup": ("DPAD_UP",), "dpdown": ("DPAD_DOWN",),
    "dpleft": ("DPAD_LEFT",), "dpright": ("DPAD_RIGHT",),
    "leftx": ("LEFT", "RIGHT"), "lefty": ("UP", "DOWN"),
    "rightx": ("RS_LEFT", "RS_RIGHT"), "righty": ("RS_UP", "RS_DOWN"),
}
#: How far an analog trigger is pulled to press, and let go to release, of its travel.
_TRIGGER_ON, _TRIGGER_OFF = 0.5, 0.3
_GAMEPAD_EVENTS = (pygame.CONTROLLERBUTTONDOWN, pygame.CONTROLLERBUTTONUP,
                   pygame.CONTROLLERAXISMOTION)
_PAD_EVENTS = (pygame.JOYBUTTONDOWN, pygame.JOYBUTTONUP, pygame.JOYAXISMOTION, *_GAMEPAD_EVENTS)
_DOWN_EVENTS = (pygame.JOYBUTTONDOWN, pygame.CONTROLLERBUTTONDOWN)


def _scaled(value: float) -> float:
    """A gamepad axis's reading, -32768..32767, as -1..1."""
    return max(-1.0, min(1.0, value / 32767))


class Pad:
    """One plugged-in pad: its SDL handles, and which of its numbers is which control."""

    def __init__(self, index: int) -> None:
        self.joystick = pygame.joystick.Joystick(index)
        self.gamepad = controller.Controller(index) if controller.is_controller(index) else None
        self.layout = CURRENT_DESIGNS if self.gamepad is None else GAMEPAD
        self.name = self.joystick.get_name()
        self.id = self.joystick.get_instance_id()

    def controls(self) -> tuple[str, ...]:
        """The controls this pad has, in :data:`CONTROLS` order."""
        if self.gamepad is None:
            have = {*STICK, *self.layout.buttons.values()}
        else:
            have = {c for key in self.gamepad.get_mapping() for c in _MAPPED.get(key, ())}
        return tuple(c for c in CONTROLS if c in have)

    def source(self, control: str) -> str:
        """Which of its raw buttons or axes a control is, as SDL numbers them ("b6", "+a2")."""
        if self.gamepad is not None:
            key = next((k for k, cs in _MAPPED.items() if control in cs), "")
            return self.gamepad.get_mapping().get(key, "")
        number = {c: b for b, c in self.layout.buttons.items()}.get(control)
        return "" if number is None else f"b{number}"

    def read(self) -> tuple[dict[int, float], set[str]]:
        """Its stick axes now (by :data:`STICKS`, raw), and the buttons and triggers held."""
        if self.gamepad is None:
            j = self.joystick
            axes = {s: j.get_axis(a) for a, s in self.layout.axes.items() if a < j.get_numaxes()}
            return axes, {c for b, c in self.layout.buttons.items()
                          if b < j.get_numbuttons() and j.get_button(b)}
        g = self.gamepad
        axes = {s: _scaled(g.get_axis(a)) for a, s in self.layout.axes.items()}
        held = {c for b, c in self.layout.buttons.items() if g.get_button(b)}
        return axes, held | {c for a, c in GAMEPAD_TRIGGERS.items()
                             if _scaled(g.get_axis(a)) >= _TRIGGER_ON}

#: Where each direction sits as an angle of push. A push is classified by which
#: of these it points nearest to, so these four are the whole of what "which
#: direction" means. They are the obvious four, not fitted to where a cued push
#: pointed (up to 32 degrees off, on the pad this was written for), since that
#: skew is as much the hand as the stick; a calibration's ``"sectors"`` moves them.
IDEAL_SECTORS: dict[str, float] = {
    "RIGHT": 0.0, "DOWN": math.pi / 2, "LEFT": math.pi, "UP": -math.pi / 2,
}
#: How much nearer another direction must be before the stick changes hands.
_SECTOR_KEEP = math.radians(15)

#: Calibrations live outside the checkout, since they describe the controllers on
#: *this* machine, not the task: one file per calibration in :func:`pads_dir`, of
#: which a run reads the newest made on the pad plugged in. This names one file
#: instead, which a run reads and a calibration writes.
CONFIG_ENV = "FMRI_GYM_PAD_CONFIG"

_DOMINANCE = 1.15           # with diagonals, the margin for one axis to take the stick over


@dataclass(frozen=True)
class Tuning:
    """How one stick's push is read. The defaults are an uncalibrated stick's.

    ``on`` and ``off`` are how far the stick must go to press a direction and
    come back to release it, as a fraction of its travel. ``diagonal`` is
    whether it can press two directions at once, and how far the second must go
    as a fraction of the first: at 1.0 it cannot, and a push is exactly one
    direction, what a game with four discrete actions wants; below, the axes are
    read independently, for one that steers. ``dwell`` is how long a new
    direction must hold before it is pressed: a thumb leaving rest crosses a
    neighbouring sector for a frame or two (a real push lasted 254 ms at the
    shortest against transients of 17 ms). ``sectors`` is where each direction
    points, by the left stick's names.
    """

    on: float = 0.2
    off: float = 0.2
    diagonal: float = 1.0
    dwell: float = 0.05
    sectors: dict[str, float] = field(default_factory=lambda: dict(IDEAL_SECTORS))

    @classmethod
    def from_dict(cls, d: dict) -> Tuning:
        """One stick's settings as a calibration file writes them; a value left out is the default's."""
        given = {k: float(d[key]) for k, key in (("on", "axis_on"), ("off", "axis_off"),
                                                 ("diagonal", "axis_diagonal"),
                                                 ("dwell", "stick_dwell_s"))
                 if isinstance(d.get(key), (int, float))}
        sectors = {**IDEAL_SECTORS, **{n: math.radians(float(deg)) for n, deg
                                       in (d.get("sectors") or {}).items()
                                       if n in IDEAL_SECTORS and isinstance(deg, (int, float))}}
        return cls(**given, sectors=sectors)

    def record(self) -> dict:
        """Its thresholds, as a calibration file writes them."""
        return {"axis_on": round(self.on, 3), "axis_off": round(self.off, 3),
                "axis_diagonal": round(self.diagonal, 2), "stick_dwell_s": round(self.dwell, 3)}

    def rule(self) -> str:
        """How it reads a push, in words."""
        how = ("one direction at a time" if self.diagonal >= 1.0
               else f"diagonals past {self.diagonal:.2f}")
        return f"{self.on:.2f}/{self.off:.2f}, {how}, {self.dwell * 1000:.0f} ms dwell"


#: Where the right stick's settings sit in a calibration file; the left's are at its top level.
RIGHT_KEY = "right_stick"
#: The stick's centre is taken once, after it has first been used and then let
#: go: cold and warm are not the same place (the gimbal does not return where it
#: started) and the warm one moves between sessions. How far it may be taken
#: from the calibrated centre is what the calibration measured that movement to
#: be, so a stick held slightly off can never be mistaken for centre.
_SETTLE_S, _SETTLE_STILL, _SETTLE_SLACK = 1.0, 0.01, 1.5

_enabled = False
_calibration: dict = {}                 # axis -> {rest, min, max, drift}
_tuning: dict[tuple[int, int], Tuning] = {s: Tuning() for s in STICKS}
_held: set[str] = set()                 # the controls the pad is holding down
_bindings: dict[str, str] = {}          # control -> the key it presses (the rig's)
_value: dict[int, float] = {}           # axis -> last reading, calibrated
_raw: dict[int, float] = {}             # axis -> last reading, untouched
_zero: dict[int, float] = {}            # axis -> where it is centred now
_pushed: dict[int, str | None] = {}     # axis -> its side, in diagonal mode
_since: dict[str, float] = {}           # direction -> when it first wanted pressing
_quiet: dict[int, tuple] = {}           # axis -> (since, low, high) while still
_settled: set[int] = set()              # axes whose centre has been taken
_pads: dict[int, Pad] = {}              # instance id -> the pad
_unmapped: set[tuple[str, int]] = set()  # (layout, button) already warned of
_foreign = ""                           # the devices only other calibrations were made on
_path: Path | None = None               # the calibration file read
_warmed: set[tuple[int, int]] = set()   # sticks used since their calibration was read
_status = 'off (the rig\'s "has_pad": false)'


def _config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return Path(base).expanduser() / "fmri-gym"


def pads_dir() -> Path:
    """Where this machine's calibrations are written, one file per calibration."""
    return _config_dir() / "pads"


def calibration_path() -> Path | None:
    """The calibration file the pad plugged in is read with, or None if it has none."""
    return _path


def _candidates() -> list[Path]:
    """The calibration files a run chooses from: the one :data:`CONFIG_ENV` names, if
    set; otherwise every one in :func:`pads_dir`, and a ``pad.json`` beside it."""
    override = os.environ.get(CONFIG_ENV)
    if override:
        return [Path(override).expanduser()]
    single = _config_dir() / "pad.json"
    return sorted(pads_dir().glob("*.json")) + ([single] if single.exists() else [])


def _read(path: Path) -> dict | None:
    """A calibration file's contents, or None (and a line on stderr) if it cannot be read."""
    try:
        blob = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        print(f"pad: ignoring unreadable {path}: {exc}", file=sys.stderr)
        return None
    if not isinstance(blob, dict):
        print(f"pad: ignoring {path}: expected {{...}}, got {type(blob).__name__}", file=sys.stderr)
        return None
    return blob


def init(enabled: bool = True, bindings: dict[str, str] | None = None) -> str:
    """Open whatever pad is plugged in, unless the rig file said not to.

    :param enabled: ``False`` to leave the pad off for this run.
    :param bindings: control (:data:`CONTROLS`) -> the key name it presses; a
        control left out presses nothing.
    :return: a one-line status for the run's stderr banner and the manifest.
    """
    global _status, _enabled, _bindings
    _bindings = dict(bindings or {})
    if not enabled:
        _status = 'off (the rig\'s "has_pad": false)'
        return _status
    _enabled = True
    register_held_source(held)
    register_event_source(pump)
    found = _open() or _arrived()
    _load_calibration()
    if not found:
        _status = "no controller plugged in; the keyboard answers"
        return _status
    where = os.environ.get(CONFIG_ENV) or pads_dir()
    cal = (f"calibrated {_path}" if _path is not None
           else f"UNCALIBRATED: no calibration of this pad, only of {_foreign} (run "
                f"fmri-pad-calibrate; writing to {where})" if _foreign
           else f"UNCALIBRATED (run fmri-pad-calibrate; writing to {where})")
    bound = (", ".join(f"{c}={k}" for c, k in _bindings.items()) if _bindings else
             'WARNING: the rig file has no "bindings", so its controls press no key (a '
             "controller that types keys itself needs none)")
    pads = ", ".join(f"{p.name} ({p.layout.name})" for p in _pads.values())
    sticks = f"stick at {_tuning[STICK_AXES].rule()}"
    if any(RIGHT_STICK[0] in p.controls() for p in _pads.values()):
        sticks += f"; right stick at {_tuning[RIGHT_AXES].rule()}"
    _status = f"{pads} -> {bound}; {sticks}; {cal}"
    return _status


def status() -> str:
    """The line :func:`init` last returned, for the manifest and the banner."""
    return _status


def _load_calibration() -> None:
    """Read the newest calibration made on the pad plugged in, if it has one.

    Newest by the ``"measured"`` each file records. One made on another device is
    never read: its device's name is kept for the status line instead.
    """
    global _foreign, _path
    _calibration.clear()
    _tuning.update({s: Tuning() for s in STICKS})
    _foreign, _path = "", None
    # A new calibration means a new centre to find: forget the one taken under
    # the old one rather than carrying it across.
    _zero.clear()
    _settled.clear()
    _quiet.clear()
    _warmed.clear()
    names, mine, others = {p.name for p in _pads.values()}, [], set()
    for path in _candidates():
        blob = _read(path)
        if blob is None:
            continue
        device = str(blob.get("device", "")).split(" (refit of ")[0]
        if device and names and device not in names:
            others.add(repr(device))
            continue
        mine.append((str(blob.get("measured", "")), path, blob))
    if not mine:
        _foreign = ", ".join(sorted(others))
        return
    _, _path, blob = max(mine, key=lambda m: m[0])
    _apply(blob, _path)


def _apply(blob: dict, path: Path) -> None:
    """Take a calibration's sticks: the left's at its top level, the right's, when it
    was measured, under :data:`RIGHT_KEY`. A stick it does not measure reads uncalibrated."""
    right = blob.get(RIGHT_KEY)
    if right is not None and not isinstance(right, dict):
        print(f"pad: ignoring {RIGHT_KEY!r} in {path}: expected {{...}}, got {right!r}",
              file=sys.stderr)
        right = None
    measured = STICK_AXES + (RIGHT_AXES if right is not None else ())
    for index, entry in (blob.get("axes") or {}).items():
        if int(index) not in measured:
            continue
        try:
            axis = {k: float(entry[k]) for k in ("rest", "min", "max")}
        except (KeyError, TypeError, ValueError):
            print(f"pad: ignoring malformed axis {index} in {path}", file=sys.stderr)
            continue
        axis["drift"] = float(entry.get("drift") or 0.0)
        _calibration[int(index)] = axis
    _tuning[STICK_AXES] = Tuning.from_dict(blob)
    if right is not None:
        _tuning[RIGHT_AXES] = Tuning.from_dict(right)


def _open() -> bool:
    """Open the pads, or reopen them after something reset the joystick subsystem.

    ``Display`` calls ``pygame.init()`` when the window goes up, which restarts
    every subsystem and quietly closes handles opened before it. Rather than
    depend on the order, they are checked here and taken again when stale.
    """
    if _pads and next(iter(_pads.values())).joystick.get_init():
        return True
    _pads.clear()
    if not pygame.joystick.get_init():
        pygame.joystick.init()
    if not controller.get_init():
        controller.init()
    for i in range(pygame.joystick.get_count()):
        pad = Pad(i)
        _pads[pad.id] = pad
    return bool(_pads)


def _arrived(seconds: float = 0.5) -> bool:
    """Wait for a pad SDL announces late, and open it.

    On macOS SDL sees a gamepad only with its display subsystem up (which opens
    no window), and the pad arrives as an event about 0.1 s after that.
    """
    pygame.display.init()
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        pygame.event.pump()
        if _open():
            return True
        time.sleep(0.05)
    return False


def first() -> Pad | None:
    """The pad the tools watch and measure: the first one plugged in."""
    return next(iter(_pads.values()), None)


def held() -> frozenset[str]:
    """The keys the pad's held controls are bound to, this frame."""
    return frozenset(_bindings[c] for c in _held if c in _bindings)


def centre(index: int) -> float:
    """Where axis ``index`` is currently taken to rest."""
    return _zero.get(index, _calibration.get(index, {}).get("rest", 0.0))


def _normalise(index: int, value: float) -> float:
    """A raw reading as a fraction of that direction's own travel, -1..1.

    Each side is scaled by its own span, so a pad reaching -0.5 one way and +1.0
    the other still reports -1.0 and +1.0 at its stops. The stops do not move,
    so both spans are measured from wherever the stick is currently centred.
    """
    cal = _calibration.get(index)
    if not cal:
        return value
    rest = centre(index)
    span = (cal["max"] - rest) if value >= rest else (rest - cal["min"])
    if span <= 1e-6:
        return 0.0
    return max(-1.0, min(1.0, (value - rest) / span))


def _settle(now: float) -> None:
    """Take each stick's centre, once, after it has been used and let go."""
    for axes, directions in STICKS.items():
        if axes in _warmed and not set(directions) & _held:
            for index in axes:
                _settle_axis(index, now)


def _settle_axis(index: int, now: float) -> None:
    """Take one axis's centre once it has been still long enough."""
    if index in _settled or index not in _calibration or index not in _raw:
        return
    value = _raw[index]
    since, low, high = _quiet.get(index, (now, value, value))
    low, high = min(low, value), max(high, value)
    if high - low > _SETTLE_STILL:
        _quiet[index] = (now, value, value)         # moved: start again
        return
    _quiet[index] = (since, low, high)
    if now - since < _SETTLE_S:
        return
    _settled.add(index)
    cal, found = _calibration[index], (low + high) / 2
    allowed = max(cal["drift"] * _SETTLE_SLACK, _SETTLE_STILL)
    if abs(found - cal["rest"]) <= allowed:
        _zero[index] = found
    else:
        print(f"pad: axis {index} settled at {found:+.3f}, {abs(found - cal['rest']):.3f} "
              f"from the calibrated centre and further than the {cal['drift']:.3f} it was "
              f"measured to move; keeping the calibrated one. Rerun fmri-pad-calibrate.",
              file=sys.stderr)


def _press(control: str, down: bool) -> None:
    """Hold or release ``control``, and post the keypress of the key it is bound to."""
    if down:
        _held.add(control)
    else:
        _held.discard(control)
    key = _bindings.get(control)
    if key is None:
        return
    # `unicode` is what a "press <char> to go on" screen reads; `pad` marks the
    # event as ours for anything that wants to tell the two apart.
    char = " " if key == "SPACE" else key.lower() if len(key) == 1 else ""
    pygame.event.post(pygame.event.Event(
        pygame.KEYDOWN if down else pygame.KEYUP,
        key=KEYCODE_BY_NAME[key], mod=0, unicode=char if down else "",
        scancode=0, pad=True))


def _away(a: float, b: float) -> float:
    """How far apart two angles are, the short way round."""
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def _classified(axes: tuple[int, int], directions: tuple[str, ...]) -> set[str]:
    """The one direction a stick is being pushed, or none.

    How far it is from centre says *whether* a direction is held, and the angle
    says *which*. Classifying the push as a whole is what keeps the horizontal
    axis from answering a push meant for down: that push does move both axes, a
    lot, but it points much closer to down than to left.
    """
    tuning, (x, y) = _tuning[axes], axes
    nx, ny = _value.get(x, 0.0), _value.get(y, 0.0)
    current = set(directions) & _held
    if math.hypot(nx, ny) < (tuning.off if current else tuning.on):
        return set()
    sectors = {d: tuning.sectors[s] for s, d in zip(STICK, directions)}
    theta = math.atan2(ny, nx)
    best = min(sectors, key=lambda d: _away(theta, sectors[d]))
    for name in current:                    # the one already pressed keeps it
        if _away(theta, sectors[name]) - _away(theta, sectors[best]) < _SECTOR_KEEP:
            return {name}
    return {best}


def _side(index: int, tuning: Tuning) -> str | None:
    """Which way an axis is pushed, holding whatever it was doing in the gap."""
    value, was = _value.get(index, 0.0), _pushed.get(index)
    if value <= -tuning.on:
        return "-"
    if value >= tuning.on:
        return "+"
    return None if abs(value) < tuning.off else was


def _independent(axes: tuple[int, int], directions: tuple[str, ...]) -> set[str]:
    """A stick's two axes read separately, for a game that wants diagonals."""
    tuning, (x, y) = _tuning[axes], axes
    up, down, left, right = directions
    sides = {(x, "-"): left, (x, "+"): right, (y, "-"): up, (y, "+"): down}
    wanted: dict[int, str] = {}
    for index in axes:
        side = _side(index, tuning)
        _pushed[index] = side
        if side is not None:
            wanted[index] = sides[(index, side)]
    if x in wanted and y in wanted:
        strong, weak = (x, y) if abs(_value[x]) >= abs(_value[y]) else (y, x)
        if wanted[weak] in _held and wanted[strong] not in _held \
                and abs(_value[weak]) * _DOMINANCE > abs(_value[strong]):
            strong, weak = weak, strong
        lead = abs(_value[strong])
        ratio = abs(_value[weak]) / lead if lead > 1e-6 else 0.0
        if ratio < tuning.diagonal * (0.8 if wanted[weak] in _held else 1.0):
            del wanted[weak]
    return set(wanted.values())


def _dwelt(names: set[str], directions: tuple[str, ...], dwell: float, now: float) -> set[str]:
    """Of the directions a stick asks for, the ones that have asked long enough."""
    for name in [n for n in _since if n in directions and n not in names]:
        del _since[name]
    for name in names:
        _since.setdefault(name, now)
    return {n for n in names if n in _held or now - _since[n] >= dwell}


def _resolve() -> None:
    """Press what the sticks' positions now mean, and release what they do not."""
    now = time.monotonic()
    _settle(now)
    for index, value in _raw.items():
        _value[index] = _normalise(index, value)
    names: set[str] = set()
    for axes, directions in STICKS.items():
        if not any(a in _raw for a in axes):
            continue
        tuning = _tuning[axes]
        wanted = (_classified(axes, directions) if tuning.diagonal >= 1.0
                  else _independent(axes, directions))
        names |= _dwelt(wanted, directions, tuning.dwell, now) if tuning.dwell > 0 else wanted
    for name in sorted(_DIRECTIONS & (_held - names)):
        _press(name, False)
    for name in sorted(names - _held):
        _warmed.add(next(a for a, ds in STICKS.items() if name in ds))
        _press(name, True)


def pump() -> None:
    """Move the pad's events into the keyboard ones (run by ``keys.get_events``).

    Takes only the joystick and gamepad events off the queue, so whatever else
    is waiting there is still the caller's to read. A gamepad's events come
    twice, as a joystick's and as a gamepad's; only the gamepad's are read.
    """
    if not (_enabled and _open()):
        return
    moved = False
    for event in pygame.event.get(_PAD_EVENTS):
        pad = _pads.get(event.instance_id)
        if pad is None or (event.type in _GAMEPAD_EVENTS) != (pad.gamepad is not None):
            continue
        if event.type in (pygame.JOYAXISMOTION, pygame.CONTROLLERAXISMOTION):
            moved |= _axis(pad, event.axis, event.value)
        else:
            _button(pad, event.button, event.type in _DOWN_EVENTS)
    if moved:
        _resolve()


def _axis(pad: Pad, axis: int, value: float) -> bool:
    """Take an axis's reading. Returns whether it was a stick's."""
    if pad.gamepad is not None:
        value = _scaled(value)
        if axis in GAMEPAD_TRIGGERS:
            _trigger(GAMEPAD_TRIGGERS[axis], value)
            return False
    stick = pad.layout.axes.get(axis)
    if stick is None:
        return False
    _raw[stick] = value
    return True


def _trigger(control: str, value: float) -> None:
    """Press an analog trigger's control past :data:`_TRIGGER_ON`, release it under the off."""
    if control not in _held and value >= _TRIGGER_ON:
        _press(control, True)
    elif control in _held and value < _TRIGGER_OFF:
        _press(control, False)


def _button(pad: Pad, button: int, down: bool) -> None:
    """Press or release the control a button is, or warn once of one that is none."""
    control = pad.layout.buttons.get(button)
    if control is not None:
        _press(control, down)
    elif down and (pad.layout.name, button) not in _unmapped:
        _unmapped.add((pad.layout.name, button))
        print(f"pad: {pad.name}'s button {button} pressed, but it is no control of "
              f"{pad.layout.name} (fmri_gym/pad.py)", file=sys.stderr)


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
    """ESC, or the window closed: the calibration stops, and writes nothing."""


class _Task:
    """The calibration's window, and its recording: one row per frame."""

    def __init__(self, pad: Pad, screen) -> None:
        self.pad, self.screen = pad, screen
        w, self.h = screen.get_size()
        self.mx, scale = w // 2, min(2.0, max(0.8, self.h / 600))
        font = "Menlo,Monaco,monospace"
        self.big, self.mid, self.sml = (pygame.font.SysFont(font, int(n * scale))
                                        for n in (62, 25, 15))
        self.clock = pygame.time.Clock()
        self.buttons = [c for c in CUE_BUTTONS if c in pad.controls()]
        self.samples: list[dict] = []
        self.t0 = time.perf_counter()

    def frame(self, phase: str, cue: str) -> tuple[dict[int, float], set[str]]:
        """Wait for the next frame and record the pad's reading.

        :raises _Quit: on ESC or the window closing.
        """
        self.clock.tick(60)
        for e in pygame.event.get((pygame.QUIT, pygame.KEYDOWN)):
            if e.type == pygame.QUIT or (not getattr(e, "pad", False)
                                         and e.key == pygame.K_ESCAPE):
                raise _Quit
        pygame.event.clear(_PAD_EVENTS)         # read by polling here, not queued for a run
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
        top = 300 if arrow or ring else 235
        for k, line in enumerate(lines):
            surf = (self.big if k == 0 else self.mid).render(line, True, colour if k == 0 else INK2)
            self.screen.blit(surf, surf.get_rect(center=(self.mx, top + 60 + k * 66
                                                          if ring else top + k * 66)))
        s = self.sml.render(sub or "ESC stops; nothing is written", True, INK2)
        self.screen.blit(s, s.get_rect(center=(self.mx, self.h - 40)))
        pygame.display.flip()


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
    return sum(_ask(task, names[d], f"push {which}".strip(), INK, d, answer, idle)
               for d in CUE_DIRECTIONS)


def run(screen=None) -> tuple[list[dict], str, int, int]:
    """Measure the pad's sticks and confirm each of its controls, waiting on each step.

    :param screen: draw on this surface instead of opening a window, so the rig
        check can run the calibration inside the session it has already opened.
    :return: (every frame recorded, the pad's name, controls confirmed, wrong answers).
    :raises SystemExit: with no controller plugged in.
    :raises _Quit: on ESC.
    """
    own = screen is None
    if own:
        pygame.init()
        screen = pygame.display.set_mode((900, 600))
        pygame.display.set_caption("fmri-gym pad calibration")
    try:
        pad = first() if _open() or _arrived() else None
        if pad is None:
            raise SystemExit("no controller found: plug it in and run this again")
        task, has_right = _Task(pad, screen), RIGHT_STICK[0] in pad.controls()
        wrong = _calibrate_stick(task, STICK_AXES, "the left stick" if has_right else "the stick")
        if has_right:
            wrong += _calibrate_stick(task, RIGHT_AXES, "the right stick")
        for name in task.buttons:
            wrong += _ask(task, name, f"press  {name}", FACE.get(name, INK), None,
                          lambda _ax, bt: min(bt) if bt else None, lambda _ax, bt: not bt)
        task.draw(["done"], LIVE)
        confirmed = len(CUE_DIRECTIONS) * (2 if has_right else 1) + len(task.buttons)
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
    """Turn a calibration's recording into its file: each stick's rest, travel and thresholds.

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
            col = [r[f"ax{i}"] for r in samples]
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
           else pads_dir() / f"{_slug(blob['device'])}_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2) + "\n")
    with out.with_suffix(".csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(samples[0]))
        w.writeheader()
        w.writerows(samples)
    return out


def detect() -> str:
    """The name of the controller plugged in, or "" if there is none."""
    if not pygame.joystick.get_init():
        pygame.joystick.init()
    for i in range(pygame.joystick.get_count()):
        return pygame.joystick.Joystick(i).get_name()
    return ""


def calibrate(screen=None, extra: dict | None = None) -> dict:
    """Calibrate the pad and write its calibration file. Returns what was measured.

    :param screen: draw here (the rig check's window, the scope's) rather than in a window of its own.
    :param extra: merged into the file (the rig check records which site and rig measured it).
    """
    try:
        samples, device, confirmed, wrong = run(screen)
    except _Quit:
        return {"measured": False, "why": "stopped before it finished (ESC); nothing written"}
    blob = {**fit(samples), "device": device, **(extra or {})}
    path = save(blob, samples)
    _load_calibration()                      # what follows plays calibrated
    return {"measured": True, "device": device, "config": str(path),
            "samples": str(path.with_suffix(".csv")), "confirmed": confirmed,
            "corrections": wrong, "axis_on": blob["axis_on"], "axis_off": blob["axis_off"],
            "stick_dwell_s": blob["stick_dwell_s"], "axes": blob["axes"],
            **({RIGHT_KEY: blob[RIGHT_KEY]} if RIGHT_KEY in blob else {})}


def calibrate_main() -> None:
    p = argparse.ArgumentParser(
        description="Calibrate this rig's controller: each stick's rest and travel, and a "
                    "check that each control is the one it is named. Every step waits for "
                    "the controller.")
    p.add_argument("--no-scope", action="store_true",
                   help="do not open the scope to check the calibration afterwards")
    args = p.parse_args()
    out = calibrate()
    if not out["measured"]:
        raise SystemExit(out["why"])
    print(f"\n{out['device']}: {out['confirmed']} controls confirmed, "
          f"{out['corrections']} wrong answers asked again", file=sys.stderr)
    for i, a in out["axes"].items():
        print(f"  axis {i}: rest {a['rest']:+.3f}  travel {a['min']:+.3f} .. {a['max']:+.3f}"
              f"  (dither {a['noise']:.4f}, centre moves {a['drift']:.3f})", file=sys.stderr)
    print(f"  stick: on past {out['axis_on']}, off under {out['axis_off']}", file=sys.stderr)
    if RIGHT_KEY in out:
        right = out[RIGHT_KEY]
        print(f"  right stick: on past {right['axis_on']}, off under {right['axis_off']}",
              file=sys.stderr)
    print(f"\ncalibration -> {out['config']}\nrecording   -> {out['samples']}", file=sys.stderr)
    if not args.no_scope:
        # Look at it before trusting it: the scope is where a calibration is
        # checked, by pushing the stick and watching the thresholds act on it.
        print("opening the scope to check it - ESC closes it", file=sys.stderr)
        scope_main()


W, H = 1020, 800
WINDOW_S, TRAIL_S = 8.0, 0.8
#: The arrow each stick direction is drawn as.
_ARROW_OF = {**{d: d for d in STICK}, **dict(zip(RIGHT_STICK, STICK))}


def _draw_buttons(screen, fonts, pad: Pad | None, down: set[str], keys: frozenset) -> None:
    """One tile per button the pad has: its name, where it comes from, and its key."""
    f_big, f_sm = fonts
    names = [c for c in pad.controls() if c not in _DIRECTIONS] if pad else []
    cols, bx, by, gap = 7, 20, 66, 4
    bw = (W - 2 * bx - (cols - 1) * gap) // cols
    for k, name in enumerate(names):
        x, y = bx + (k % cols) * (bw + gap), by + (k // cols) * (62 + gap)
        key, is_down, hue = _bindings.get(name), name in down, FACE.get(name, INK2)
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
    path = _path
    blob = None if path is None else _read(path)
    if blob is None:
        return "NOT saved: this pad has no calibration to save into (K calibrates it)"
    record = _tuning[stick].record()
    if stick == STICK_AXES:
        blob.update(record)
    else:
        blob[RIGHT_KEY] = {**(blob.get(RIGHT_KEY) or {}), **record}
    path.write_text(json.dumps(blob, indent=2) + "\n")
    return (f"saved {record['axis_on']} / {record['axis_off']} / "
            f"diag {record['axis_diagonal']} -> {path}")


def _watched() -> dict[str, float]:
    """What the scope reloads on a change to: this file, the calibrations, the one in use."""
    paths = [__file__, str(pads_dir()), *([str(_path)] if _path else [])]
    return {p: (os.path.getmtime(p) if os.path.exists(p) else 0.0) for p in paths}


#: The scope's keys that tune the stick it is showing.
_TUNING_KEYS = (pygame.K_UP, pygame.K_DOWN, pygame.K_RIGHT, pygame.K_LEFT,
                pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET, pygame.K_d)


def _retuned(t: Tuning, key: int, step: float) -> Tuning:
    """A stick's settings after one of :data:`_TUNING_KEYS`."""
    if key == pygame.K_UP:
        return replace(t, on=round(min(0.99, t.on + step), 3))
    if key == pygame.K_DOWN:
        return replace(t, on=round(max(0.01, t.on - step), 3))
    if key == pygame.K_RIGHT:
        return replace(t, off=round(min(0.99, t.off + step), 3))
    if key == pygame.K_LEFT:
        return replace(t, off=round(max(0.01, t.off - step), 3))
    if key == pygame.K_LEFTBRACKET:
        return replace(t, dwell=round(max(0.0, t.dwell - 0.01), 3))
    if key == pygame.K_RIGHTBRACKET:
        return replace(t, dwell=round(min(0.3, t.dwell + 0.01), 3))
    return replace(t, diagonal=0.7 if t.diagonal >= 1.0 else round(min(1.0, t.diagonal + step * 2), 2))


def _draw_square(screen, fonts, stick: tuple[int, int], raw: dict[int, float],
                 controls: frozenset, hist: deque, now: float, at: tuple[int, int, int]) -> None:
    """The stick shown, where it is and the rule deciding it; the other stick a small dot."""
    f, f_sm = fonts
    qx, qy, q = at
    tuning, directions = _tuning[stick], STICKS[stick]
    sx, sy = (_normalise(i, raw.get(i, 0.0)) for i in stick)
    pygame.draw.rect(screen, PANEL, (qx, qy, q, q))
    mid = (qx + q // 2, qy + q // 2)
    pygame.draw.line(screen, GRID, (qx, mid[1]), (qx + q, mid[1]))
    pygame.draw.line(screen, GRID, (mid[0], qy), (mid[0], qy + q))
    # Draw the rule that is actually deciding, so the picture cannot imply a
    # different one: with the classifier that is a distance from rest and a
    # set of fitted directions, not a pair of per-axis boxes.
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
        ox, oy = (_normalise(i, raw.get(i, 0.0)) for i in other)
        pygame.draw.circle(screen, INK3, _qpos(ox, oy, mid, q), 4)
    cur = _qpos(sx, sy, mid, q)
    pygame.draw.circle(screen, SURFACE, cur, 9)
    pygame.draw.circle(screen, LIVE if controls & set(directions) else X_HUE, cur, 6)
    name = "right stick" if stick == RIGHT_AXES else "left stick" if other else "stick"
    screen.blit(f.render(f"{name} position", True, INK2), (qx + 8, qy + 6))
    rows = [(f"{tuning.rule()}   pushed {math.hypot(sx, sy):.2f}", INK2),
            (f"calibrated {sx:+.3f} {sy:+.3f}", INK),
            (f"raw        {raw.get(stick[0], 0.0):+.3f} {raw.get(stick[1], 0.0):+.3f}", INK3),
            ("centre     " + " ".join(f"{centre(i):+.3f}" for i in stick)
             + ("  (tracked while idle)" if stick[0] in _calibration else "  (uncalibrated)"), INK3)]
    for k, (text, colour) in enumerate(rows):
        screen.blit(f_sm.render(text, True, colour), (qx + 8, qy + q - 60 + k * 14))


def scope_main() -> None:
    argparse.ArgumentParser(
        description="Watch what the controller sends and what fmri-gym makes of it.",
        epilog="TAB switches between the sticks; for the one shown, UP/DOWN sets axis_on, "
               "LEFT/RIGHT sets axis_off, [ and ] set the dwell a direction must hold for, "
               "D switches between one direction at a time and diagonals, S saves them into "
               "the calibration; K calibrates the pad, R reloads, ESC quits.").parse_args()
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("fmri-gym pad scope")
    clock = pygame.time.Clock()
    f_big = pygame.font.SysFont("Menlo,Monaco,monospace", 17)
    f = pygame.font.SysFont("Menlo,Monaco,monospace", 13)
    f_sm = pygame.font.SysFont("Menlo,Monaco,monospace", 11)

    status = init()
    watched = _watched()
    flash, flash_until = "", 0.0
    hist: deque = deque()
    log: deque = deque(maxlen=11)
    prev_held: frozenset = frozenset()
    stick = STICK_AXES
    t0 = time.perf_counter()

    running = True
    while running:
        now = time.perf_counter() - t0
        pad = first()
        has_right = pad is not None and RIGHT_STICK[0] in pad.controls()
        for e in pygame.event.get((pygame.QUIT, pygame.KEYDOWN)):
            if e.type == pygame.QUIT:
                running = False
                continue
            if getattr(e, "pad", False):
                continue                    # the pad's own presses are data, not UI
            step = 0.05 if e.mod & pygame.KMOD_SHIFT else 0.01
            if e.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif e.key == pygame.K_TAB and has_right:
                stick = RIGHT_AXES if stick == STICK_AXES else STICK_AXES
                hist.clear()
            elif e.key in _TUNING_KEYS:
                _tuning[stick] = _retuned(_tuning[stick], e.key, step)
                if e.key == pygame.K_d:
                    flash, flash_until = f"axis_diagonal {_tuning[stick].diagonal}", now + 1.5
            elif e.key == pygame.K_s:
                flash, flash_until = _save_thresholds(stick), now + 2.5
                watched = _watched()
            elif e.key == pygame.K_k:
                out = calibrate(screen=screen)
                status, watched = init(), _watched()
                hist.clear()
                flash = (f"calibrated -> {out['config']}" if out["measured"]
                         else f"NOT calibrated: {out['why']}")
                flash_until = now + 4.0
            elif e.key == pygame.K_r:
                _load_calibration()
                status = init()
                flash, flash_until = "reloaded", now + 1.5
            elif e.key == pygame.K_c:
                hist.clear()
                log.clear()

        if now % 1 < 0.02:                  # pick up an edit to a watched file
            for p, was in list(watched.items()):
                if (os.path.getmtime(p) if os.path.exists(p) else 0.0) != was:
                    status, watched = init(), _watched()
                    flash, flash_until = f"{os.path.basename(p)} changed - reloaded", now + 2.0
                    break

        pump()
        pressed, controls = held(), frozenset(_held)
        raw, down = pad.read() if pad else ({}, set())
        sx, sy = (_normalise(i, raw.get(i, 0.0)) for i in stick)

        for name in sorted(controls - prev_held):
            log.appendleft((now, f"{name:<10} DOWN  -> {_bindings.get(name) or 'unbound'}", LIVE))
        for name in sorted(prev_held - controls):
            log.appendleft((now, f"{name:<10} up", INK3))
        prev_held = controls
        hist.append((now, sx, sy, controls))
        while hist and now - hist[0][0] > WINDOW_S:
            hist.popleft()

        # ------------------------------------------------------------- draw
        screen.fill(SURFACE)
        tuning, directions = _tuning[stick], STICKS[stick]
        on, off = tuning.on, tuning.off
        which = ("right stick" if stick == RIGHT_AXES else "left stick") if has_right else "stick"
        screen.blit(f_big.render("pad scope", True, INK), (20, 14))
        screen.blit(f.render(status[:60], True, INK2), (140, 18))
        screen.blit(f.render(f"{which}: axis_on {on:.2f}   axis_off {off:.2f}", True, INK),
                    (W - 360, 10))
        screen.blit(f_sm.render(("TAB other stick, " if has_right else "")
                                + "UP/DOWN on, LEFT/RIGHT off, S saves"
                                if off < on else "OFF >= ON: no hysteresis, expect chatter",
                                True, INK3 if off < on else WARN), (W - 360, 28))
        if flash and now < flash_until:
            screen.blit(f.render(flash, True, WARN if flash.startswith("NOT") else LIVE), (20, 40))

        _draw_buttons(screen, (f_big, f_sm), pad, down, pressed)

        px, py, pw, ph = 20, 210, 630, 200
        cy, half = py + ph // 2, ph // 2
        pygame.draw.rect(screen, PANEL, (px, py, pw, ph))
        for lvl in (-1.0, -0.5, 0.5, 1.0):
            pygame.draw.line(screen, GRID, (px, _vy(lvl, cy, half)), (px + pw, _vy(lvl, cy, half)))
            screen.blit(f_sm.render(f"{lvl:+.1f}", True, INK3), (px + 3, _vy(lvl, cy, half) - 12))
        pygame.draw.line(screen, (60, 68, 86), (px, cy), (px + pw, cy))
        for sgn in (1, -1):
            pygame.draw.line(screen, (120, 130, 150), (px, _vy(sgn * on, cy, half)), (px + pw, _vy(sgn * on, cy, half)))
            for seg in range(px, px + pw, 10):
                pygame.draw.line(screen, (80, 88, 106), (seg, _vy(sgn * off, cy, half)), (seg + 5, _vy(sgn * off, cy, half)))
        if len(hist) > 1:
            for idx, hue in ((1, X_HUE), (2, Y_HUE)):
                pygame.draw.lines(screen, hue, False, [(_tx(h[0], now, px, pw), _vy(h[idx], cy, half)) for h in hist], 2)
            for a, b in zip(hist, list(hist)[1:]):
                if (b[3] - a[3]) & set(directions):
                    pygame.draw.line(screen, LIVE, (_tx(b[0], now, px, pw), py), (_tx(b[0], now, px, pw), py + ph), 1)
        screen.blit(f.render(f"{which} x", True, X_HUE), (px + 8, py + 6))
        screen.blit(f.render(f"{which} y", True, Y_HUE), (px + 150, py + 6))
        screen.blit(f_sm.render("calibrated units", True, INK3), (px + pw - 100, py + 6))

        ry = py + ph + 10
        lanes = [d for d in (pad.controls() if pad else STICK) if d in _DIRECTIONS]
        for r, name in enumerate(lanes):
            y = ry + r * 16
            _arrow(screen, _ARROW_OF[name], (px + 8, y + 7), 7,
                   LIVE if name in controls else INK3)
            if name in RIGHT_STICK:
                screen.blit(f_sm.render("RS", True, INK3), (px + 20, y))
            pygame.draw.rect(screen, PANEL, (px + 46, y + 2, pw - 46, 10))
            if len(hist) > 1:
                for a, b in zip(hist, list(hist)[1:]):
                    if name in b[3]:
                        x1 = int(px + pw - (now - a[0]) / WINDOW_S * pw)
                        x2 = int(px + pw - (now - b[0]) / WINDOW_S * pw)
                        pygame.draw.rect(screen, LIVE, (max(x1, px + 46), y + 2, max(1, x2 - x1), 10))

        _draw_square(screen, (f, f_sm), stick, raw, controls, hist, now, (676, py, 310))

        ly = ry + len(lanes) * 16 + 14
        screen.blit(f.render("events", True, INK2), (px, ly))
        for i, (t, text, colour) in enumerate(log):
            screen.blit(f_sm.render(f"{t:7.2f}s  {text}", True, colour), (px, ly + 20 + i * 14))
        screen.blit(f_sm.render(("TAB stick   " if has_right else "")
                                + "UP/DOWN on   LEFT/RIGHT off   [ ] dwell   D diagonals   "
                                "S save   K calibrate   R reload   C clear   ESC quit", True, INK3), (px, H - 22))
        if pad is None:
            screen.blit(f_big.render("no controller plugged in", True, WARN), (684, py + 155))

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


def main() -> None:
    """``python -m fmri_gym.pad``: name each control as it is pressed and released."""
    pygame.init()
    pygame.display.set_mode((320, 64))
    print(init(), file=sys.stderr)
    if not _pads:
        return
    print("Press each control (Ctrl-C to stop).", file=sys.stderr)
    try:
        while True:
            before = set(_held)
            pump()
            for name in sorted(_held - before):
                print(f"  pressed -> {name}")
            for name in sorted(before - _held):
                print(f"  released   {name}")
            pygame.time.wait(5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
