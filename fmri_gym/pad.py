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
``fmri-pad-calibrate`` (:mod:`fmri_gym.pad_scope`), one file per calibration in :func:`pads_dir`, of
which a run reads the newest made on the pad plugged in, and checked in
``fmri-pad-scope``.

It turns itself on when a pad is plugged in, and a rig file's ``"has_pad": false`` leaves it off.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
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
STICK_DIRECTIONS = frozenset(STICK + RIGHT_STICK)


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
PAD_EVENTS = (pygame.JOYBUTTONDOWN, pygame.JOYBUTTONUP, pygame.JOYAXISMOTION, *_GAMEPAD_EVENTS)
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
tunings: dict[tuple[int, int], Tuning] = {s: Tuning() for s in STICKS}
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


def calibration_status() -> str:
    """Which calibration the pad plugged in is read with, or why it has none, in a line."""
    where = os.environ.get(CONFIG_ENV) or pads_dir()
    if _path is not None:
        return f"calibrated {_path}"
    if _foreign:
        return (f"UNCALIBRATED: no calibration of this pad, only of {_foreign} (run "
                f"fmri-pad-calibrate; writing to {where})")
    return f"UNCALIBRATED (run fmri-pad-calibrate; writing to {where})"


def _candidates() -> list[Path]:
    """The calibration files a run chooses from: the one :data:`CONFIG_ENV` names, if
    set; otherwise every one in :func:`pads_dir`, and a ``pad.json`` beside it."""
    override = os.environ.get(CONFIG_ENV)
    if override:
        return [Path(override).expanduser()]
    single = _config_dir() / "pad.json"
    return sorted(pads_dir().glob("*.json")) + ([single] if single.exists() else [])


def read_calibration(path: Path) -> dict | None:
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
    load_calibration()
    if not found:
        _status = "no controller plugged in; the keyboard answers"
        return _status
    cal = calibration_status()
    bound = (", ".join(f"{c}={k}" for c, k in _bindings.items()) if _bindings else
             'WARNING: the rig file has no "bindings", so its controls press no key (a '
             "controller that types keys itself needs none)")
    pads = ", ".join(f"{p.name} ({p.layout.name})" for p in _pads.values())
    sticks = f"stick at {tunings[STICK_AXES].rule()}"
    if any(RIGHT_STICK[0] in p.controls() for p in _pads.values()):
        sticks += f"; right stick at {tunings[RIGHT_AXES].rule()}"
    _status = f"{pads} -> {bound}; {sticks}; {cal}"
    return _status


def status() -> str:
    """The line :func:`init` last returned, for the manifest and the banner."""
    return _status


def load_calibration() -> None:
    """Read the newest calibration made on the pad plugged in, if it has one.

    Newest by the ``"measured"`` each file records. One made on another device is
    never read: its device's name is kept for the status line instead.
    """
    global _foreign, _path
    _calibration.clear()
    tunings.update({s: Tuning() for s in STICKS})
    _foreign, _path = "", None
    # A new calibration means a new centre to find: forget the one taken under
    # the old one rather than carrying it across.
    _zero.clear()
    _settled.clear()
    _quiet.clear()
    _warmed.clear()
    names, mine, others = {p.name for p in _pads.values()}, [], set()
    for path in _candidates():
        blob = read_calibration(path)
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
    tunings[STICK_AXES] = Tuning.from_dict(blob)
    if right is not None:
        tunings[RIGHT_AXES] = Tuning.from_dict(right)


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


def find() -> Pad | None:
    """The first pad plugged in, opened, waiting briefly for one SDL announces late; or None."""
    return first() if _open() or _arrived() else None


def binding(control: str) -> str | None:
    """The key the rig binds ``control`` to, or None."""
    return _bindings.get(control)


def held_controls() -> frozenset[str]:
    """The controls the pad is holding down, this frame."""
    return frozenset(_held)


def is_calibrated(index: int) -> bool:
    """Whether stick axis ``index`` reads by a calibration."""
    return index in _calibration


def first() -> Pad | None:
    """The pad the tools watch and measure: the first one plugged in."""
    return next(iter(_pads.values()), None)


def held() -> frozenset[str]:
    """The keys the pad's held controls are bound to, this frame."""
    return frozenset(_bindings[c] for c in _held if c in _bindings)


def centre(index: int) -> float:
    """Where axis ``index`` is currently taken to rest."""
    return _zero.get(index, _calibration.get(index, {}).get("rest", 0.0))


def normalise(index: int, value: float) -> float:
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
    tuning, (x, y) = tunings[axes], axes
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
    tuning, (x, y) = tunings[axes], axes
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
        _value[index] = normalise(index, value)
    names: set[str] = set()
    for axes, directions in STICKS.items():
        if not any(a in _raw for a in axes):
            continue
        tuning = tunings[axes]
        wanted = (_classified(axes, directions) if tuning.diagonal >= 1.0
                  else _independent(axes, directions))
        names |= _dwelt(wanted, directions, tuning.dwell, now) if tuning.dwell > 0 else wanted
    for name in sorted(STICK_DIRECTIONS & (_held - names)):
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
    for event in pygame.event.get(PAD_EVENTS):
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


def detect() -> str:
    """The name of the controller plugged in, or "" if there is none."""
    if not pygame.joystick.get_init():
        pygame.joystick.init()
    for i in range(pygame.joystick.get_count()):
        return pygame.joystick.Joystick(i).get_name()
    return ""


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
