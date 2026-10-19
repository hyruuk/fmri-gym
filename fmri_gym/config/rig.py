"""The rig config: how one rig plays every run, whatever its curriculum.

A **rig** is one setup a run is played on -- a monitor, a scanner, a response
device, a trigger line -- and a site can have several. Each has its own rig
file in the repo, ``configs/rigs/<site>/<rig>.json`` (``configs/rigs/README.md``),
named by its ``site`` and ``rig`` fields, so every site's rigs are kept and
reviewed with the code; ``$FMRI_GYM_RIGS`` points at another folder of them.
A curriculum says what is played and plays the same on every rig; the rig file
says how this rig plays it, and every run played on the rig reads it.

``fmri-play --rig <site>/<rig>`` picks the rig (:meth:`RigConfig.open`). With
one rig file there is nothing to pick; with none, or with several and no
``--rig``, nothing starts: which window, controller and data folder a session
gets is not something to guess.

A rig file holds what the rig is -- :data:`FIELDS`, every one required, filed
with every run so that rig checks from every site can be pooled -- and, each
optional, how its runs are played (:data:`DEFAULTS` when left out), under the
field names of :class:`RigConfig` and :class:`ScreenConfig`, so the file and the
code say the same thing:

- ``"screen"``: ``{"size": "1024x768", "is_fullscreen": false, "monitor": 0,
  "has_vsync": true}``, the monitor by index (0: the first); fullscreen takes
  the monitor's resolution;
- ``"has_pad"``: read a plugged-in game controller (:mod:`fmri_gym.pad`);
- ``"has_audio"``: play game audio; ``false`` mutes every block;
- ``"data_root"``: where the BIDS tree goes, ``<root>/sub-XX/ses-NNN/beh/<run>/``;
- ``"triggers"``: the scanner's start sync and the trigger codes, over which
  line (:class:`~fmri_gym.triggers.TriggerSettings`); left out, the run waits
  for ``=`` and sends no codes, and says so on the experimenter screen and in
  the manifest (``defaulted``).

It is written by hand or in the rig form (``python -m fmri_gym.checks rig --rig
<site>/<rig>``), which a rig check also opens when its rig has no valid file.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from ..triggers import TriggerError, TriggerSettings

MODALITIES = ("fmri", "meg", "eeg", "behav")
#: What the rig is: every key required (``notes`` may be empty).
FIELDS = {
    "site": "the site, and the rig file's folder: letters, digits, _ and -",
    "rig": "this rig at the site, and the rig file's name: letters, digits, _ and -",
    "pi": "the PI's initials, in capitals (Momchil Tomov: MT)",
    "modality": f"one of {', '.join(MODALITIES)}",
    "monitor": "monitor or projector, make and model, and its settings that matter",
    "photodiode": "photodiode make/model and where it is plugged in",
    "audio_path": "from the sound card to the ear: amplifier, tubes, headphones",
    "trigger_hardware": "trigger box / port / scanner interface",
    "notes": "anything else (may be empty)",
}
#: What each of :data:`FIELDS` looks like, for the form's placeholders.
EXAMPLES = {
    "site": "mri_3T", "rig": "stimpc1", "pi": "MT", "modality": "fmri",
    "monitor": "BOLDscreen 32 LCD, 120 Hz, 1920x1080",
    "photodiode": "BioSemi photodiode on sound card input 0",
    "audio_path": "RME Babyface -> Sensimetrics S14 insert earphones",
    "trigger_hardware": "fORP 932 USB, '=' at each volume",
}
#: How a run is played where the rig file says nothing: a window at the desk,
#: the pad and the audio on.
DEFAULTS: dict[str, Any] = {
    "screen": {"size": "1024x768", "is_fullscreen": False, "monitor": 0, "has_vsync": True},
    "has_pad": True, "has_audio": True, "data_root": "data", "triggers": {},
}
#: Trigger presets the editor and the rig check offer. Each names ``sync.mode``
#: and ``backend`` outright: a section that leaves them out runs, but is
#: reported as NOT SET.
TRIGGER_PRESETS: dict[str, dict] = {
    "fMRI (wait for '=', no codes)": {
        "sync": {"mode": "wait", "key": "="}, "backend": "null"},
    # MEG and EEG alike: the stimulus PC starts the recording and marks it, over serial.
    "MEG (start from the trigger line, serial codes)": {
        "sync": {"mode": "send", "delay": 0.0}, "backend": "serial", "port": "/dev/ttyUSB0"},
    "EEG (start from the trigger line, serial codes)": {
        "sync": {"mode": "send", "delay": 0.0}, "backend": "serial", "port": "/dev/ttyUSB0"},
    "Behavioural (no scanner)": {
        "sync": {"mode": "none"}, "backend": "null"},
}
#: Site and rig: values in the TSV, never in a file name, so BIDS's letters-and-digits
#: rule does not bind them; still no spaces, as rows are grouped by them when pooled.
_LABEL = re.compile(r"[A-Za-z0-9_-]+")
_INITIALS = re.compile(r"[A-Z]{2,4}")
_SIZE = re.compile(r"([1-9]\d*)x([1-9]\d*)")
_MAKE = ("make one with the rig's long rig check, which opens a form for it: fmri-play "
         "--curriculum configs/curricula/checks/rig-check-long.json --subject sub-rig --ses 1 "
         "--run 1 --rig <site>/<rig> (python -m fmri_gym.checks rig --rig <site>/<rig> opens "
         "the form alone)")
_REPO_RIGS = Path(__file__).resolve().parents[2] / "configs" / "rigs"


@dataclass(frozen=True, slots=True)
class ScreenConfig:
    """The window a rig's runs are shown in: a rig file's ``"screen"``."""

    #: the window, ``(width, height)`` in pixels, when not fullscreen.
    size: tuple[int, int]
    is_fullscreen: bool
    #: which monitor the window opens on, by index.
    monitor: int
    #: lock flips to the monitor refresh.
    has_vsync: bool

    @classmethod
    def from_dict(cls, screen: dict) -> ScreenConfig:
        """Read a checked ``"screen"`` (:func:`problems`), ``DEFAULTS["screen"]`` where it
        says nothing; ``size`` is written ``<width>x<height>``."""
        screen = {**DEFAULTS["screen"], **screen}
        width, height = (int(n) for n in _SIZE.fullmatch(screen["size"]).groups())
        return cls(size=(width, height), is_fullscreen=screen["is_fullscreen"],
                   monitor=screen["monitor"], has_vsync=screen["has_vsync"])

    def record(self) -> dict[str, Any]:
        """As a rig file writes it."""
        return {"size": f"{self.size[0]}x{self.size[1]}", "is_fullscreen": self.is_fullscreen,
                "monitor": self.monitor, "has_vsync": self.has_vsync}


@dataclass(frozen=True, slots=True)
class RigConfig:
    """One rig file, checked: what the rig is, and how every run on it is played."""

    #: the rig's name, ``<site>/<rig>``: its file's place under :func:`folder`.
    name: str
    #: what the rig is, :data:`FIELDS` as written.
    fields: dict[str, str]
    #: the window.
    screen: ScreenConfig
    #: read a plugged-in game controller as key presses.
    has_pad: bool
    #: play game audio; ``False`` mutes every block and opens no output.
    has_audio: bool
    #: the BIDS tree's root.
    data_root: str
    #: the scanner's start sync and the trigger codes, over which line.
    triggers: TriggerSettings

    @classmethod
    def from_dict(cls, rig: dict, name: str) -> RigConfig:
        """Read a rig file's JSON, :data:`DEFAULTS` where it says nothing.

        :param rig: the file's JSON.
        :param name: ``<site>/<rig>``, which the file's ``site`` and ``rig`` fields must match.
        :raises ValueError: naming each field that is missing, unknown or invalid, and
            how to fix it.
        """
        found = problems(rig)
        named = name_of(rig) if not found else name
        if named != name:
            found.append(f"site/rig are {named!r}: a rig file sits at <site>/<rig>.json, "
                         f"{path(named)}")
        if found:
            raise ValueError(f"{path(name)}: " + "; ".join(found) + ". Fix it by hand, or in "
                             f"the form: python -m fmri_gym.checks rig --rig {name}")
        settings = {**DEFAULTS, **rig}
        return cls(name=name, fields={k: rig[k] for k in FIELDS},
                   screen=ScreenConfig.from_dict(rig.get("screen", {})),
                   has_pad=settings["has_pad"], has_audio=settings["has_audio"],
                   data_root=settings["data_root"],
                   triggers=TriggerSettings.from_dict(settings["triggers"]))

    @classmethod
    def open(cls, name: str | None) -> RigConfig:
        """The rig ``name``'s file, or the only one there is, checked.

        :param name: ``<site>/<rig>``; ``None`` for the only rig file there is.
        :raises ValueError: see :func:`choose`, :func:`load` and :meth:`from_dict`.
        """
        name = choose(name)
        return cls.from_dict(load(name), name)

    @property
    def has_trigger_setup(self) -> bool:
        """Whether the file says how the run starts or where codes go (not both defaulted)."""
        return set(self.triggers.defaulted) != {"sync.mode", "backend"}

    def without_trigger_line(self) -> RigConfig:
        """This rig with no trigger line (``null`` backend): what a rig check plays on when
        the line does not open, so its other checks still run. ``send`` becomes ``none``,
        since it needs a line; the file is untouched."""
        sync = self.triggers.sync
        if sync.mode == "send":
            sync = replace(sync, mode="none")
        return replace(self, triggers=replace(self.triggers, backend="null", sync=sync))

    def record(self) -> dict[str, Any]:
        """The rig file as the run played it, every default written out: the manifest's."""
        return {**self.fields, "screen": self.screen.record(), "has_pad": self.has_pad,
                "has_audio": self.has_audio, "data_root": self.data_root,
                "triggers": asdict(self.triggers)}

    def status(self) -> str:
        """One line for the console: which file, and what it settles."""
        screen = self.screen
        window = ("fullscreen" if screen.is_fullscreen
                  else f"{screen.size[0]}x{screen.size[1]} window")
        return (f"{path(self.name)} | {window} on monitor {screen.monitor}, vsync "
                f"{'on' if screen.has_vsync else 'off'} | pad {'on' if self.has_pad else 'off'}"
                f" | audio {'on' if self.has_audio else 'off'} | data in {self.data_root}"
                f" | triggers {self.triggers.sync.mode}, codes over {self.triggers.backend}")


def folder() -> Path:
    """Where the rig files are: the repo's ``configs/rigs/``, or ``$FMRI_GYM_RIGS``."""
    return Path(os.environ.get("FMRI_GYM_RIGS") or _REPO_RIGS)


def path(name: str) -> Path:
    """The file of the rig ``name``, ``<site>/<rig>`` (it need not exist)."""
    return folder() / f"{name}.json"


def names() -> list[str]:
    """The rigs there is a file for, as ``<site>/<rig>``."""
    return sorted(f"{p.parent.name}/{p.stem}" for p in folder().glob("*/*.json"))


def choose(name: str | None) -> str:
    """The rig to play on: ``name``, or the only one there is.

    :raises ValueError: when ``name`` is not ``<site>/<rig>`` or has no file, or none
        is given and there is not exactly one; the message says how to pick or make one.
    """
    found = names()
    have = f"there are {', '.join(found)}" if found else "there are none"
    if name is not None and name.count("/") != 1:
        raise ValueError(f"--rig {name}: a rig is named <site>/<rig> ({have})")
    if name is not None:
        if name in found:
            return name
        raise ValueError(f"--rig {name}: no {path(name)} ({have}); {_MAKE}")
    if len(found) == 1:
        return found[0]
    if found:
        raise ValueError(f"{len(found)} rigs in {folder()} ({', '.join(found)}): "
                         "pick one with --rig <site>/<rig>")
    if Path("rig.json").exists():  # a rig check's file in the working directory is not read
        raise ValueError(f"no rig file in {folder()}: move rig.json there as <site>/<rig>.json, "
                         f"named by its site and rig fields, or {_MAKE}")
    raise ValueError(f"no rig file in {folder()}: {_MAKE}")


def load(name: str) -> dict:
    """The rig file of ``name`` as written, unchecked (the form edits an invalid one).

    :raises ValueError: a file that is not a JSON object.
    """
    where = path(name)
    try:
        with where.open() as f:
            rig = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where}: not JSON ({exc})") from None
    if not isinstance(rig, dict):
        raise ValueError(f"{where}: expected a JSON object")
    return rig


def draft(name: str | None) -> dict:
    """What the rig form starts from for the rig ``name`` (``<site>/<rig>``): its file
    as written, named after the file; only the name when it has no file, or one that
    is not JSON; nothing at all for no name.
    """
    if name is None:
        return {}
    try:
        written = load(name) if name in names() else {}
    except ValueError:
        written = {}
    site, _, rig = name.partition("/")
    return {**written, "site": site, "rig": rig}


def name_of(rig: dict) -> str:
    """A rig file's name, ``<site>/<rig>``, from its fields."""
    return f"{rig['site']}/{rig['rig']}"


def save(rig: dict) -> Path:
    """Write ``rig`` as the file of the rig it names, ``<site>/<rig>.json``.

    :return: the file written.
    """
    where = path(name_of(rig))
    where.parent.mkdir(parents=True, exist_ok=True)
    with where.open("w") as f:
        json.dump(rig, f, indent=2)
        f.write("\n")
    return where


def problems(rig: dict) -> list[str]:
    """What keeps ``rig`` from being a rig file: every one of :data:`FIELDS`, the
    optional :data:`DEFAULTS` of the right type, no other key.

    :return: one line per problem; empty when it is valid.
    """
    out = [f"missing {k!r} ({what})" for k, what in FIELDS.items() if k not in rig]
    out += [f"unknown key {k!r}" for k in rig if k not in FIELDS and k not in DEFAULTS]
    fields = {k: v for k, v in rig.items() if k in FIELDS}
    out += [f"{k!r} must be a string" for k, v in fields.items() if not isinstance(v, str)]
    out += [f"{k!r} is empty" for k, v in fields.items() if k != "notes" and v == ""]
    for k in ("site", "rig"):
        if isinstance(rig.get(k), str) and rig[k] and not _LABEL.fullmatch(rig[k]):
            out.append(f"{k!r} must be letters, digits, _ or -, got {rig[k]!r}")
    if isinstance(rig.get("pi"), str) and rig["pi"] and not _INITIALS.fullmatch(rig["pi"]):
        out.append(f"'pi' must be 2 to 4 capital letters (initials), got {rig['pi']!r}")
    if rig.get("modality") not in (*MODALITIES, "", None):
        out.append(f"'modality' must be one of {MODALITIES}, got {rig['modality']!r}")
    return out + _setting_problems(rig)


def _setting_problems(rig: dict) -> list[str]:
    """What is wrong with the file's ``screen``, ``has_pad``, ``has_audio`` and ``data_root``."""
    screen = rig.get("screen", {})
    if not isinstance(screen, dict):
        return [f'"screen": expected {{"size", "is_fullscreen", "monitor", "has_vsync"}}, '
                f"got {screen!r}"]
    out = [f'"screen": unknown key {k!r}' for k in screen if k not in DEFAULTS["screen"]]
    size = screen.get("size", DEFAULTS["screen"]["size"])
    if not (isinstance(size, str) and _SIZE.fullmatch(size)):
        out.append(f'"screen": "size" must be <width>x<height> such as 1024x768, got {size!r}')
    monitor = screen.get("monitor", DEFAULTS["screen"]["monitor"])
    if not isinstance(monitor, int) or isinstance(monitor, bool) or monitor < 0:
        out.append(f"\"screen\": \"monitor\" must be a monitor's index (0: the first), "
                   f"got {monitor!r}")
    flags = (("screen.is_fullscreen", screen.get("is_fullscreen", False)),
             ("screen.has_vsync", screen.get("has_vsync", True)),
             ("has_pad", rig.get("has_pad", True)), ("has_audio", rig.get("has_audio", True)))
    out += [f'"{k}" must be true or false, got {v!r}' for k, v in flags if not isinstance(v, bool)]
    root = rig.get("data_root", DEFAULTS["data_root"])
    if not isinstance(root, str) or not root:
        out.append(f'"data_root" must be a folder, got {root!r}')
    return out + trigger_problems(rig.get("triggers", {}))


def trigger_problems(section: Any) -> list[str]:
    """What :class:`~fmri_gym.triggers.TriggerSettings` refuses in a ``triggers`` section, as text.

    The port is the one check added here: the settings accept a missing port
    and the backend refuses it on opening, which a check before the run never does.

    :return: at most one problem (the first the settings raise).
    """
    if not isinstance(section, dict):
        return [f'"triggers": expected an object, got {section!r}']
    try:
        s = TriggerSettings.from_dict(section)
    except (TypeError, ValueError, TriggerError) as exc:
        return [str(exc)]
    if s.backend in ("serial", "parallel") and not s.port:
        return [f"triggers: backend {s.backend!r} needs a port"]
    return []
