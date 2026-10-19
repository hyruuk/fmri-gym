"""The rig config: how one rig plays every run, whatever its curriculum.

A **rig** is one setup a run is played on -- a monitor, a scanner, a response
device, a trigger line -- and a site can have several. Each has its own rig
file, ``~/.config/fmri-gym/rigs/<name>.json`` (``$XDG_CONFIG_HOME`` moves it),
outside the checkout because it describes the hardware here, not the task. A
curriculum says what is played and plays the same on every rig; the rig file
says how this rig plays it, and every run played on the rig reads it.

``fmri-play --rig <name>`` picks the rig (:meth:`RigConfig.open`). A machine
with one rig file needs no ``--rig``; with none, or with several and no
``--rig``, nothing starts: which window, controller and data folder a session
gets is not something to guess.

A rig file holds what the rig is -- :data:`FIELDS`, every one required, filed
with every run so that rig checks from every site can be pooled -- and, each
optional, how its runs are played (:data:`DEFAULTS` when left out):

- ``"screen"``: ``{"size": "1024x768", "fullscreen": false, "monitor": 0,
  "vsync": true}``, the monitor by index (0: the first); fullscreen takes the
  monitor's resolution;
- ``"pad"``: read a plugged-in game controller (:mod:`fmri_gym.pad`);
- ``"audio"``: play game audio; ``false`` mutes every block;
- ``"data_root"``: where the BIDS tree goes, ``<root>/sub-XX/ses-NNN/beh/<run>/``.

It is written by hand or in the rig form (``python -m fmri_gym.checks rig --rig
<name>``), which a rig check also opens when its rig has no valid file.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODALITIES = ("fmri", "meg", "eeg", "behav")
#: What the rig is: every key required (``notes`` may be empty).
FIELDS = {
    "site": "the site: letters, digits, _ and -",
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
    "screen": {"size": "1024x768", "fullscreen": False, "monitor": 0, "vsync": True},
    "pad": True, "audio": True, "data_root": "data",
}
#: Site and rig: values in the TSV, never in a file name, so BIDS's letters-and-digits
#: rule does not bind them; still no spaces, as rows are grouped by them when pooled.
_LABEL = re.compile(r"[A-Za-z0-9_-]+")
_INITIALS = re.compile(r"[A-Z]{2,4}")
_SIZE = re.compile(r"([1-9]\d*)x([1-9]\d*)")
_MAKE = ("make one with the rig's long rig check, which opens a form for it: fmri-play "
         "--curriculum configs/rig-check-long.json --subject sub-rig --ses 1 --run 1 --rig <name> "
         "(python -m fmri_gym.checks rig --rig <name> opens the form alone)")


@dataclass(frozen=True, slots=True)
class RigConfig:
    """One rig file, checked: what the rig is, and how every run on it is played."""

    #: the rig's name: its file's, and its ``"rig"`` field.
    name: str
    #: what the rig is, :data:`FIELDS` as written.
    fields: dict[str, str]
    #: the window, ``(width, height)`` in pixels, when not fullscreen.
    size: tuple[int, int]
    is_fullscreen: bool
    #: which monitor the window opens on, by index.
    monitor: int
    #: lock flips to the monitor refresh.
    has_vsync: bool
    #: read a plugged-in game controller as key presses.
    has_pad: bool
    #: play game audio; ``False`` mutes every block and opens no output.
    has_audio: bool
    #: the BIDS tree's root.
    data_root: str

    @classmethod
    def from_dict(cls, rig: dict, name: str) -> RigConfig:
        """Read a rig file's JSON, :data:`DEFAULTS` where it says nothing.

        :param rig: the file's JSON.
        :param name: the file's name, which its ``"rig"`` field must match.
        :raises ValueError: naming each field that is missing, unknown or invalid, and
            how to fix it.
        """
        found = problems(rig)
        if rig.get("rig") not in (name, None, ""):
            found.append(f'"rig" is {rig["rig"]!r}: a rig file is named after its rig, '
                         f"{rig['rig']}.json")
        if found:
            raise ValueError(f"{path(name)}: " + "; ".join(found) + ". Fix it by hand, or in "
                             f"the form: python -m fmri_gym.checks rig --rig {name}")
        screen = {**DEFAULTS["screen"], **rig.get("screen", {})}
        width, height = (int(n) for n in _SIZE.fullmatch(screen["size"]).groups())
        return cls(name=name, fields={k: rig[k] for k in FIELDS}, size=(width, height),
                   is_fullscreen=screen["fullscreen"], monitor=screen["monitor"],
                   has_vsync=screen["vsync"], has_pad=rig.get("pad", DEFAULTS["pad"]),
                   has_audio=rig.get("audio", DEFAULTS["audio"]),
                   data_root=rig.get("data_root", DEFAULTS["data_root"]))

    @classmethod
    def open(cls, name: str | None) -> RigConfig:
        """The rig ``name``'s file, or this machine's only one, checked.

        :param name: the rig, by name; ``None`` for the machine's only one.
        :raises ValueError: see :func:`choose`, :func:`load` and :meth:`from_dict`.
        """
        name = choose(name)
        return cls.from_dict(load(name), name)

    def record(self) -> dict[str, Any]:
        """The rig file as the run played it, every default written out: the manifest's."""
        return {**self.fields,
                "screen": {"size": f"{self.size[0]}x{self.size[1]}",
                           "fullscreen": self.is_fullscreen, "monitor": self.monitor,
                           "vsync": self.has_vsync},
                "pad": self.has_pad, "audio": self.has_audio, "data_root": self.data_root}

    def status(self) -> str:
        """One line for the console: which file, and what it settles."""
        window = ("fullscreen" if self.is_fullscreen
                  else f"{self.size[0]}x{self.size[1]} window")
        return (f"{path(self.name)} | {window} on monitor {self.monitor}, vsync "
                f"{'on' if self.has_vsync else 'off'} | pad {'on' if self.has_pad else 'off'}"
                f" | audio {'on' if self.has_audio else 'off'} | data in {self.data_root}")


def folder() -> Path:
    """Where the rig files are: ``~/.config/fmri-gym/rigs/``."""
    base = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return Path(base).expanduser() / "fmri-gym" / "rigs"


def path(name: str) -> Path:
    """The file of the rig ``name`` (it need not exist)."""
    return folder() / f"{name}.json"


def names() -> list[str]:
    """The rigs this machine has a file for, by name."""
    return sorted(p.stem for p in folder().glob("*.json"))


def choose(name: str | None) -> str:
    """The rig to play on: ``name``, or this machine's only one.

    :raises ValueError: when ``name`` has no file, or none is given and there
        is not exactly one; the message says how to pick or make one.
    """
    found = names()
    if name is not None:
        if name in found:
            return name
        have = f"this machine has {', '.join(found)}" if found else "this machine has none"
        raise ValueError(f"--rig {name}: no {path(name)} ({have}); {_MAKE}")
    if len(found) == 1:
        return found[0]
    if found:
        raise ValueError(f"{len(found)} rigs in {folder()} ({', '.join(found)}): "
                         "pick one with --rig <name>")
    # A rig.json in the working directory or beside rigs/ is not read: say where it goes.
    loose = next((p for p in (folder().parent / "rig.json", Path("rig.json")) if p.exists()),
                 None)
    if loose:
        raise ValueError(f"no rig file in {folder()}: move {loose} there as <name>.json, its "
                         f'"rig" field as the name, or {_MAKE}')
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
    """What the rig form starts from for the rig ``name``: its file as written, named
    after the file; only the name when it has no file, or one that is not JSON.
    """
    try:
        written = load(name) if name in names() else {}
    except ValueError:
        written = {}
    return {**written, "rig": name or ""}


def save(rig: dict) -> Path:
    """Write ``rig`` as the file of the rig it names (``rigs/<rig>.json``).

    :return: the file written.
    """
    where = path(rig["rig"])
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
    """What is wrong with the file's ``screen``, ``pad``, ``audio`` and ``data_root``."""
    screen = rig.get("screen", {})
    if not isinstance(screen, dict):
        return [f'"screen": expected {{"size", "fullscreen", "monitor", "vsync"}}, got {screen!r}']
    out = [f'"screen": unknown key {k!r}' for k in screen if k not in DEFAULTS["screen"]]
    size = screen.get("size", DEFAULTS["screen"]["size"])
    if not (isinstance(size, str) and _SIZE.fullmatch(size)):
        out.append(f'"screen": "size" must be <width>x<height> such as 1024x768, got {size!r}')
    monitor = screen.get("monitor", DEFAULTS["screen"]["monitor"])
    if not isinstance(monitor, int) or isinstance(monitor, bool) or monitor < 0:
        out.append(f"\"screen\": \"monitor\" must be a monitor's index (0: the first), "
                   f"got {monitor!r}")
    flags = (("screen.fullscreen", screen.get("fullscreen", False)),
             ("screen.vsync", screen.get("vsync", True)),
             ("pad", rig.get("pad", True)), ("audio", rig.get("audio", True)))
    out += [f'"{k}" must be true or false, got {v!r}' for k, v in flags if not isinstance(v, bool)]
    root = rig.get("data_root", DEFAULTS["data_root"])
    if not isinstance(root, str) or not root:
        out.append(f'"data_root" must be a folder, got {root!r}')
    return out
