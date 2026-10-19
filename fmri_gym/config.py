"""The run config: one run's command line, as the run reads it.

Three kinds of settings are kept apart, because they are shared differently:

- the **curriculum** (:mod:`fmri_gym.curriculum`): what is played, the
  sequence of phases of one run. It is tied to a game and shared by every
  subject at every site;
- the **run config** (this module): who plays that curriculum, where the data
  goes and how this machine shows it -- subject, session and run number,
  window, the test switches. It is specific to one run;
- the **rig config**: the site's own settings, the same for all of its
  subjects. Today that is the curriculum's ``triggers`` section and the rig
  file of :mod:`fmri_gym.checks`.

``fmri_play`` owns the flags and their spellings; :class:`RunConfig` names what
they mean to the run, and :meth:`RunConfig.from_args` is the one place the
command line is read.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunConfig:
    """One run's command line, as the run reads it.

    Every field is required, so a flag a caller forgets is a ``TypeError`` at
    the desk and never a default in the scanner, and the switches are stated
    positively (``has_audio``, ``has_pad``, ``has_vsync``): the ``--no-*``
    reading is the command line's, made once at its edge.
    """

    #: ``sub-<label>``: the BIDS subject.
    subject: str
    #: the curriculum file; its name is the run's task label.
    curriculum_path: str
    #: the BIDS tree's root.
    data_root: str
    #: the BIDS session number, from 1.
    ses: int
    #: this task's run number in the session, from 1.
    run: int
    #: the run's seconds from the trigger, closing fixation included;
    #: ``None`` plays the curriculum to its end.
    duration: float | None
    #: seconds of the fixation a run with a ``duration`` closes on.
    end_fixation: float
    #: carry on from where the subject's latest run of this curriculum stopped.
    resume: bool
    #: the window, ``(width, height)`` in pixels.
    size: tuple[int, int]
    is_fullscreen: bool
    #: which monitor the window opens on, by index.
    monitor: int
    #: lock flips to the monitor refresh.
    has_vsync: bool
    #: skip the experimenter and scanner waits: a test run, not a session.
    dummy_trigger: bool
    #: play game audio; ``False`` mutes every block and opens no output.
    has_audio: bool
    #: read a plugged-in game controller as key presses.
    has_pad: bool

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> RunConfig:
        """Read ``fmri_play``'s parsed flags.

        ``--size`` is split into ``(width, height)`` and the ``--no-*``
        switches are put positively.

        :param args: the namespace of ``fmri_play``'s parser, one attribute per flag.
        :raises ValueError: on a ``--size`` that is not ``WIDTHxHEIGHT``.
        """
        try:
            width, height = (int(x) for x in args.size.lower().split("x"))
        except ValueError:
            raise ValueError(f"--size {args.size!r}: expected WIDTHxHEIGHT, such as 1024x768"
                             ) from None
        return cls(
            subject=args.subject, curriculum_path=args.curriculum, data_root=args.data_root,
            ses=args.ses, run=args.run, duration=args.duration,
            end_fixation=args.end_fixation, resume=args.resume,
            size=(width, height), is_fullscreen=args.fullscreen, monitor=args.monitor,
            has_vsync=not args.no_vsync, dummy_trigger=args.dummy_trigger,
            has_audio=not args.no_audio, has_pad=not args.no_pad)
