"""The run config: one run's command line, and the rig it is played on.

``fmri_play`` owns the flags and their spellings; :class:`RunConfig` names what
they mean to the run, and :meth:`RunConfig.from_args` is the one place the
command line is read. How the run is shown and heard and where its data goes
are the rig's (:class:`~fmri_gym.config.rig.RigConfig`), held here rather than
copied, so each setting has one name and one home.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from .rig import RigConfig


@dataclass(frozen=True, slots=True)
class RunConfig:
    """One run's command line and the rig it is played on, as the run reads it.

    Every field is required, so a flag a caller forgets is a ``TypeError`` at
    the desk and never a default in the scanner.
    """

    #: ``sub-<label>``: the BIDS subject.
    subject: str
    #: the curriculum file; its name is the run's task label.
    curriculum_path: str
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
    #: skip the experimenter and scanner waits: a test run, not a session.
    dummy_trigger: bool
    #: the rig it is played on: window, pad, audio, data root.
    rig: RigConfig

    @classmethod
    def from_args(cls, args: argparse.Namespace, rig: RigConfig) -> RunConfig:
        """Read ``fmri_play``'s parsed flags.

        :param args: the namespace of ``fmri_play``'s parser, one attribute per flag.
        :param rig: the rig ``--rig`` picked.
        """
        return cls(subject=args.subject, curriculum_path=args.curriculum, ses=args.ses,
                   run=args.run, duration=args.duration, end_fixation=args.end_fixation,
                   resume=args.resume,
                   dummy_trigger=args.dummy_trigger, rig=rig)
