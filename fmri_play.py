"""fmri_play.py -- run any Gymnasium-compatible game as an fMRI task.

One experiment framework across backends: Atari (ALE), stable-retro consoles
(NES/SNES/Genesis/...), and any plain Gymnasium env. The backend is chosen
per game block in the curriculum; the experiment loop is identical for all.

One config file is one run, and this plays it: nothing here loops over runs,
opens an editor or writes a config. A session of several runs is a shell
script with one of these commands per line (README, "Runs and sessions"),
which ``fmri-edit`` writes and any shell plays.

How it is played -- the window, the controller, the audio, where the data
goes -- is the rig's, not the command's: each rig has a file
(:mod:`fmri_gym.rig`), and ``--rig`` names it when the machine has several.

Usage (--ses and --run say which run of which session this is; a session
script passes the same --ses to all of its runs, each with its own --run):
    python fmri_play.py --subject sub-01 --curriculum my.json --ses 1 --run 1
    python fmri_play.py ... --rig scanner3T    # one of this machine's rigs
    python fmri_play.py ... --dummy-trigger    # testing: no experimenter/scanner wait
    python fmri_play.py ... --duration 600 --resume  # a 10-min run of a longer curriculum

See configs/curricula/demos/mixed.json for a curriculum that mixes all three backends,
and README.md for the config schema.
"""

from __future__ import annotations

import argparse
import signal
import sys

from fmri_gym import Run, checks
from fmri_gym.config import RunConfig
from fmri_gym.curriculum import EXIT_QUIT, load_curriculum, validate_curriculum
from fmri_gym.display import quit_like_esc
from fmri_gym.rig import RigConfig
from fmri_gym.run import END_FIXATION_S

#: Settings of the rig file (:mod:`fmri_gym.rig`), under the flags a command might try.
_RIG_SETTINGS = ("--size", "--fullscreen", "--monitor", "--no-vsync", "--no-pad", "--no-audio",
                 "--data-root")


class _Parser(argparse.ArgumentParser):
    """argparse, plus where the numbers it insists on come from."""

    def error(self, message: str) -> None:
        if "--ses" in message or "--run" in message:
            message += ("\n  a run says which run of which session it is: a session script "
                        "passes one --ses to all its runs and gives each its own --run. At the "
                        "desk: `fmri-ses --subject <sub>` prints the next free session, and "
                        "--run 1 is the first run of this task in it")
        flags = [f for f in _RIG_SETTINGS if f in message]
        if flags:
            message += (f"\n  {', '.join(flags)}: how a run is played is the rig's, in its file "
                        "~/.config/fmri-gym/rigs/<rig>.json (\"screen\", \"pad\", \"audio\", "
                        "\"data_root\"); --rig picks the rig")
        super().error(message)


def _parser() -> argparse.ArgumentParser:
    """The command line: one run, and the rig that plays it."""
    p = _Parser(description="Run any gym game as an fMRI task.")
    p.add_argument("--subject", default="sub-test", help="BIDS subject: sub-<letters/digits>")
    p.add_argument("--curriculum", required=True, help="the run's curriculum JSON (see "
                   "README); fmri-edit writes one without the JSON")
    p.add_argument("--ses", type=int, required=True,
                   help="BIDS session number, from 1: which scanning session this run belongs "
                   "to. A session script takes it once and passes it to every run (fmri-ses)")
    p.add_argument("--run", type=int, required=True,
                   help="this task's run number in the session, from 1: which run of the "
                   "design this is, so it stays the same however the session went. A run that "
                   "already has data is re-acquired beside it, never overwritten")
    p.add_argument("--duration", type=float, default=None,
                   help="the run's length in seconds from the scanner trigger: the phases stop "
                   "--end-fixation seconds before it, whatever is playing, and the run closes "
                   "on a fixation cross. Default: play the curriculum to its end")
    p.add_argument("--end-fixation", type=float, default=END_FIXATION_S, metavar="SECONDS",
                   help="seconds of the fixation a run with --duration closes on, for the "
                   f"haemodynamic response to the last thing played (default {END_FIXATION_S:g})")
    p.add_argument("--resume", action="store_true",
                   help="carry on from where this subject's latest run of this config (any "
                   "session) stopped, mid-block included. Default: start at the beginning")
    p.add_argument("--rig", help="the rig it is played on, by name: its file "
                   "~/.config/fmri-gym/rigs/<rig>.json says the window, controller, audio and "
                   "data root. Default: this machine's only rig; a rig check makes one")
    p.add_argument("--dummy-trigger", action="store_true")
    return p


def main() -> None:
    args = _parser().parse_args()
    curriculum = load_curriculum(args.curriculum)
    # The editor's Check, so a file edited by hand gets it too.
    problems = validate_curriculum(curriculum)
    if problems:
        raise ValueError(f"{args.curriculum}: " + "; ".join(problems))

    is_rig_check = checks.has_checks(curriculum)
    # A rig check is how a rig gets its file: it opens the form when there is none.
    rig = checks.ensure_rig(args.rig) if is_rig_check else RigConfig.open(args.rig)
    print(f"rig: {rig.status()}", file=sys.stderr)
    run_config = RunConfig.from_args(args, rig)
    trigger_error = None
    if is_rig_check:  # a line that opens, first
        trigger_error = checks.prepare(curriculum, run_config.curriculum_path)
    run = Run.from_config_and_curriculum(run_config, curriculum)
    run.logger.set_extra("rig", rig.record())
    if is_rig_check:
        run.logger.set_extra("trigger_error", trigger_error)
    previous = signal.signal(signal.SIGINT, quit_like_esc)
    try:
        completed = run.play()
    finally:
        run.close()
        # Last: a terminal's Ctrl+C can come twice (to uv and to us), the second one late.
        signal.signal(signal.SIGINT, previous)
    if is_rig_check:
        # A failed test is a finding, listed in the report: the session goes on.
        checks.finish(run, run_config.ses, run_config.data_root)
    if not completed:
        # A session script (set -e) must not start the next run after an ESC.
        sys.exit(EXIT_QUIT)


if __name__ == "__main__":
    main()
