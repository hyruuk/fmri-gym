# Rig checks

Curricula whose phases are checks instead of games (README, "Timing").

- `rig-check.json`: before a session, under a minute, most of it pressing the buttons. The controls check asks for the keys on the editor's Controls tab. The photodiode reads on this PC's input 0 (readout `soundcard`) or in the MEG/EEG recording (readout `recording`). Results: `report.md` and `rigcheck.tsv` in the run's folder; this rig's rig file is asked for when it has none.
- `rig-check-long.json`: once per rig, and after any hardware, driver or OS change; about 25 minutes. The display for 2 min; the frame test at 60, 30, 20 and 50 fps (50 divides no common refresh), 1 min each, idle and with every core busy; 20 passes of every code; 850 flashes. It measures offsets to a fraction of a ms, their drift, missed refreshes and pulses, and the TR on this PC's clock.

**Triggers.** Both ship with `"triggers": null` until the rig's are set: the first rig check asks for them (a preset or your own) and saves them in the file; the Triggers tab changes them. They must be the session's own: with sync `wait` the check waits for the scanner and counts its pulses.
