# Atari (ALE) curricula

The `ale` backend's curricula: ten Atari 2600 games, one file each (300 s of play), and `all.json`, the ten for 15 s each. A game's file and `all.json` both include `common/<game>.json` (its env id, mode and fps) and state their own keys and duration (README, "Shared phases").

## Setup

`uv sync --extra ale` installs ale-py, whose ROMs ship with it: nothing to import.

## Keys

A phase's `keys` values are indices into the game's own action set, which differs per game. Read it with `gym.make("ALE/Pong-v5").unwrapped.get_action_meanings()` (README, "Keys"): for Pong, `['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']`, so RIGHT=2, LEFT=3, and `""` (no key held) is 0.

## MsPacman's turns

The game turns only on the frame Ms. Pac-Man is centred on a junction, and only if a direction is held on that frame: a key tapped and released before it is lost. `mspacman.json` sets `"has_buffered_keys": true`, so a frame with no key held sends the last direction held and a turn pressed early is taken at the next junction it fits; the log records the action sent.

## Curricula

| file | plays |
|---|---|
| `all.json` | the ten games below, 15 s each |
| `asterix.json` | Asterix |
| `beamrider.json` | BeamRider |
| `boxing.json` | Boxing |
| `breakout.json` | Breakout |
| `enduro.json` | Enduro |
| `mspacman.json` | MsPacman |
| `pong.json` | Pong |
| `qbert.json` | Qbert |
| `seaquest.json` | Seaquest |
| `spaceinvaders.json` | SpaceInvaders |
