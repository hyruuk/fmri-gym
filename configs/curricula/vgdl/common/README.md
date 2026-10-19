# VGDL curricula

The `vgdl` backend's curricula: one per game, plus `all.json`, every game for 15 s in a row. Each game's phase lives once in `common/<game>.json` (keys, level 0, `block_size` 25, 15 fps), which its own curriculum and `all.json` both include (README, "Shared phases").

## Setup

The games come from the language_and_experience fork, cloned into `external/vgdl` at the commit the README pins (README, "External checkouts").

## Keys

Every game uses the same map: UP/DOWN/LEFT/RIGHT are actions 0/1/2/3, SPACE is 5, and no key held is 4.

## Curricula

| file | plays |
|---|---|
| `aliens.json` | aliens |
| `avoidGeorge.json` | avoidGeorge |
| `beesAndBirds.json` | beesAndBirds |
| `jaws.json` | jaws |
| `missile_command.json` | missile_command |
| `plaqueAttack.json` | plaqueAttack |
| `portals.json` | portals |
| `preconditions.json` | preconditions |
| `pushBoulders.json` | pushBoulders |
| `relational.json` | relational |
| `all.json` | all ten, 15 s each |

The single-game curricula play their game for 300 s.
