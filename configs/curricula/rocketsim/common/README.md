# Rocket League curricula

The `rocketsim` backend's curricula: Rocket League's soccar, the subject's blue car against an orange bot that chases the ball, one match per episode. `levels/soccar.json` includes `common/soccar.json` (a 5:00 match, the keyboard's controls, the instructions) and names its title (README, "Shared phases").

## Setup

`uv sync --extra rocketsim` installs [rocketsim-gym](https://github.com/hyruuk/rocketsim-gym) at the commit pyproject.toml pins, and RocketSim's wheel: macOS 14+ on arm64, Linux (x86_64, aarch64) or Windows x86_64. The frames are drawn offscreen with OpenGL 3.3; on Linux that is Mesa's EGL (`libegl1 libegl-mesa0 libgl1-mesa-dri`), which needs no display. The arena is rocketsim-gym's homemade one, built from Rocket League's published field dimensions: its curves, corners and goals are close to the game's, not the same.

## Buttons

The env's action is MultiBinary(12), a controller's buttons: 0 throttle, 1 reverse, 2 stick left, 3 stick right (steer on the ground, yaw in the air), 4 stick up (nose down), 5 stick down (nose up), 6 jump, 7 boost, 8 powerslide (in the air, the stick left / right rolls), 9 ball cam (a press switches between car cam and ball cam), 10 air roll left, 11 air roll right. Held buttons combine, and each stick button is the stick pushed all the way. `soccar.json` keeps Rocket League's PC keys W / S (throttle, reverse), A / D (steer) and Q / E (air roll). A key holds one button here, so pitch, which Rocket League also puts on W / S, has keys of its own (UP / DOWN); jump and boost, on the mouse in Rocket League, are on SPACE and LEFT SHIFT, which moves powerslide to LEFT CTRL and ball cam to C. Keep the Controls text in sync by hand.

## Timing

Physics runs at 120 Hz and a step advances `ticks_per_step` (2) ticks, so the game plays at its real speed at 60 fps; a phase at another fps plays faster or slower, and the run says so. A step takes under 0.1 ms and a 1024x576 frame about 3 ms on an Apple M4.

## Logging

Per frame: the action, the reward (+1 for each blue goal, -1 for each orange one), and the env's `SAMPLE_FIELDS`: the match's phase, clock, overtime and score, the goal scored during the step, whether ball cam is on, the ball, and for each car its position, velocity, rotation, boost, ground and supersonic flags, demolition, the touches and demolitions of the step, and the controls it obeyed. Every frame is stored, about 48 KB each at 1024x576 (170 MB a minute). The block's `block_end` record carries the end state and the env's provenance: package versions, the collision meshes' SHA-256s, the tick rates, the OpenGL renderer and the camera and controller settings. A block replays bit for bit from `episode_seeds` and the actions on a machine where those are the same.
