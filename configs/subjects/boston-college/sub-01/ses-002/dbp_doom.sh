#!/bin/sh
# fmri-gym session: one line per run, in order.
set -e
SES=${1:-002}
uv run fmri-play --curriculum configs/curricula/coom/pitfall.json --subject sub-01 --ses "$SES" --run 1 --rig example/laptop
uv run fmri-play --curriculum configs/curricula/coom/arms_dealer.json --subject sub-01 --ses "$SES" --run 1 --rig example/laptop
uv run fmri-play --curriculum configs/curricula/coom/floor_is_lava.json --subject sub-01 --ses "$SES" --run 1 --rig example/laptop
uv run fmri-play --curriculum configs/curricula/coom/hide_and_seek.json --subject sub-01 --ses "$SES" --run 1 --rig example/laptop
uv run fmri-play --curriculum configs/curricula/coom/chainsaw.json --subject sub-01 --ses "$SES" --run 1 --rig example/laptop
uv run fmri-play --curriculum configs/curricula/coom/raise_the_roof.json --subject sub-01 --ses "$SES" --run 1 --rig example/laptop
uv run fmri-play --curriculum configs/curricula/coom/run_and_gun.json --subject sub-01 --ses "$SES" --run 1 --rig example/laptop
uv run fmri-play --curriculum configs/curricula/coom/health_gathering.json --subject sub-01 --ses "$SES" --run 1 --rig example/laptop
uv run fmri-play --curriculum configs/curricula/coom/parkour.json --subject sub-01 --ses "$SES" --run 1 --rig example/laptop
uv run fmri-play --curriculum configs/curricula/vizdoom/defend_center.json --subject sub-01 --ses "$SES" --run 1 --rig example/laptop
uv run fmri-play --curriculum configs/curricula/vizdoom/deadly_corridor.json --subject sub-01 --ses "$SES" --run 1 --rig example/laptop
