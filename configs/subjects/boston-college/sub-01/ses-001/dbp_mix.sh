#!/bin/sh
# fmri-gym session: one line per run, in order.
set -e
SES=${1:-001}
uv run fmri-play --curriculum configs/curricula/crafter/L4.json --subject sub-01 --ses "$SES" --run 1 --rig example/laptop
uv run fmri-play --curriculum configs/curricula/coom/pitfall.json --subject sub-01 --ses "$SES" --run 1 --rig example/laptop
uv run fmri-play --curriculum configs/curricula/minihack/room5x5.json --subject sub-01 --ses "$SES" --run 1 --rig example/laptop
uv run fmri-play --curriculum configs/curricula/rushhour/easy.json --subject sub-01 --ses "$SES" --run 1 --rig example/laptop
uv run fmri-play --curriculum configs/curricula/baba/make_win.json --subject sub-01 --ses "$SES" --run 1 --rig example/laptop
uv run fmri-play --curriculum configs/curricula/vizdoom/defend_center.json --subject sub-01 --ses "$SES" --run 1 --rig example/laptop
uv run fmri-play --curriculum configs/curricula/aigamestore/game1.json --subject sub-01 --ses "$SES" --run 1 --rig example/laptop
uv run fmri-play --curriculum configs/curricula/coom/run_and_gun.json --subject sub-01 --ses "$SES" --run 1 --rig example/laptop
uv run fmri-play --curriculum configs/curricula/minihack/mazewalk9x9.json --subject sub-01 --ses "$SES" --run 1 --rig example/laptop
uv run fmri-play --curriculum configs/curricula/aigamestore/game2.json --subject sub-01 --ses "$SES" --run 1 --rig example/laptop
uv run fmri-play --curriculum configs/curricula/coom/chainsaw.json --subject sub-01 --ses "$SES" --run 1 --rig example/laptop
