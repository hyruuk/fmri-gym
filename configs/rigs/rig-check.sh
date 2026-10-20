#!/bin/sh
# fmri-gym session: one line per run, in order.
set -e
SES=${1:-$(uv run fmri-ses --subject sub-rig --rig configs/rigs/example/laptop.json)}
# uv run fmri-play --curriculum configs/curricula/checks/rig-check.json --subject sub-rig --ses "$SES" --run 1 --rig configs/rigs/example/laptop.json
uv run fmri-play --curriculum configs/curricula/checks/rig-check-long.json --subject sub-rig --ses "$SES" --run 1 --rig configs/rigs/example/laptop.json
