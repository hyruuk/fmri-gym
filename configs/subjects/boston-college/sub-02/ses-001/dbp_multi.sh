#!/bin/sh
# fmri-gym session: one line per run, in order.
set -e
SES=${1:-001}
uv run fmri-play --curriculum configs/curricula/coom/all.json --subject sub-02 --ses "$SES" --run 1 --rig configs/rigs/example/laptop.json --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/coom/all.json --subject sub-02 --ses "$SES" --run 2 --rig configs/rigs/example/laptop.json --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/vizdoom/take_cover_defend_line.json --subject sub-02 --ses "$SES" --run 1 --rig configs/rigs/example/laptop.json --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/vizdoom/take_cover_defend_line.json --subject sub-02 --ses "$SES" --run 2 --rig configs/rigs/example/laptop.json --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/aigamestore/game1.json --subject sub-02 --ses "$SES" --run 1 --rig configs/rigs/example/laptop.json --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/aigamestore/game1.json --subject sub-02 --ses "$SES" --run 2 --rig configs/rigs/example/laptop.json --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/aigamestore/levels/game4.json --subject sub-02 --ses "$SES" --run 1 --rig configs/rigs/example/laptop.json --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/aigamestore/levels/game4.json --subject sub-02 --ses "$SES" --run 2 --rig configs/rigs/example/laptop.json --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/aigamestore/game7.json --subject sub-02 --ses "$SES" --run 1 --rig configs/rigs/example/laptop.json --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/aigamestore/game7.json --subject sub-02 --ses "$SES" --run 2 --rig configs/rigs/example/laptop.json --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/aigamestore/game8.json --subject sub-02 --ses "$SES" --run 1 --rig configs/rigs/example/laptop.json --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/aigamestore/game8.json --subject sub-02 --ses "$SES" --run 2 --rig configs/rigs/example/laptop.json --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/crafter/levels/L1.json --subject sub-02 --ses "$SES" --run 1 --rig configs/rigs/example/laptop.json --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/crafter/levels/L1.json --subject sub-02 --ses "$SES" --run 2 --rig configs/rigs/example/laptop.json --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/rushhour/complete.json --subject sub-02 --ses "$SES" --run 1 --rig configs/rigs/example/laptop.json --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/rushhour/complete.json --subject sub-02 --ses "$SES" --run 2 --rig configs/rigs/example/laptop.json --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/baba_auto/all.json --subject sub-02 --ses "$SES" --run 1 --rig configs/rigs/example/laptop.json --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/baba_auto/all.json --subject sub-02 --ses "$SES" --run 2 --rig configs/rigs/example/laptop.json --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/minihack/all.json --subject sub-02 --ses "$SES" --run 1 --rig configs/rigs/example/laptop.json --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/minihack/all.json --subject sub-02 --ses "$SES" --run 2 --rig configs/rigs/example/laptop.json --duration 60 --loop --resume
uv run fmri-play --curriculum configs/curricula/stk/all.json --subject sub-02 --ses "$SES" --run 1 --rig configs/rigs/example/laptop.json --duration 60 --loop
uv run fmri-play --curriculum configs/curricula/stk/all.json --subject sub-02 --ses "$SES" --run 2 --rig configs/rigs/example/laptop.json --duration 60 --loop
