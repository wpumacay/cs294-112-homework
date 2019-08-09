#!/bin/bash
set -eux

for config in bc_humanoid bc_walker2d bc_ant bc_hopper bc_halfcheetah bc_reacher
do
    python trainer.py train --config="./configs/${config}.gin"
done
