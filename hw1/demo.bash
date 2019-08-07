#!/bin/bash
set -eux

NUM_ROLLOUTS=$1

for e in Humanoid-v2 Walker2d-v2 Ant-v2 Hopper-v2 HalfCheetah-v2 Reacher-v2
do
    python run_expert.py experts/$e.pkl $e --num_rollouts=$NUM_ROLLOUTS
done
