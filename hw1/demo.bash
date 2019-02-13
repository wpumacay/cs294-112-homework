#!/bin/bash
set -eux
## for e in Humanoid-v2 Walker2d-v2 Ant-v2 Hopper-v2 HalfCheetah-v2 Reacher-v2
#for e in Walker2d-v2
for e in HalfCheetah-v2
do
    python run_expert.py experts/$e.pkl $e --num_rollouts=2000
done
