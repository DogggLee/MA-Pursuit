#!/bin/bash

python ../src/test_model.py \
  --model_dir $1 \
  --num_hunters 6 \
  --num_targets 2 \
  --max_steps 150 \
  --num_test_episodes 1 \
  --seed 5 \
  --ifrender true \
  --visualizelaser true 