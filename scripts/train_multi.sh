#!/bin/bash

python ../src/debug.py \
  --exp_name $1 \
  --num_episodes 500 \
  --num_hunters 10 \
  --num_targets 5 \
  --save_frequency 100 \
  --ifrender false \
  --visualizelaser false \
  --update_freq 10 \
  --batch_size 256 \
  --lr 5e-4 