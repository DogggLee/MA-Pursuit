import argparse
import numpy as np
import torch
import os
import csv
import warnings
from datetime import datetime
from path import Path
import shutil

from src.MAPursuitEnv import MAPursuitEnv, set_global_seeds
from src.MATD3 import MATD3Agent
from src.replaybuffer import ReplayBuffer
from src.utils import load_config, generate_exp_dirname, calc_dim

parser = argparse.ArgumentParser()
parser.add_argument('config_path', type=str, default="config/base.yaml", help='Exp config file path')
parser.add_argument('--dump_root', type=str, default="checkpoints", help='Checkpoint path')
parser.add_argument('--checkpoint', type=str, help='Checkpoint path')
parser.add_argument('--render', action="store_true", default=True, help='whether to render the environment')
parser.add_argument('--visualize_laser', action="store_true", default=True, help='whether to visualize laser')
parser.add_argument('--fill_laser_range', action="store_true", default=True, help='whether to visualize laser')

def test_render(args, config):
    env = MAPursuitEnv(config, args.visualize_laser)

    env.render("Test visualization", pause=-1)


if __name__ == "__main__":
    args = parser.parse_args()

    args.config_path = Path(args.config_path)
    config = load_config(args.config_path)

    test_render(args, config)

