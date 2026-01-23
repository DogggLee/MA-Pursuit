import argparse
import numpy as np
import torch
import os
import csv
import warnings
from datetime import datetime
from path import Path

from src.MultiTargetEnv import MultiTarEnv, set_global_seeds
from src.MATD3 import MATD3Agent
from src.replaybuffer import ReplayBuffer
from src.utils import load_config, generate_exp_dirname, calc_dim

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('config_path', type=str, default="config/base.yaml", help='Exp config file path')
parser.add_argument('--dump_root', type=str, help='Checkpoint path')
parser.add_argument('--checkpoint', type=str, help='Checkpoint path')

def main(args):
    config = load_config(args.config_path)

    

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
