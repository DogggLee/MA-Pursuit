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

import matplotlib.pyplot as plt

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('config_path', type=str, default="config/base.yaml", help='Exp config file path')
parser.add_argument('--dump_root', type=str, default="checkpoints", help='Checkpoint path')
parser.add_argument('--checkpoint', type=str, help='Checkpoint path')
parser.add_argument('--render', action="store_true", default=False, help='whether to render the environment')
parser.add_argument('--visualize_laser', action="store_true", default=True, help='whether to visualize laser')

def train_split(args):
    args.config_path = Path(args.config_path)
    config = load_config(args.config_path)

    exp_dirname = generate_exp_dirname(config)

    exp_dir = Path(args.dump_root) / exp_dirname
    exp_model_dir = exp_dir / "model"
    exp_model_dir.makedirs_p()
    shutil.copy(args.config_path, exp_dir/ args.config_path.name)
    print(f"New experiment, save to {exp_dir}")

    set_global_seeds(config.Base.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    h_dim, t_dim = calc_dim(config)
    env = MAPursuitEnv(config, h_dim, t_dim, args.visualize_laser)

    # Initialize agents for hunters and targets
    hunters = [MATD3Agent(obs_dim=h_dim,
                          action_dim=config.Hunter.action_dim,
                          lr=config.Train.lr,
                          gamma=config.Train.gamma,
                          tau=config.Train.tau,
                          noise_std=config.Train.noise_std,
                          device=device,
                          iforthogonalize=config.Model.hunter_orthogonalize,
                          noise_clip=config.Train.noise_clip,
                          max_acc=config.Hunter.max_acc,
                          if_lr_decay=config.Model.hunter_lrdecay,
                          total_episodes=config.Train.num_episodes) for _ in range(env.num_hunter)]

    targets = [MATD3Agent(obs_dim=t_dim,
                          action_dim=config.Target.action_dim,
                          lr=config.Train.lr,
                          gamma=config.Train.gamma,
                          tau=config.Train.tau,
                          noise_std=config.Train.noise_std,
                          device=device,
                          iforthogonalize=config.Model.target_orthogonalize,
                          noise_clip=config.Train.noise_clip,
                          max_acc=config.Target.max_acc,
                          if_lr_decay=config.Model.target_lrdecay,
                          total_episodes=config.Train.num_episodes) for _ in range(env.num_target)]


    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading models from checkpoint: {args.checkpoint}")
        try:
            for i, hunter in enumerate(hunters):
                hunter.load_model(args.checkpoint, agent_id=i, agent_type='hunter')
            for i, target in enumerate(targets):
                target.load_model(args.checkpoint, agent_id=i, agent_type='target')
            print("Successfully loaded models from checkpoint")
        except Exception as e:
            print(f"Error loading models from checkpoint: {e}")
            print("Training will start with newly initialized models")
    
    # Initialize replay buffers for hunters & targets
    hunters_buffer = ReplayBuffer(max_size=config.Train.buffer_size,
                                 obs_dim=h_dim,
                                 action_dim=config.Hunter.action_dim)

    targets_buffer = ReplayBuffer(max_size=config.Train.buffer_size,
                                 obs_dim=t_dim,
                                 action_dim=config.Target.action_dim)

    # initialize CSV file
    rewards_csv_path = exp_dir / "rewards.csv"
    with open(rewards_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["episode", "total_reward_hunters", "total_reward_targets"])
    
    update_counter = 0
    score_threshold = config.Train.ckp_score_threshold
    for episode in range(config.Train.num_episodes):
        h_obs, t_obs = env.reset()
        episode_rewards_hunters = np.zeros(env.num_hunter)
        episode_rewards_targets = np.zeros(env.num_target)
        done = False
        current_step = 1

        # Play 直至episode终止
        while (not done) and (current_step <= config.Train.max_steps):
            actions_hunters = []
            actions_targets = []

            # hunters choose action
            for i, hunter in enumerate(hunters):
                action = hunter.select_action(h_obs[i])
                actions_hunters.append(action)

            # targets choose action
            for i, target in enumerate(targets):
                action = target.select_action(t_obs[i])
                actions_targets.append(action)
            
            # concatenate all actions
            actions = actions_hunters + actions_targets
            # execute all actions & interact with env
            h_next_obs, t_next_obs, rewards, dones = env.step(actions)
            
            if args.render:
                env.render()
            current_step += 1

            rewards_hunters = rewards[:env.num_hunter]
            rewards_targets = rewards[env.num_hunter:]
            dones_hunters = dones[:env.num_hunter]
            dones_targets = dones[env.num_hunter:]

            # store transitions in Buffer
            for i in range(env.num_hunter):
                hunters_buffer.store_transition(h_obs[i], actions_hunters[i], rewards_hunters[i], h_next_obs[i], dones_hunters[i])

            for i in range(env.num_target):
                targets_buffer.store_transition(t_obs[i], actions_targets[i], rewards_targets[i], t_next_obs[i], dones_targets[i])

            episode_rewards_hunters += rewards_hunters
            episode_rewards_targets += rewards_targets

            h_obs = h_next_obs
            t_obs = t_next_obs

            done = all(dones)

            update_counter += 1
            if update_counter % config.Train.update_freq == 0:
                if hunters_buffer.size() >= config.Train.min_buffer_size:
                    for _ in range(config.Train.update_iterations):
                        batch = hunters_buffer.sample(config.Train.batch_size)
                        for hunter in hunters:
                            hunter.update(batch)
                if targets_buffer.size() >= config.Train.min_buffer_size:
                    for _ in range(config.Train.update_iterations):
                        batch = targets_buffer.sample(config.Train.batch_size)
                        for target in targets:
                            target.update(batch)

        total_reward_hunters = episode_rewards_hunters.sum()
        total_reward_targets = episode_rewards_targets.sum()
        print(f"Episode {episode}/{config.Train.num_episodes}, "
              f"Total Reward Hunters: {total_reward_hunters:.2f}, "
              f"Total Reward Targets: {total_reward_targets:.2f}")

        with open(rewards_csv_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([episode, total_reward_hunters, total_reward_targets])

        # save model
        should_save = False
        save_reason = ""
        if episode % config.Train.ckp_save_interval == 0 and episode > 0:
            should_save = True
            save_reason = f"ckp_{config.Train.ckp_save_interval}"
        if total_reward_hunters > score_threshold:
            score_threshold = total_reward_hunters
            should_save = True
            save_reason = f"score_{total_reward_hunters:.0f}"
        
        if should_save:
            save_dir = exp_model_dir / f"{save_reason}"

            for i, hunter in enumerate(hunters):
                hunter.save_model(save_dir, agent_id=i, agent_type='hunter')

            for i, target in enumerate(targets):
                target.save_model(save_dir, agent_id=i, agent_type='target')

            print(f"Models saved at episode {episode} in {save_dir}")


if __name__ == '__main__':
    args = parser.parse_args()
    train_split(args)
