# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 11:02:18 2021

@author: Leon Jovanovic
"""
import gym
from agent import Agent
import atari_wrappers
import torch

import wandb
import argparse

import random

# ---------------------------------Arguments-----------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="set the name of the experiment")
parser.add_argument("--wandb", help="sets wandb to be true", action="store_true")
args = parser.parse_args()

# ---------------------------------Parameters----------------------------------

DQN_HYPERPARAMS = {
    'eps_start': 1,
    'eps_end': 0.02,
    'eps_decay': 10 ** 5,
    'buffer_size': 15000,
    'buffer_minimum': 10001,
    'learning_rate': 5e-5,
    'gamma': 0.99,
    'n_iter_update_nn': 1000,
    'multi_step': 2,
    'double_dqn': False,
    'dueling': False
}

ENV_NAME = "PongNoFrameskip-v4"
RECORD = False
MAX_GAMES = 500
DEVICE = 'cuda'
BATCH_SIZE = 32

WANDB = args.wandb

EXPERIMENT_NAME = args.name

# ------------------------Create enviroment and agent--------------------------
# Initialize WANDB
run = wandb.init(project="modified_hirl", name=EXPERIMENT_NAME, config=DQN_HYPERPARAMS) if WANDB else None

# Create / Initialize Environment
env = atari_wrappers.make_env("PongNoFrameskip-v4")  # gym.make("PongNoFrameskip-v4")

env_seed = random.randint(0, 10000)
env.seed(env_seed)

obs = env.reset()

# Create Agent
agent = Agent(env, hyperparameters=DQN_HYPERPARAMS, device=DEVICE, max_games=MAX_GAMES, wandb=run)

# --------------------------------Learning-------------------------------------
num_games = 0
while num_games < MAX_GAMES:
    # Select one action with e-greedy policy and observe s,a,s',r and done
    action = agent.select_eps_greedy_action(obs)
    new_obs, reward, done, catastrophe = env.step(action)

    # Catastrophe implementation
    human_action = int(action)
    human_reward = reward
    if catastrophe:
        human_action = 2
        human_reward = -100
        agent.num_catasrophe += 1

    # Add s, a, s', r, a_H, r_H to buffer B
    agent.add_to_buffer(obs, action, new_obs, reward, done, human_action, human_reward)
    # Sample a mini-batch from buffer B if B is large enough. If not skip until it is.
    # Use that mini-batch to improve NN value function approximation
    agent.sample_and_improve(BATCH_SIZE)

    obs = new_obs
    if done:
        num_games = num_games + 1
        agent.print_info()
        agent.reset_parameters()

        env_seed = random.randint(0, 10000)
        env.seed(env_seed)

        obs = env.reset()
        
    
savename = "./checkpoints/{EXPERIMENT_NAME}/dqn_" + str(num_games) + ".pth"
torch.save(agent.agent_control.moving_nn.state_dict(), savename)