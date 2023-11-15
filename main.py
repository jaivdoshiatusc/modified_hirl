# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 11:02:18 2021

@author: Leon Jovanovic
"""
import gym
from agent import Agent
import atari_wrappers
import torch
from torch.utils.tensorboard import SummaryWriter
import time

import wandb

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

# For TensorBoard
SUMMARY_WRITER = False
WANDB = True
LOG_DIR = 'content/runs'
name = 'DQN Multi-step=%d,Double=%r,Dueling=%r' % (DQN_HYPERPARAMS['multi_step'], DQN_HYPERPARAMS['double_dqn'], DQN_HYPERPARAMS['dueling'])
# For Telegram
TG_BOT = False

# ------------------------Create enviroment and agent--------------------------
env = atari_wrappers.make_env("PongNoFrameskip-v4")  # gym.make("PongNoFrameskip-v4")
# For recording few seelcted episodes. 'force' means overwriting earlier recordings
if RECORD:
    env = gym.wrappers.Monitor(env, "main-" + ENV_NAME, force=True)
obs = env.reset()
# Create TensorBoard writer that will create graphs
writer = SummaryWriter(log_dir=LOG_DIR + '/' + name + str(time.time())) if SUMMARY_WRITER else None
run = wandb.init() if WANDB else None
# Create agent that will learn
agent = Agent(env, hyperparameters=DQN_HYPERPARAMS, device=DEVICE, writer=writer, max_games=MAX_GAMES, tg_bot=TG_BOT, wandb=run)
# --------------------------------Learning-------------------------------------
num_games = 0
while num_games < MAX_GAMES:
    # Select one action with e-greedy policy and observe s,a,s',r and done
    action = agent.select_eps_greedy_action(obs)
    new_obs, reward, done, catastrophe = env.step(action)

    # catastrophe implementation
    human_action = int(action)
    human_reward = reward
    if catastrophe:
        human_action = 2
        human_reward = -100
        agent.num_catasrophe += 1

    # import ipdb; ipdb.set_trace()

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
        obs = env.reset()
    
savename = "./checkpoints/dqn_" + str(num_games) + ".pth"
torch.save(agent.agent_control.moving_nn.state_dict(), savename)

writer.close()
gym.wrappers.Monitor.close(env)

# !tensorboard --logdir="D:\Users\Leon Jovanovic\Documents\Reinforcement Learning\reinforcement-learning-atari-pong\content\runs" --host=127.0.0.1
