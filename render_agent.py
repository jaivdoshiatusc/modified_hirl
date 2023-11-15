from agent import Agent
import atari_wrappers
import torch
import imageio

# ---------------------------------Parameters----------------------------------

DQN_HYPERPARAMS = {
    'eps_start': 0.02,
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
MAX_GAMES = 500
DEVICE = 'cuda'
BATCH_SIZE = 32

# CHANGE CHECKPOINT PATH!
checkpoint_path = "/home/jaiv/modified_dqn_pong/checkpoints/dqn_500.pth"

# ------------------------Create enviroment and agent--------------------------

# Set render=True to render the Red Line!
env = atari_wrappers.make_env("PongNoFrameskip-v4", render=True)

agent = Agent(env, hyperparameters=DQN_HYPERPARAMS, device=DEVICE, max_games=MAX_GAMES, wandb=None)
agent.agent_control.moving_nn.load_state_dict(torch.load(checkpoint_path))

# -------------------------------Render Agent-----------------------------------
'''
The following implementation gets the weights of the final model from the checkpoint path and renders the agent for
10000 frames (and resets completed games). 

Since there was no dedicated "act" implementation, I set the eps_start and eps_start to be equal to the eps_end value 
from the training code so that it uses the model action.

The render code has been modified to include the red line to indicate a catastrophe.
'''

obs = env.reset()
frames = []
score = 0
catastrophes = 0

for i in range(10000):
    action = agent.select_eps_greedy_action(obs)
    new_obs, reward, done, catastrophe = env.step(action)

    frame = env.render()
    frames.append(frame)

    if done == True:
        env.reset()

    obs = new_obs
    score += reward
    if catastrophe:
        catastrophes =+ 1


imageio.mimsave(f'./images/initial_line.gif', frames, duration=30)
print(f'Score: {score}\tNumber of Catastrophes: {catastrophes}')
