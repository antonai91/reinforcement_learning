#!/usr/bin/python
import os

# Name of the Gym environment for the agent to learn & play
ENV_NAME = 'PongDeterministic-v4'

INPUT_SHAPE = (84, 84, 4)
BATCH_SIZE = 64
N_UPDATES = 1000000
HIDDEN = 1024
#LR = 1e-4
LR = 1e-5
GAMMA = 0.99
VALUE_C = 0.5
ENTROPY_C = 1e-4
#CLIP_RATIO = 0.2
CLIP_RATIO = 0.1
STD_ADV = True
AGENT = "PPO"

PATH_SAVE_MODEL = f"../model/{AGENT.lower()}"

PATH_LOAD_MODEL = "../model/ppo/save_agent_202102090741/model.tf/"
#PATH_LOAD_MODEL = None

CONFIG_WANDB = dict (
  learning_rate = LR,
  batch_size = BATCH_SIZE,
  agent = AGENT,
  operating_system = os.name
)
