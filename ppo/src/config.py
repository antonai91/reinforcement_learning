#!/usr/bin/python

# Name of the Gym environment for the agent to learn & play
ENV_NAME = 'PongDeterministic-v4'

PATH_SAVE_MODEL = "../model/"
PATH_LOAD_MODEL = None

HIDDEN = 1024
LR = 0.005
GAMMA= 0.99
VALUE_C = 0.3
ENTROPY_C = 1e-3
CLIP_RATIO = 0.2

MAX_UPDATES=10000000
BATCH_SIZE=64
INPUT_SHAPE=(84, 84)
HISTORY_LENGHT=4
NUM_ACTIONS=6


CONFIG_WANDB = dict (
  learning_rate = LR,
  batch_size = 64,
  architecture = "ppo",
  infra = "Ubuntu"
)
