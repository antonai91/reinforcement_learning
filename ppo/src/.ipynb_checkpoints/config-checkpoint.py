#!/usr/bin/python

# Name of the Gym environment for the agent to learn & play
ENV_NAME = 'PongDeterministic-v4'

PATH_SAVE_MODEL = None
PATH_LOAD_MODEL = None

HIDDEN = 1024
LR = 1e-4
GAMMA= 0.99
VALUE_W = 0.5
ENTROPY_C = 1e-4
CLIP_RATIO = 0.1

MAX_UPDATES=100
BATCH_SIZE=64
INPUT_SHAPE=(84, 84)
HISTORY_LENGHT=4


CONFIG_WANDB = dict (
  learning_rate = LR,
  batch_size = 64,
  architecture = "ppo",
  infra = "Ubuntu"
)
