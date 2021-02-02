#!/usr/bin/python

# Name of the Gym environment for the agent to learn & play
ENV_NAME = 'PongDeterministic-v4'
ENV_NAME = 'CartPole-v0'

HIDDEN = 1024

PATH_SAVE_MODEL = "../model"

PATH_LOAD_MODEL = None

CONFIG_WANDB = dict (
  learning_rate = 1e-3,
  batch_size = 64,
  architecture = "a2c_ppo",
  infra = "Ubuntu"
)
