#!/usr/bin/python

# Name of the Gym environment for the agent to learn & play
ENV_NAME = 'PongDeterministic-v4'

HIDDEN = 1024

PATH_SAVE_MODEL = "../model"

PATH_LOAD_MODEL = "../model/save_agent_202101121017_00080000"

CONFIG_WANDB = dict (
  learning_rate = 1e-3,
  batch_size = 64,
  architecture = "a2c",
  infra = "Ubuntu"
)
