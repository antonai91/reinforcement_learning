#!/usr/bin/python
import os

#******************************
#******** Enviroment **********
#******************************

ENV_NAME = 'simple_adversary'


PATH_SAVE_MODEL = "../model/"
PATH_LOAD_FOLDER = None


BUFFER_CAPACITY = 1000000
BATCH_SIZE = 64
MIN_SIZE_BUFFER = 1024

CRITIC_HIDDEN_0 = 256
CRITIC_HIDDEN_1 = 128
ACTOR_HIDDEN_0 = 256 
ACTOR_HIDDEN_1 = 128

ACTOR_LR = 0.01
CRITIC_LR = 0.01
GAMMA = 0.95
TAU = 0.01

MAX_GAMES = 1000000
MAX_STEPS = 50
EVALUATION_FREQUENCY = 500
SAVE_FREQUENCY = 5000