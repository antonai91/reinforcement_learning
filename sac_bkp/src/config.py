#!/usr/bin/python
import os

#******************************
#******** Enviroment **********
#******************************

#ENV_NAME = 'BipedalWalkerHardcore-v3'
#ENV_NAME = 'LunarLanderContinuous-v2'
ENV_NAME = 'Pendulum-v0'


PATH_SAVE = "../model/"
PATH_LOAD = None
#PATH_LOAD = "../model/save_agent_202102130731"

#******************************
#****** Replay Buffer *********
#******************************

BATCH_SIZE = 64
MIN_SIZE_BUFFER = 100 # Minimum size of the buffer to start learning, until then random actions
BUFFER_CAPACITY = 1000000

#******************************
#******** Networks ************
#******************************

ACTOR_HIDDEN_0 = 512
ACTOR_HIDDEN_1 = 256

CRITIC_HIDDEN_0 = 512
CRITIC_HIDDEN_1 = 256

LOG_STD_MIN = -20
LOG_STD_MAX = 2
NOISE = 1e-6

#******************************
#********** Agent *************
#******************************

GAMMA = 0.99
ACTOR_LR = 0.001
CRITIC_LR = 0.002
#ACTOR_LR = 0.01
#CRITIC_LR = 0.005

TAU = 0.05 # For soft update the target network

REWARD_SCALE = 2 # Scale factor for rewards

# Parameters for Ornstein–Uhlenbeck process
THETA=0.15
DT=1e-1

#******************************
#********** Main **************
#******************************

MAX_GAMES = 250
EVALUATION_FREQUENCY = 100
SAVE_FREQUENCY = 200