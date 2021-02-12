#!/usr/bin/python
import os

# Name of the Gym environment for the agent to learn & play
#ENV_NAME = 'LunarLanderContinuous-v2'
ENV_NAME = 'Pendulum-v0'

BATCH_SIZE = 64
GAMMA = 0.99
ACTOR_LR = 0.0005
CRITIC_LR = 0.001

TAU = 0.05 # For soft update the target network
TAU = 1 # For soft update the target network
MIN_SIZE_BUFFER = 5000 # Minimum size of the buffer to start learning, until then random actions
BUFFER_CAPACITY = 20000
MAX_EPISODES = 10000
STEPS_BETWEEN_UPDATE_BUFFER = 10
# Parameters for Ornstein–Uhlenbeck process
THETA=0.15
DT=1e-1

# Network parameters
ACTOR_HIDDEN_0 = 512
ACTOR_HIDDEN_1 = 256

CRITIC_HIDDEN_0 = 512
CRITIC_HIDDEN_1 = 256
CRITIC_HIDDEN_2 = 256
CRITIC_HIDDEN_3 = 128
