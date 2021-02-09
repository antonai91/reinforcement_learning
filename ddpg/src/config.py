#!/usr/bin/python
import os

# Name of the Gym environment for the agent to learn & play
ENV_NAME = 'LunarLanderContinuous-v2'

BATCH_SIZE = 64
GAMMA = 0.99
ACTOR_LR = 0.0005
CRITIC_LR = 0.001
TAU = 0.05 # For soft update the target network
MIN_SIZE_BUFFER = 2000 # Minimum size of the buffer to start learning, until then random actions
BUFFER_CAPACITY = 20000
# Parameters for Ornstein–Uhlenbeck process
THETA=0.15
DT=1e-1

# Network parameters
ACTOR_HIDDEN_0 = 32
ACTOR_HIDDEN_1 = 32

CRITIC_HIDDEN_0 = 64
CRITIC_HIDDEN_1 = 32
CRITIC_HIDDEN_2 = 32
CRITIC_HIDDEN_3 = 16
