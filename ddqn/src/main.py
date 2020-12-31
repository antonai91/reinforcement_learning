#!/usr/bin/python

import sys
sys.path.append("../src/")
import gym
import random
import time
from config import *
from dddqn_agent import *
from dueling_dqn_network import *
from pong_wrapper import *
from process_image import *
from replay_buffer import *
from utilities import *

pong_wrapper = PongWrapper(ENV_NAME, NO_OP_STEPS)
print("The environment has the following {} actions: {}".format(pong_wrapper.env.action_space.n, pong_wrapper.env.unwrapped.get_action_meanings()))

MAIN_DQN = build_q_network(pong_wrapper.env.action_space.n, LEARNING_RATE, input_shape=INPUT_SHAPE)
TARGET_DQN = build_q_network(pong_wrapper.env.action_space.n, input_shape=INPUT_SHAPE)

replay_buffer = ReplayBuffer(size=MEMORY_SIZE, input_shape=INPUT_SHAPE)
dddqn_agent = DDDQN_AGENT(MAIN_DQN, TARGET_DQN, replay_buffer, pong_wrapper.env.action_space.n, 
                    input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, 
                   replay_buffer_start_size=REPLAY_MEMORY_START_SIZE,
                   max_frames=MAX_FRAMES)

if PATH_LOAD_MODEL is not None:
    start_time = time.time()
    print('Loading model and info from the folder ', LOAD_FROM)
    info = dddqn_agent.load(LOAD_FROM, LOAD_REPLAY_BUFFER)

    # Apply information loaded from meta
    frame_number = info['frame_number']
    rewards = info['rewards']
    loss_list = info['loss_list']

    print(f'Loaded in {time.time() - start_time:.1f} seconds')
else:
    frame_number = 0
    rewards = []
    loss_list = []
