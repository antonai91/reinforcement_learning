#!/usr/bin/python

# Name of the Gym environment for the agent to learn & play
ENV_NAME = 'PongDeterministic-v4'
NO_OP_STEPS = 20 # in evaluation mode, number of 1 (fire) action to add randomness

PATH_LOAD_MODEL = "../model/save_agent_202101040929_00093744"
PATH_SAVE_MODEL = "../model"
LOAD_REPLAY_BUFFER = False

CLIP_REWARD = True                # Any positive reward is +1, and negative reward is -1, 0 is unchanged

MAX_FRAMES = 30000000             # Total number of frames the agent sees
MAX_EPISODE_LENGTH = 18000        # Equivalent of 5 minutes of gameplay at 60 frames per second (18000 frames / 60 fps = 5 minutes)
EVAL_FREQUENCY = 100000           # Number of frames between evaluations
EVAL_STEPS = 10000                # Number of frames for one evaluation

DISCOUNT_FACTOR = 0.99            # Gamma, how much to discount future rewards
REPLAY_MEMORY_START_SIZE = 50000  # Number of completely random actions,
MEMORY_SIZE = 1000000             # Number of transitions stored in the replay memory

UPDATE_FREQ = 4                   # Every four actions a gradient descend step is performed
NETW_UPDATE_FREQ = 10000          # Number of chosen actions between updating the target network.
                                  # According to Mnih et al. 2015 this is measured in the number of
                                  # parameter updates (every four actions), however, in the
                                  # DeepMind code, it is clearly measured in the number
                                  # of actions the agent choses
                                  # before the agent starts learning

INPUT_SHAPE = (84, 84)            # Size of the preprocessed input frame. With the current model architecture, anything below ~80 won't work.
BATCH_SIZE = 32                   # Batch size
LEARNING_RATE = 0.00025
HIDDEN = 1024                    # Number of filters in the final convolutional layer. The output
                                 # has the shape (1,1,1024) which is split into two streams. Both
                                 # the advantage stream and value stream have the shape
                                 # (1,1,512). This is slightly different from the original
                                 # implementation but tests I did with the environment Pong
                                 # have shown that this way the score increases more quickly
