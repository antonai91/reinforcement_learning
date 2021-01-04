#!/usr/bin/python

# Name of the Gym environment for the agent to learn & play
ENV_NAME = 'PongDeterministic-v4'
NO_OP_STEPS = 20 # in evaluation mode, number of 1 (fire) action to add randomness
HIDDEN = 1024    # Number of filters in the final convolutional layer. The output 
                 # has the shape (1,1,1024) which is split into two streams. Both 
                 # the advantage stream and value stream have the shape 
                 # (1,1,512). This is slightly different from the original 
                 # implementation, followed the test from https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb