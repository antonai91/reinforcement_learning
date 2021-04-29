import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import random_uniform
import sys
sys.path.append("../src")
from config import *

class Critic(tf.keras.Model):
    def __init__(self, name, hidden_0=CRITIC_HIDDEN_0, hidden_1=CRITIC_HIDDEN_1):
            
        super(Critic, self).__init__()
        
        self.hidden_0 = hidden_0
        self.hidden_1 = hidden_1

        self.net_name = name

        self.dense_0 = Dense(self.hidden_0, activation='relu')
        self.dense_1 = Dense(self.hidden_1, activation='relu')
        self.q_value = Dense(1, activation=None)
    
    def call(self, state, actors_actions):
        state_action_value = self.dense_0(tf.concat([state, actors_actions], axis=1)) # multiple actions
        state_action_value = self.dense_1(state_action_value)

        q_value = self.q_value(state_action_value)

        return q_value

class Actor(tf.keras.Model):
    def __init__(self, name, actions_dim, hidden_0=ACTOR_HIDDEN_0, hidden_1=ACTOR_HIDDEN_1):
        super(Actor, self).__init__()
        self.hidden_0 = hidden_0
        self.hidden_1 = hidden_1
        self.actions_dim = actions_dim
        
        self.net_name = name

        self.dense_0 = Dense(self.hidden_0, activation='relu')
        self.dense_1 = Dense(self.hidden_1, activation='relu')
        self.policy = Dense(self.actions_dim, activation='sigmoid') # we want something beetween zero and one

    def call(self, state):
        x = self.dense_0(state)
        policy = self.dense_1(x)
        policy = self.policy(policy)
        return policy
