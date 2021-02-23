import sys
sys.path.append("../src")
from replay_buffer import *
from config import *
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import random_uniform
import tensorflow_probability as tfp

class Critic(tf.keras.Model):
    def __init__(self, name, hidden_0=CRITIC_HIDDEN_0, hidden_1=CRITIC_HIDDEN_1):
        super(Critic, self).__init__()
        
        self.hidden_0 = hidden_0
        self.hidden_1 = hidden_1

        self.net_name = name

        self.dense_0 = Dense(self.hidden_0, activation='relu')
        self.dense_1 = Dense(self.hidden_1, activation='relu')
        self.q_value = Dense(1, activation=None)

    def call(self, state, action):
        state_action_value = self.dense_0(tf.concat([state, action], axis=1))
        state_action_value = self.dense_1(state_action_value)

        q_value = self.q_value(state_action_value)

        return q_value
    
class CriticValue(tf.keras.Model):
    def __init__(self, name, hidden_0=CRITIC_HIDDEN_0, hidden_1=CRITIC_HIDDEN_1):
        super(CriticValue, self).__init__()
        
        self.hidden_0 = hidden_0
        self.hidden_1 = hidden_1

        self.net_name = name

        self.dense_0 = Dense(self.hidden_0, activation='relu')
        self.dense_1 = Dense(self.hidden_1, activation='relu')
        self.value = Dense(1, activation=None)

    def call(self, state):
        state_value = self.dense_0(state)
        state_value = self.dense_1(state_value)

        value = self.value(state_value)

        return value

class Actor(tf.keras.Model):
    def __init__(self, name, actions_dim, upper_bound, hidden_0=CRITIC_HIDDEN_0, hidden_1=CRITIC_HIDDEN_1, noise=NOISE, log_std_min=LOG_STD_MIN, log_std_max=LOG_STD_MAX):
        super(Actor, self).__init__()
        self.hidden_0 = hidden_0
        self.hidden_1 = hidden_1
        self.actions_dim = actions_dim
        self.upper_bound = upper_bound
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.noise = noise
        
        self.net_name = name

        self.dense_0 = Dense(self.hidden_0, activation='relu')
        self.dense_1 = Dense(self.hidden_1, activation='relu')
        self.mean = Dense(self.actions_dim, activation=None)
        self.log_std = Dense(self.actions_dim, activation=None)

    def call(self, state):
        policy = self.dense_0(state)
        policy = self.dense_1(policy)
        mean = self.mean(policy)
        log_std = self.log_std(policy)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std
    
    def evaluate(self, state, reparameterization=False):
        mean, log_std = self.call(state)
        std = tf.exp(log_std)
        normal_mean_std = tfp.distributions.Normal(mean, std)
        
        if reparameterization:
            actions = normal_mean_std.sample()
        else:
            actions = normal_mean_std.sample()
            
        actions = tf.math.tanh(actions)*self.upper_bound
        
        log_probs = normal_mean_std.log_prob(actions) - tf.math.log(1 - tf.math.pow(actions, 2) + self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)
        
        return actions, log_probs
