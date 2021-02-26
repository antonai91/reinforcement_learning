import sys
sys.path.append("../src")
from replay_buffer import *
from config import *
import tensorflow as tf
from tensorflow.keras.layers import Dense
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
        value = self.dense_0(state)
        value = self.dense_1(value)

        value = self.value(value)

        return value

class Actor(tf.keras.Model):
    def __init__(self, name, upper_bound, actions_dim, hidden_0=CRITIC_HIDDEN_0, hidden_1=CRITIC_HIDDEN_1, epsilon=EPSILON):
        super(Actor, self).__init__()
        self.hidden_0 = hidden_0
        self.hidden_1 = hidden_1
        self.actions_dim = actions_dim
        self.net_name = name
        self.upper_bound = upper_bound
        self.epsilon = epsilon

        self.dense_0 = Dense(self.hidden_0, activation='relu')
        self.dense_1 = Dense(self.hidden_1, activation='relu')
        self.mean = Dense(self.actions_dim, activation=None)
        self.std = Dense(self.actions_dim, activation=None)

    def call(self, state):
        policy = self.dense_0(state)
        policy = self.dense_1(policy)

        mean = self.mean(policy)
        std = self.std(policy)

        std = tf.clip_by_value(std, self.epsilon, 1)

        return mean, std

    def get_action_log_probs(self, state, reparameterization_trick=True):
        mean, std = self.call(state)
        normal_distr = tfp.distributions.Normal(mean, std)
        # Reparameterization trick
        z = tf.random.normal(shape=mean.shape, mean=0., stddev=1.)

        if reparameterization_trick:
            actions = mean + std * z
        else:
            actions = normal_distr.sample()

        action = tf.math.tanh(actions) * self.upper_bound
        log_probs = normal_distr.log_prob(actions) - tf.math.log(1 - tf.math.pow(action,2) + self.epsilon)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        return action, log_probs
