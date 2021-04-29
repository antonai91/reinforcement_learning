import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers as opt
import random
import time
import json
import os
import sys
sys.path.append("../src")
from config import *
from make_env import *
from replay_buffer import *
from networks import *

class Agent:
    def __init__(self, env, n_agent, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, gamma=GAMMA, tau=TAU):
        
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        
        self.actor_dims = env.observation_space[n_agent].shape[0]
        self.n_actions = env.action_space[n_agent].n
        
        self.agent_name = "agent_number_{}".format(n_agent)
        
        self.actor = Actor("actor_" + self.agent_name, self.n_actions)
        self.critic = Critic("critic_" + self.agent_name)
        self.target_actor = Actor("target_actor_" + self.agent_name, self.n_actions)
        self.target_critic = Critic("critic_" + self.agent_name)
        
        self.actor.compile(optimizer=opt.Adam(learning_rate=actor_lr))
        self.critic.compile(optimizer=opt.Adam(learning_rate=critic_lr))
        self.target_actor.compile(optimizer=opt.Adam(learning_rate=actor_lr))
        self.target_critic.compile(optimizer=opt.Adam(learning_rate=critic_lr))
        
        actor_weights = self.actor.get_weights()
        critic_weights = self.critic.get_weights()
        
        self.target_actor.set_weights(actor_weights)
        self.target_critic.set_weights(critic_weights)
        
    def update_target_networks(self, tau):
        actor_weights = self.actor.weights
        target_actor_weights = self.target_actor.weights
        for index in range(len(actor_weights)):
            target_actor_weights[index] = tau * actor_weights[index] + (1 - tau) * target_actor_weights[index]

        self.target_actor.set_weights(target_actor_weights)
        
        critic_weights = self.critic.weights
        target_critic_weights = self.target_critic.weights
    
        for index in range(len(critic_weights)):
            target_critic_weights[index] = tau * critic_weights[index] + (1 - tau) * target_critic_weights[index]

        self.target_critic.set_weights(target_critic_weights)
        
    def get_actions(self, actor_states):
        noise = tf.random.uniform(shape=[self.n_actions])
        actions = self.actor(actor_states)
        actions = actions + noise

        return actions.numpy()[0]
    
    def save(self, path_save):
        self.actor.save_weights(f"{path_save}/{self.actor.net_name}.h5")
        self.target_actor.save_weights(f"{path_save}/{self.target_actor.net_name}.h5")
        self.critic.save_weights(f"{path_save}/{self.critic.net_name}.h5")
        self.target_critic.save_weights(f"{path_save}/{self.target_critic.net_name}.h5")
        
    def load(self, path_load):
        self.actor.load_weights(f"{path_load}/{self.actor.net_name}.h5")
        self.target_actor.load_weights(f"{path_load}/{self.target_actor.net_name}.h5")
        self.critic.load_weights(f"{path_load}/{self.critic.net_name}.h5")
        self.target_critic.load_weights(f"{path_load}/{self.target_critic.net_name}.h5")
