import sys
sys.path.append("../src")
import tensorflow as tf
import numpy as np
import random
from config import *
from replay_buffer import *
from networks import *

class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_upper_bound = self.env.action_space.high[0]

        self.buffer = ReplayBuffer(self.env)

        self.actor = Actor(self.action_dim, self.action_upper_bound)
        self.critic = Critic(self.action_dim)
        
        self.target_actor = Actor(self.action_dim, self.action_upper_bound)
        self.target_critic = Critic(self.action_dim)
    
    def update_target_networks(self):
        actor_weights = self.actor.variables
        target_actor_weights = self.target_actor.variables
        critic_weights = self.critic.variables
        target_critic_weights = self.target_critic.variables
        _update_target(target_actor_weights, actor_weights, TAU)
        _update_target(target_critic_weights, critic_weights, TAU)        
    
    def _get_target_qvalues(self, rewards, q_values, dones):
        targets = rewards * dones + (1 - dones) * (GAMMA * q_values)
        return targets
    
    def _update_target(target_weights, weights, tau):
        for (t_w, w) in zip(target_weights, weights):
            tw.assign(t_w * (1 - tau) + w * tau)
            
    def _ornstein_uhlenbeck_process(self, x, theta=THETA, mu=0, dt=DT, std=0.2, dim=1):
        """
        Ornstein–Uhlenbeck process
        """
        return x + theta * (mu-x) * dt + std * np.sqrt(dt) * np.random.normal(size=dim)
