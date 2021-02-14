import sys
sys.path.append("../src")
import tensorflow as tf
from tensorflow.keras import optimizers as opt
import numpy as np
import random
import time
from config import *
from replay_buffer import *
from networks import *

class Agent:
    def __init__(self, env, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, gamma=GAMMA, max_size=BUFFER_CAPACITY, tau=TAU, path_save=PATH_SAVE, path_load=PATH_LOAD):
        
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(env, max_size)
        self.actions_dim = env.action_space.shape[0]
        self.upper_bound = env.action_space.high[0]
        self.lower_bound = env.action_space.low[0]
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.path_save = path_save
        self.path_load = path_load
        
        self.actor = Actor(name='actor', actions_dim=self.actions_dim, upper_bound=self.upper_bound)
        self.critic = Critic(name='critic')
        self.target_actor = Actor(name='target_actor', actions_dim=self.actions_dim, upper_bound=self.upper_bound)
        self.target_critic = Critic(name='target_critic')

        self.actor.compile(optimizer=opt.Adam(learning_rate=actor_lr))
        self.critic.compile(optimizer=opt.Adam(learning_rate=critic_lr))
        self.target_actor.compile(optimizer=opt.Adam(learning_rate=actor_lr))
        self.target_critic.compile(optimizer=opt.Adam(learning_rate=critic_lr))

        actor_weights = self.actor.get_weights()
        critic_weights = self.critic.get_weights()
        
        self.target_actor.set_weights(actor_weights)
        self.target_critic.set_weights(critic_weights)
        
        self.noise = np.zeros(self.actions_dim)

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
    
    def add_to_replay_buffer(self, state, action, reward, new_state, done):
        self.replay_buffer.add_record(state, action, reward, new_state, done)

    def save(self):
        date_now = time.strftime("%Y%m%d%H%M")
        if not os.path.isdir(f"{self.path_save}/save_agent_{date_now}"):
            os.makedirs(f"{self.path_save}/save_agent_{date_now}")
        self.actor.save_weights(f"{self.path_save}/save_agent_{date_now}/{self.actor.net_name}.h5")
        self.target_actor.save_weights(f"{self.path_save}/save_agent_{date_now}/{self.target_actor.net_name}.h5")
        self.critic.save_weights(f"{self.path_save}/save_agent_{date_now}/{self.critic.net_name}.h5")
        self.target_critic.save_weights(f"{self.path_save}/save_agent_{date_now}/{self.target_critic.net_name}.h5")
        
        np.save(f"{self.path_save}/save_agent_{date_now}/noise.npy", self.noise)
        
        self.replay_buffer.save(f"{self.path_save}/save_agent_{date_now}")

    def load(self):
        self.actor.load_weights(f"{self.path_load}/{self.actor.net_name}.h5")
        self.target_actor.load_weights(f"{self.path_load}/{self.target_actor.net_name}.h5")
        self.critic.load_weights(f"{self.path_load}/{self.critic.net_name}.h5")
        self.target_critic.load_weights(f"{self.path_load}/{self.target_critic.net_name}.h5")
        
        self.noise = np.load(f"{self.path_load}/noise.npy")
        
        self.replay_buffer.load(f"{self.path_load}")
        
        
        
    def _ornstein_uhlenbeck_process(self, x, theta=THETA, mu=0, dt=DT, std=0.2):
        """
        Ornstein–Uhlenbeck process
        """
        return x + theta * (mu-x) * dt + std * np.sqrt(dt) * np.random.normal(size=self.actions_dim)

    def get_action(self, observation, noise, evaluation=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluation:
            self.noise = self._ornstein_uhlenbeck_process(noise)
            actions += self.noise

        actions = tf.clip_by_value(actions, self.lower_bound, self.upper_bound)

        return actions[0]

    def learn(self):
        if self.replay_buffer.check_buffer_size() == False:
            return

        state, action, reward, new_state, done = self.replay_buffer.get_minibatch()

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(new_states)
            target_critic_values = tf.squeeze(self.target_critic(
                                new_states, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = reward + self.gamma * target_critic_values * (1-done)
            critic_loss = tf.keras.losses.MSE(target, critic_value)

        critic_gradient = tape.gradient(critic_loss,
                                            self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            policy_actions = self.actor(states)
            actor_loss = -self.critic(states, policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_gradient = tape.gradient(actor_loss, 
                                    self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_gradient, self.actor.trainable_variables))

        self.update_target_networks(self.tau)
