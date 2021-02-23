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
    def __init__(self, env, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, gamma=GAMMA, max_size=BUFFER_CAPACITY, tau=TAU, reward_scale=REWARD_SCALE, path_save=PATH_SAVE, path_load=PATH_LOAD):
        
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
        self.critic_0 = Critic(name='critic_0')
        self.critic_1 = Critic(name='critic_1')
        self.value_critic = CriticValue(name="value_critic")
        self.target_value_critic = CriticValue(name="target_value_critic")

        self.actor.compile(optimizer=opt.Adam(learning_rate=actor_lr))
        self.critic_0.compile(optimizer=opt.Adam(learning_rate=critic_lr))
        self.critic_1.compile(optimizer=opt.Adam(learning_rate=critic_lr))
        self.value_critic.compile(optimizer=opt.Adam(learning_rate=critic_lr))
        self.target_value_critic.compile(optimizer=opt.Adam(learning_rate=critic_lr))

        value_critic_weights = self.value_critic.get_weights()
        
        self.target_value_critic.set_weights(value_critic_weights)
        
        self.reward_scale = reward_scale
        

    def update_target_networks(self, tau):
        value_critic_weights = self.value_critic.weights
        target_value_critic_weights = self.target_value_critic.weights
        for index in range(len(value_critic_weights)):
            target_value_critic_weights[index] = tau * value_critic_weights[index] + (1 - tau) * target_value_critic_weights[index]

        self.target_value_critic.set_weights(target_value_critic_weights)
    
    def add_to_replay_buffer(self, state, action, reward, new_state, done):
        self.replay_buffer.add_record(state, action, reward, new_state, done)
    """
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
    """
    def get_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)

        actions, _ = self.actor.evaluate(state)

        return actions[0]

    def learn(self):
        if self.replay_buffer.check_buffer_size() == False:
            return None

        state, action, reward, new_state, done = self.replay_buffer.get_minibatch()

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            value = tf.squeeze(self.value_critic(states), 1)
            target_value = tf.squeeze(self.target_value_critic(new_states), 1)
            
            current_policy_actions, log_probs = self.actor.evaluate(states, False)
            log_probs = tf.squeeze(log_probs,1)

            q_value_0 = self.critic_0(states, current_policy_actions)
            q_value_1 = self.critic_1(states, current_policy_actions)
            q_value = tf.squeeze(tf.math.minimum(q_value_0, q_value_1), 1)

            y = q_value - log_probs
            value_critic_loss = 0.5 * tf.keras.losses.MSE(value, y)
        
        value_critic_gradient = tape.gradient(value_critic_loss, self.value_critic.trainable_variables)
        self.value_critic.optimizer.apply_gradients(zip(value_critic_gradient, self.value_critic.trainable_variables))
        
        
        with tf.GradientTape() as tape:
            new_policy_actions, log_probs = self.actor.evaluate(states, False)
            log_probs = tf.squeeze(log_probs, 1)
            new_q_value_0 = self.critic_0(states, new_policy_actions)
            new_q_value_1 = self.critic_1(states, new_policy_actions)
            new_q_value = tf.squeeze(tf.math.minimum(new_q_value_0, new_q_value_1), 1)       

            actor_loss = log_probs - new_q_value
            actor_loss = tf.math.reduce_mean(actor_loss)
            
        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))
        
        with tf.GradientTape(persistent=True) as tape:
            q_pred = self.reward_scale * reward + self.gamma * target_value * (1-done)
            old_q0_value = tf.squeeze(self.critic_0(state, action), 1)
            old_q1_value = tf.squeeze(self.critic_1(state, action), 1)
            
            loss_critic_0 = 0.5 * tf.keras.losses.MSE(old_q0_value, q_pred)
            loss_critic_1 = 0.5 * tf.keras.losses.MSE(old_q1_value, q_pred)
    
        critic_0_gradient = tape.gradient(loss_critic_0, self.critic_0.trainable_variables)
        critic_1_gradient = tape.gradient(loss_critic_1, self.critic_1.trainable_variables)

        self.critic_0.optimizer.apply_gradients(zip(critic_0_gradient, self.critic_0.trainable_variables))
        self.critic_1.optimizer.apply_gradients(zip(critic_1_gradient, self.critic_1.trainable_variables))

        self.update_target_networks(self.tau)
