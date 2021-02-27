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
    def __init__(self, env, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, gamma=GAMMA, tau=TAU, reward_scale=REWARD_SCALE):
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(env)
        self.actions_dim = env.action_space.shape[0]
        self.upper_bound = env.action_space.high[0]
        self.lower_bound = env.action_space.low[0]
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.actor = Actor(actions_dim=self.actions_dim, name='actor', upper_bound=env.action_space.high)
        self.critic_0 = Critic(name='critic_0')
        self.critic_1 = Critic(name='critic_1')
        self.critic_value = CriticValue(name='value')
        self.critic_target_value = CriticValue(name='target_value')

        self.actor.compile(optimizer=opt.Adam(learning_rate=self.actor_lr))
        self.critic_0.compile(optimizer=opt.Adam(learning_rate=self.critic_lr))
        self.critic_1.compile(optimizer=opt.Adam(learning_rate=self.critic_lr))
        self.critic_value.compile(optimizer=opt.Adam(learning_rate=self.critic_lr))
        self.critic_target_value.compile(optimizer=opt.Adam(learning_rate=self.critic_lr))

        self.reward_scale = reward_scale

        self.critic_target_value.set_weights(self.critic_value.weights)
        
    def update_target_networks(self, tau):
        critic_value_weights = self.critic_value.weights
        critic_target_value_weights = self.critic_target_value.weights
        for index in range(len(critic_value_weights)):
            critic_target_value_weights[index] = tau * critic_value_weights[index] + (1 - tau) * critic_target_value_weights[index]

        self.critic_target_value.set_weights(critic_target_value_weights)
        
    def add_to_replay_buffer(self, state, action, reward, new_state, done):
        self.replay_buffer.add_record(state, action, reward, new_state, done)
        
    def save(self):
        date_now = time.strftime("%Y%m%d%H%M")
        if not os.path.isdir(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}"):
            os.makedirs(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}")
        self.actor.save_weights(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}/{self.actor.net_name}.h5")
        self.critic_0.save_weights(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}/{self.critic_0.net_name}.h5")
        self.critic_1.save_weights(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}/{self.critic_1.net_name}.h5")
        self.critic_value.save_weights(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}/{self.critic_value.net_name}.h5")
        self.critic_target_value.save_weights(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}/{self.critic_target_value.net_name}.h5")
        
        self.replay_buffer.save(f"{self.path_save}/save_agent_{ENV_NAME.lower()}_{date_now}")

    def load(self):
        self.actor.load_weights(f"{self.path_load}/{self.actor.net_name}.h5")
        self.critic_0.load_weights(f"{self.path_load}/{self.critic_0.net_name}.h5")
        self.critic_1.load_weights(f"{self.path_load}/{self.critic_1.net_name}.h5")
        self.critic_value.load_weights(f"{self.path_load}/{self.critic_value.net_name}.h5")
        self.critic_target_value.load_weights(f"{self.path_load}/{self.critic_target_value.net_name}.h5")
        
        #self.replay_buffer.load(f"{self.path_load}")

    def get_action(self, observation):
        state = tf.convert_to_tensor([observation])
        actions, _ = self.actor.get_action_log_probs(state, reparameterization_trick=False)

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
            value = tf.squeeze(self.critic_value(states), 1)
            target_value = tf.squeeze(self.critic_target_value(new_states), 1)

            policy_actions, log_probs = self.actor.get_action_log_probs(states, reparameterization_trick=False)
            log_probs = tf.squeeze(log_probs,1)
            q_value_0 = self.critic_0(states, policy_actions)
            q_value_1 = self.critic_1(states, policy_actions)
            q_value = tf.squeeze(tf.math.minimum(q_value_0, q_value_1), 1)

            value_target = q_value - log_probs
            value_critic_loss = 0.5 * tf.keras.losses.MSE(value, value_target)

        value_critic_gradient = tape.gradient(value_critic_loss, self.critic_value.trainable_variables)
        self.critic_value.optimizer.apply_gradients(zip(value_critic_gradient, self.critic_value.trainable_variables))


        with tf.GradientTape() as tape:
            new_policy_actions, log_probs = self.actor.get_action_log_probs(states, reparameterization_trick=True)
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
            old_q_value_0 = tf.squeeze(self.critic_0(state, action), 1)
            old_q_value_1 = tf.squeeze(self.critic_1(state, action), 1)
            critic_0_loss = 0.5 * tf.keras.losses.MSE(old_q_value_0, q_pred)
            critic_1_loss = 0.5 * tf.keras.losses.MSE(old_q_value_1, q_pred)
    
        critic_0_network_gradient = tape.gradient(critic_0_loss, self.critic_0.trainable_variables)
        critic_1_network_gradient = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)

        self.critic_0.optimizer.apply_gradients(zip(critic_0_network_gradient, self.critic_0.trainable_variables))
        self.critic_1.optimizer.apply_gradients(zip(critic_1_network_gradient, self.critic_1.trainable_variables))

        self.update_target_networks(tau=self.tau)
        
        self.replay_buffer.update_n_games()
