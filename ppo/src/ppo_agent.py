import numpy as np
import time
import wandb
import os
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error, categorical_crossentropy
import sys
sys.path.append("../src/")
from config import *

class PpoAgent():
    def __init__(self, model, save_path=PATH_SAVE_MODEL, load_path=PATH_LOAD_MODEL, gamma=GAMMA, max_updates=MAX_UPDATES, batch_size=BATCH_SIZE, input_shape=INPUT_SHAPE+(HISTORY_LENGHT, ), 
                 clip_ratio=CLIP_RATIO, entropy_c=ENTROPY_C, value_c=VALUE_C, lr=LR):
        # `gamma` is the discount factor
        self.gamma = gamma
        self.max_updates = max_updates
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.save_path = save_path
        self.load_path = load_path
        self.clip_ratio = clip_ratio
        self.entropy_c = entropy_c
        self.value_c = value_c
        self.lr = lr
        
        self.model = model
        self.opt = Adam(self.lr)
        
        if load_path is not None:
            print("loading model in {}".format(load_path))
            self.load_model(load_path)
            print("model loaded")
    
    def logit_loss(self, old_policy, new_policy, actions_oh, advs):
        advs = tf.stop_gradient(advs)
        advs =  tf.cast(advs, tf.float32)
        old_log_p = tf.math.log(
            tf.reduce_sum(old_policy * actions_oh))
        old_log_p = tf.stop_gradient(old_log_p)
        log_p = tf.math.log(tf.reduce_sum(
            new_policy * actions_oh))
        ratio = tf.math.exp(log_p - old_log_p)
        clipped_ratio = tf.clip_by_value(
            ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        surrogate = -tf.minimum(ratio * advs, clipped_ratio * advs)
        return tf.reduce_mean(surrogate) - self.entropy_c * categorical_crossentropy(new_policy, new_policy)
    
    def value_loss(self, returns, values):
        return self.value_c * mean_squared_error(returns, values)
    
    def get_prob_value_action(self, state):
        prob, value = self.model(state[None, :])
        return prob, value, tf.squeeze(tf.random.categorical(prob, 1), axis=-1)
    
    def get_returns_advantages(self, rewards, dones, values, next_value):
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # Returns are calculated as discounted sum of future rewards.
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        
        # Advantages are equal to returns - baseline (value estimates in our case).
        advantages = returns - values
        
        return returns, advantages
        
    def train(self, game_wrapper):
        
        probs = np.empty((self.batch_size, game_wrapper.env.action_space.n))
        actions = np.empty((self.batch_size,), dtype=np.int32)
        rewards, dones, values = np.empty((3, self.batch_size))
        observations = np.empty((self.batch_size,) + self.input_shape)
        
        ep_rewards = [0.0]
        next_obs = game_wrapper.reset()
        
        for update in tqdm(range(self.max_updates)):
            start_time = time.time()
            
            for step in range(self.batch_size):
                observations[step] = next_obs.copy()
                prob, value, action = self.get_prob_value_action(next_obs)
                actions[step] = action
                probs[step] = prob
                values[step] = value
                next_processed_image, rewards[step], dones[step], _ = game_wrapper.step(actions[step], "rgb_array")
                next_obs = game_wrapper.state
                ep_rewards[-1] += rewards[step]
                if dones[step]:
                    ep_rewards.append(0.0)
                    next_obs = game_wrapper.reset()
                    wandb.log({'Game number': len(ep_rewards) - 1, '# Update': update, '% Update': round(update / self.max_updates, 2), 
                                "Reward": round(ep_rewards[-2], 2), "Time taken": round(time.time() - start_time, 2)})
            
            _, next_value = self.model(next_obs[None, :])
            next_value = np.squeeze(next_value, -1)
            
            returns, advs = self.get_returns_advantages(rewards, dones, values, next_value)
            
            actions_oh = tf.one_hot(actions, game_wrapper.env.action_space.n)
            actions_oh = tf.reshape(actions_oh, [-1, game_wrapper.env.action_space.n])
            actions_oh = tf.cast(actions_oh, tf.float32)
            
            with tf.GradientTape() as tape:
                logits, v = self.model(observations, training=True)
                logit_loss = self.logit_loss(probs, logits, actions_oh, advs)
                value_loss = self.value_loss(returns, v)
                value_loss = tf.cast(value_loss, tf.float32)
                loss = logit_loss + value_loss
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
            
            wandb.log({"logit_loss": tf.math.reduce_mean(logit_loss).numpy(), "value_loss": tf.math.reduce_mean(value_loss).numpy(), 
                       "loss": tf.math.reduce_mean(loss).numpy()})
            
            if (update + 1) % 10000 == 0 and self.save_path is not None:
                print("Saving model in {}".format(self.save_path))
                self.save_model(f'{self.save_path}/save_agent_{time.strftime("%Y%m%d%H%M") + "_" + str(update).zfill(8)}')
                print("model saved")
        
        return ep_rewards

    def save_model(self, folder_path):
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        self.model.save_weights(folder_path + '/ppo')
                              
    
    def load_model(self, folder_path):
        self.model.load_weights(folder_path + '/ppo')
