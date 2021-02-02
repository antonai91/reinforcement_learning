import sys
sys.path.append("../src/")
from config import *
import tensorflow.keras.losses as kloss
import tensorflow.keras.optimizers as opt
import tensorflow as tf
import numpy as np
import time
import wandb
import os

class Agent:
    def __init__(self, model, save_path=PATH_SAVE_MODEL, load_path=PATH_LOAD_MODEL, lr=LR, gamma=GAMMA, value_c=VALUE_C,
                 entropy_c=ENTROPY_C, clip_ratio=CLIP_RATIO, std_adv=STD_ADV, agent=AGENT, input_shape=INPUT_SHAPE,
                 batch_size = BATCH_SIZE, updates=N_UPDATES):
        # `gamma` is the discount factor
        self.gamma = gamma
        # Coefficients are used for the loss terms.
        self.value_c = value_c
        self.entropy_c = entropy_c
        # `gamma` is the discount factor
        self.gamma = gamma
        self.save_path = save_path
        self.load_path = load_path
        self.clip_ratio = clip_ratio
        self.std_adv = std_adv
        self.agent = agent
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.updates = updates
        self.opt = opt.RMSprop(lr=lr)
        
        self.model = model

        if load_path is not None:
            print("loading model in {}".format(load_path))
            self.load_model(load_path)
            print("model loaded")
            
            
    def train(self, wrapper):
        # Storage helpers for a single batch of data.
        actions = np.empty((self.batch_size), dtype=np.int32)
        rewards, dones, values = np.empty((3, self.batch_size))
        observations = np.empty((self.batch_size,) + self.input_shape)
        old_logits = np.empty((self.batch_size, wrapper.env.action_space.n), dtype=np.float32)

        # Training loop: collect samples, send to optimizer, repeat updates times.
        ep_rewards = [0.0]
        next_obs = wrapper.reset()
        for update in range(self.updates):
            start_time = time.time()
            for step in range(self.batch_size):
                observations[step] = next_obs.copy()
                old_logits[step], actions[step], values[step] = self.logits_action_value(next_obs[None, :])
                next_obs, rewards[step], dones[step] = wrapper.step(actions[step])
                next_obs = wrapper.state
                ep_rewards[-1] += rewards[step]
                if dones[step]:
                    ep_rewards.append(0.0)
                    next_obs = wrapper.reset()
                    wandb.log({'Game number': len(ep_rewards) - 1, '# Update': update, '% Update': round(update / self.updates, 2),
                                "Reward": round(ep_rewards[-2], 2), "Time taken": round(time.time() - start_time, 2)})

            _, _, next_value = self.logits_action_value(next_obs[None, :])

            returns, advs = self._returns_advantages(rewards, dones, values, next_value, self.std_adv)

            # Performs a full training step on the collected batch.
            # Note: no need to mess around with gradients, Keras API handles it.
            with tf.GradientTape() as tape:
                logits, v = self.model(observations, training=True)
                if self.agent == "A2C":
                    logit_loss = self._logits_loss_a2c(actions, advs, logits)
                elif self.agent == "PPO":
                    logit_loss = self._logits_loss_ppo(old_logits, logits, actions, advs, wrapper.env.action_space.n)
                else:
                    raise Exception("Sorry agent can be just A2C or PPO")
                value_loss = self._value_loss(returns, v)
                loss = logit_loss + value_loss
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
                

        return ep_rewards

    def _returns_advantages(self, rewards, dones, values, next_value, standardize_adv):
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)

        # Returns are calculated as discounted sum of future rewards.
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]

        # Advantages are equal to returns - baseline (value estimates in our case).
        advantages = returns - values
        
        if standardize_adv:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)

        return returns, advantages

    def _new_returns_advantages(self, rewards, dones, values, next_value):
        """
        Broken for now
        """
        next_values = np.append(values, next_value, axis=-1)
        g = 0
        lmbda = 0.95
        returns = np.zeros_like(rewards)
        adv = np.zeros_like(rewards)
        for t in reversed(range(rewards.shape[0])):
            delta = rewards[t] + self.gamma * next_values[t + 1] * (1 - dones[t]) - values[t]
            g = delta + self.gamma * lmbda * g
            adv[t] = g
            returns[t] = adv[t] + values[t]
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        return returns, adv

    def test(self, wrapper, render=True):
        obs, done, ep_reward = wrapper.reset(), False, 0
        while not done:
            _, action, _ = self.logits_action_value(obs[None, :])
            obs, reward, done = wrapper.step(action)
            obs = wrapper.state
            ep_reward += reward
        return ep_reward

    def _value_loss(self, returns, value):
        # Value loss is typically MSE between value estimates and returns.
        return self.value_c * kloss.mean_squared_error(returns, value)

    def _logits_loss_a2c(self, actions, advantages, logits):

        # Sparse categorical CE loss obj that supports sample_weight arg on `call()`.
        # `from_logits` argument ensures transformation into normalized probabilities.
        weighted_sparse_ce = kloss.SparseCategoricalCrossentropy(from_logits=True)

        # Policy loss is defined by policy gradients, weighted by advantages.
        # Note: we only calculate the loss on the actions we've actually taken.
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)

        # Entropy loss can be calculated as cross-entropy over itself.
        probs = tf.nn.softmax(logits)
        entropy_loss = kloss.categorical_crossentropy(probs, probs)

        # We want to minimize policy and maximize entropy losses.
        # Here signs are flipped because the optimizer minimizes.
        return policy_loss - self.entropy_c * entropy_loss
    
    def _logits_loss_ppo(self, old_logits, logits, actions, advs, n_actions):
        actions_oh = tf.one_hot(actions, n_actions)
        actions_oh = tf.reshape(actions_oh, [-1, n_actions])
        actions_oh = tf.cast(actions_oh, tf.float32)
        actions_oh = tf.stop_gradient(actions_oh)
        
        new_policy = tf.nn.log_softmax(logits)
        old_policy = tf.nn.log_softmax(old_logits)
        old_policy = tf.stop_gradient(old_policy)
        
        old_log_p = tf.reduce_sum(old_policy * actions_oh, axis=1)
        log_p = tf.reduce_sum(new_policy * actions_oh, axis=1)
        ratio = tf.exp(log_p - old_log_p)
        clipped_ratio = tf.clip_by_value(
            ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        advs = tf.stop_gradient(advs)
        advs = tf.cast(advs, tf.float32)
        surrogate = tf.minimum(ratio * advs, clipped_ratio * advs)
        return -tf.reduce_mean(surrogate) - self.entropy_c * kloss.categorical_crossentropy(new_policy, new_policy)
    
    def save_model(self, folder_path):
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        self.model.save_weights(folder_path )
                              
    
    def load_model(self, folder_path):
        self.model.load_weights(folder_path)
        
    def logits_action_value(self, obs):
        # Executes `call()` under the hood.
        logits, value = self.model(obs)
        action = self.model.dist(logits)
        return logits, np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)
