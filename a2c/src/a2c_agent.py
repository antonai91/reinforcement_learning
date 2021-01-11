import tensorflow.keras.losses as kloss
import tensorflow.keras.optimizers as opt
import tensorflow as tf
import numpy as np
import time
import wandb
import os

class A2CAgent:
    def __init__(self, model, save_path=None, load_path=None, lr=7e-3, gamma=0.99, value_c=0.5, entropy_c=1e-4):
        # `gamma` is the discount factor
        self.gamma = gamma
        # Coefficients are used for the loss terms.
        self.value_c = value_c
        self.entropy_c = entropy_c
        # `gamma` is the discount factor
        self.gamma = gamma
        self.save_path = save_path
        self.load_path = load_path
        
        self.model = model
        self.model.compile(
                optimizer=opt.RMSprop(lr=lr),
                # Define separate losses for policy logits and value estimate.
                loss=[self._logits_loss, self._value_loss])
        if load_path is not None:
            print("loading model in {}".format(load_path))
            self.load_model(load_path)
            print("model loaded")
            
            
    def train(self, wrapper, batch_sz=64, updates=1000000, input_shape=(84, 84, 4)):
        # Storage helpers for a single batch of data.
        actions = np.empty((batch_sz,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_sz))
        observations = np.empty((batch_sz,) + input_shape)

        # Training loop: collect samples, send to optimizer, repeat updates times.
        ep_rewards = [0.0]
        next_obs = wrapper.reset()
        for update in range(updates):
            start_time = time.time()
            for step in range(batch_sz):
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs[None, :])
                next_processed_image, rewards[step], dones[step], _ = wrapper.step(actions[step], "rgb_array")
                next_obs = wrapper.state
                ep_rewards[-1] += rewards[step]
                if dones[step]:
                    ep_rewards.append(0.0)
                    next_obs = wrapper.reset()
                    wandb.log({'Game number': len(ep_rewards) - 1, '# Update': update, '% Update': round(update / updates, 2), 
                                "Reward": round(ep_rewards[-2], 2), "Time taken": round(time.time() - start_time, 2)})

            if ep_rewards[-1] > -5:
                self.learning_rate = 0.00025
                        

            _, next_value = self.model.action_value(next_obs[None, :])

            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            # A trick to input actions and advantages through same API.
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)

            # Performs a full training step on the collected batch.
            # Note: no need to mess around with gradients, Keras API handles it.
            losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
            
            if update % 500 == 0 and self.save_path is not None:
                self.save_model(f'{self.save_path}/save_agent_{time.strftime("%Y%m%d%H%M") + "_" + str(update).zfill(8)}')

        return ep_rewards

    def _returns_advantages(self, rewards, dones, values, next_value):
        # `next_value` is the bootstrap value estimate of the future state (critic).
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)

        # Returns are calculated as discounted sum of future rewards.
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]

        # Advantages are equal to returns - baseline (value estimates in our case).
        advantages = returns - values

        return returns, advantages

    def test(self, wrapper, render=True):
        obs, done, ep_reward = wrapper.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            processed_image, reward, done, _ = wrapper.step(action, "rgb_array")
            obs = wrapper.state
            ep_reward += reward
        return ep_reward

    def _value_loss(self, returns, value):
        # Value loss is typically MSE between value estimates and returns.
        return self.value_c * kloss.mean_squared_error(returns, value)

    def _logits_loss(self, actions_and_advantages, logits):
        # A trick to input actions and advantages through the same API.
        actions, advantages = tf.split(actions_and_advantages, 2, axis=-1)

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
    
    def save_model(self, folder_path):
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        self.model.save_weights(folder_path + '/a2c')
                              
    
    def load_model(self, folder_path):
        self.model.load_weights(folder_path + '/a2c')
