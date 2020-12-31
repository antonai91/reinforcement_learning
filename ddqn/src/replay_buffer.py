import os
import random
import numpy as np

class ReplayBuffer(object):
    """
    Replay Memory that stores the last size transitions
    """
    def __init__(self, size: int=1000000, input_shape: tuple=(84, 84), history_length: int=4, reward_type: str = "integer"):
        """
        Arguments:
            size: Number of stored transitions
            input_shape: Shape of the preprocessed frame
            history_length: Number of frames stacked together that the agent can see
        """
        self.size = size
        self.input_shape = input_shape
        self.history_length = history_length
        self.count = 0  # total index of memory written to, always less than self.size
        self.current = 0  # index to write to

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.input_shape[0], self.input_shape[1]), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)
        self.priorities = np.zeros(self.size, dtype=np.float32)

        self.reward_type = reward_type

    def add_experience(self, action, frame, reward, terminal, clip_reward=True):
        """Saves a transition to the replay buffer

        Arguments:
            action: An integer between 0 and env.action_space.n - 1 
                determining the action the agent perfomed
            frame: A (84, 84, 1) frame of the game in grayscale
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != self.input_shape:
            raise ValueError('Dimension of frame is wrong!')

        if clip_reward:
            if reward_type == "integer":
                reward = np.sign(reward)
            else:
                reward = np.clip(reward, -1.0, 1.0)
        # Write memory
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.priorities[self.current] = max(self.priorities.max(), 1)  # make the most recent experience important
        self.count = max(self.count, self.current+1)
        self.current = (self.current + 1) % self.size # when a < b then a % b = a

    def get_minibatch(self, batch_size: int = 32):
        """
        Returns a minibatch of size batch_size

        Arguments:
            batch_size: How many samples to return

        Returns:
            A tuple of states, actions, rewards, new_states, and terminals
        """

        if self.count < self.history_length:
            raise ValueError('Not enough memories to get a minibatch')

        indices = []
        for i in range(batch_size):
            while True:
                # Get a random number from history_length to maximum frame written with probabilities based on priority weights
                index = random.randint(self.history_length, self.count - 1)

                # We check that all frames are from same episode with the two following if statements.  If either are True, the index is invalid.
                if index >= self.current and index - self.history_length <= self.current:
                    continue
                if self.terminal_flags[index - self.history_length:index].any():
                    continue
                break
            indices.append(index)

        # Retrieve states from memory
        states = []
        new_states = []
        for idx in indices:
            states.append(self.frames[idx-self.history_length:idx, ...])
            new_states.append(self.frames[idx-self.history_length+1:idx+1, ...])

        states = np.transpose(np.asarray(states), axes=(0, 2, 3, 1))
        new_states = np.transpose(np.asarray(new_states), axes=(0, 2, 3, 1))

        return states, self.actions[indices], self.rewards[indices], new_states, self.terminal_flags[indices]

    def save(self, folder_name):
        """
        Save the replay buffer
        """

        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        np.save(folder_name + '/actions.npy', self.actions)
        np.save(folder_name + '/frames.npy', self.frames)
        np.save(folder_name + '/rewards.npy', self.rewards)
        np.save(folder_name + '/terminal_flags.npy', self.terminal_flags)

    def load(self, folder_name):
        """
        Load the replay buffer
        """
        self.actions = np.load(folder_name + '/actions.npy')
        self.frames = np.load(folder_name + '/frames.npy')
        self.rewards = np.load(folder_name + '/rewards.npy')
        self.terminal_flags = np.load(folder_name + '/terminal_flags.npy')
