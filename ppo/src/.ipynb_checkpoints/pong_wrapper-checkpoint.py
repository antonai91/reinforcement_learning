import random
import gym
import numpy as np
from process_image import process_image

class PongWrapper(object):
    """
    Wrapper for the environment provided by Openai Gym
    """

    def __init__(self, env_name: str, no_op_steps: int = 10, history_length: int = 4):
        self.env = gym.make(env_name)
        self.no_op_steps = no_op_steps
        self.history_length = 4 # number of frames to put together (we need dynamic to see where the ball is going)

        self.state = None
        
    def reset(self, evaluation: bool = False):
        """Resets the environment

        Arguments:
            evaluation: Set to True when we are in evaluation mode, in this case the agent takes a random number of no-op steps if True.
        """

        self.frame = self.env.reset()
        
        # If in evaluation model, take a random number of no-op steps
        if evaluation:
            for _ in range(random.randint(0, self.no_op_steps)):
                self.env.step(1)

        # For the initial state, we stack the first frame four times
        self.state = np.repeat(process_image(self.frame), self.history_length, axis=2)

    def step(self, action: int, render_mode=None):
        """
        Arguments:
            action: An integer describe action to take
            render_mode: None doesn't render anything, 'human' renders the screen in a new window, 'rgb_array' returns also an np.array with rgb values

        Returns:
            processed_image: The processed new frame as a result of that action
            reward: The reward for taking that action
            terminal: Whether the game has ended
        """
        new_frame, reward, terminal, info = self.env.step(action)

        processed_image = process_image(new_frame)

        self.state = np.append(self.state[:, :, 1:], processed_image, axis=2) # replace the first observation of the previous state with the last one

        if render_mode == 'rgb_array':
            return processed_image, reward, terminal, self.env.render(render_mode)
        elif render_mode == 'human':
            self.env.render(render_mode)

        return processed_image, reward, terminal
