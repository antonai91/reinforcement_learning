import sys
sys.path.append("../src/")
import wandb
import os
from config import *
from pong_wrapper import *
from process_image import *
from utilities import *
from network import *
from agent import *

wandb.init(
  project="tensorflow2_pong_a2c",
  tags=[AGENT.lower(), "CNN", "RL", "atari_pong"],
  config=CONFIG_WANDB,
)

pw = PongWrapper(ENV_NAME, history_length=4)
model = Model(num_actions=pw.env.action_space.n, hidden=HIDDEN)
agent = Agent(model)

def main():
    rewards_history = agent.train(pw)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Save the model, I need this in order to save the networks, frame number, rewards and losses. 
        # if I want to stop the script and restart without training from the beginning
        if PATH_SAVE_MODEL is None:
            print("Setting path to ../model/{}/".format(AGENT.lower())
            PATH_SAVE_MODEL = "../model/{}/".format(AGENT.lower())
        print('Saving the model in ' + f'{PATH_SAVE_MODEL}/save_agent_{time.strftime("%Y%m%d%H%M")}')
        agent.save_model(f'{PATH_SAVE_MODEL}/save_agent_{time.strftime("%Y%m%d%H%M")}')
        print('Saved.')
