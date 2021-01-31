import sys
sys.path.append("../src/")
import wandb
import os
from config import *
from pong_wrapper import *
from process_image import *
from utilities import *
from ppo_network import *
from ppo_agent import *

wandb.init(
  project="tensorflow2_pong_ppo",
  tags=["a2c", "CNN", "RL"],
  config=CONFIG_WANDB,
)

pw = PongWrapper(ENV_NAME, history_length=4)
model = PpoNetwork()
ppo_agent = PpoAgent(model)

def main():
    rewards_history = ppo_agent.train(pw)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Save the model, I need this in order to save the networks, frame number, rewards and losses. 
        # if I want to stop the script and restart without training from the beginning
        if PATH_SAVE_MODEL is None:
            print("Setting path to ../model")
            PATH_SAVE_MODEL = "../model"
        print('Saving the model in ' + f'{PATH_SAVE_MODEL}/save_agent_{time.strftime("%Y%m%d%H%M")}')
        ppo_agent.save_model(f'{PATH_SAVE_MODEL}/save_agent_{time.strftime("%Y%m%d%H%M")}')
        print('Saved.')
