#!/usr/bin/python

import sys
sys.path.append("../src")
import gym
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import wandb
from config import *
from replay_buffer import *
from networks import *
from agent import *

config = dict(
  learning_rate_actor = ACTOR_LR,
  learning_rate_critic = ACTOR_LR,
  batch_size = BATCH_SIZE,
  architecture = "DDPG",
  infra = "Ubuntu",
  env = ENV_NAME
)

wandb.init(
  project=f"tensorflow2_{ENV_NAME.lower()}",
  tags=["DDPG", "FCL", "RL"],
  config=config,
)

env = gym.make(ENV_NAME)
agent = Agent(env)

scores = []
evaluation = False

if PATH_LOAD is not None:
    print("loading weights")
    observation = env.reset()
    action = agent.actor(observation[None, :])
    agent.target_actor(observation[None, :])
    agent.critic(observation[None, :], action)
    agent.target_critic(observation[None, :], action)
    agent.load()
    print(agent.replay_buffer.buffer_counter)
    print(agent.replay_buffer.n_episodes)
    print(agent.noise)

for _ in tqdm(range(MAX_GAMES)):
    start_time = time.time()
    states = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.get_action(states, evaluation)
        new_states, reward, done, info = env.step(action)
        score += reward
        agent.add_to_replay_buffer(states, action, reward, new_states, done)
        agent.learn()
        states = new_states
        
    agent.replay_buffer.update_n_games()
    
    scores.append(score)
    
    wandb.log({'Game number': agent.replay_buffer.n_games, '# Episodes': agent.replay_buffer.buffer_counter, 
               "Average reward": round(np.mean(scores[-10:]), 2), \
                      "Time taken": round(time.time() - start_time, 2)})

    if _ + 1 % EVALUATION_FREQUENCY == 0:
        states = env.reset()
        evaluation = True
        score = 0
        done = False
        while not done:
            action = agent.get_action(states, evaluation)
            new_states, reward, done, info = env.step(action)
            score += reward
            states = new_states
        wandb.log({'Game number': agent.replay_buffer.n_games, 
                   '# Episodes': agent.replay_buffer.buffer_counter, 
                   'Evaluation score': score})
        evaluation = False
     
    if _ + 1 % SAVE_FREQUENCY == 0:
        print("saving...")
        agent.save()
        print("saved")
