import numpy as np
import tensorflow as tf
import time
import json
import os
import sys
from tqdm import tqdm
import wandb
sys.path.append("../src")
from config import *
from make_env import *
from replay_buffer import *
from networks import *
from agent import *
from super_agent import *

config = dict(
  learning_rate_actor = ACTOR_LR,
  learning_rate_critic = CRITIC_LR,
  batch_size = BATCH_SIZE,
  architecture = "MADDPG",
  infra = "Colab",
  env = ENV_NAME
)

wandb.init(
  project=f"tensorflow2_madddpg_{ENV_NAME.lower()}",
  tags=["MADDPG", "RL"],
  config=config,
)

env = make_env(ENV_NAME)
super_agent = SuperAgent(env)

scores = []

if PATH_LOAD_FOLDER is not None:
    print("loading weights")
    actors_state = env.reset()
    actors_action = super_agent.get_actions([actors_state[index][None, :] for index in range(super_agent.n_agents)])
    [super_agent.agents[index].target_actor(actors_state[index][None, :]) for index in range(super_agent.n_agents)]
    state = np.concatenate(actors_state)
    actors_action = np.concatenate(actors_action)
    [super_agent.agents[index].critic(state[None, :], actors_action[None, :]) for index in range(super_agent.n_agents)]
    [super_agent.agents[index].target_critic(state[None, :], actors_action[None, :]) for index in range(super_agent.n_agents)]
    super_agent.load()

    print(super_agent.replay_buffer.buffer_counter)
    print(super_agent.replay_buffer.n_games)



for n_game in tqdm(range(MAX_GAMES)):
    start_time = time.time()
    actors_state = env.reset()
    done = [False for index in range(super_agent.n_agents)]
    score = 0
    step = 0
    while not any(done):
        actors_action = super_agent.get_actions([actors_state[index][None, :] for index in range(super_agent.n_agents)])
        actors_next_state, reward, done, info = env.step(actors_action)
        
        state = np.concatenate(actors_state)
        next_state = np.concatenate(actors_next_state)
        
        super_agent.replay_buffer.add_record(actors_state, actors_next_state, actors_action, state, next_state, reward, done)
        
        actors_state = actors_next_state
        
        score += sum(reward)
        step += 1
        if step >= MAX_STEPS:
            break
    
    if super_agent.replay_buffer.check_buffer_size():
        super_agent.train()
        
    super_agent.replay_buffer.update_n_games()
    
    scores.append(score)
    
    wandb.log({'Game number': super_agent.replay_buffer.n_games, '# Episodes': super_agent.replay_buffer.buffer_counter, 
               "Average reward": round(np.mean(scores[-10:]), 2), \
                      "Time taken": round(time.time() - start_time, 2)})
    
    if (n_game + 1) % EVALUATION_FREQUENCY == 0 and super_agent.replay_buffer.check_buffer_size():
        actors_state = env.reset()
        done = [False for index in range(super_agent.n_agents)]
        score = 0
        step = 0
        while not any(done):
            actors_action = super_agent.get_actions([actors_state[index][None, :] for index in range(super_agent.n_agents)])
            actors_next_state, reward, done, info = env.step(actors_action)
            state = np.concatenate(actors_state)
            next_state = np.concatenate(actors_next_state)
            actors_state = actors_next_state
            score += sum(reward)
            step += 1
            if step >= MAX_STEPS:
                break
        wandb.log({'Game number': super_agent.replay_buffer.n_games, 
                   '# Episodes': super_agent.replay_buffer.buffer_counter, 
                   'Evaluation score': score})
            
    if (n_game + 1) % SAVE_FREQUENCY == 0 and super_agent.replay_buffer.check_buffer_size():
        print("saving weights and replay buffer...")
        super_agent.save()
        print("saved")
