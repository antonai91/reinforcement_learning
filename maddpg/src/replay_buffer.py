import numpy as np
import json
import os
import sys
sys.path.append("../src")
from config import *
from make_env import *

class ReplayBuffer():
    def __init__(self, env, buffer_capacity=BUFFER_CAPACITY, batch_size=BATCH_SIZE, min_size_buffer=MIN_SIZE_BUFFER):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.min_size_buffer = min_size_buffer
        self.buffer_counter = 0
        self.n_games = 0
        self.n_agents = env.n
        self.list_actors_dimension = [env.observation_space[index].shape[0] for index in range(self.n_agents)]
        self.critic_dimension = sum(self.list_actors_dimension)        
        self.list_actor_n_actions = [env.action_space[index].n for index in range(self.n_agents)]
        
        self.states = np.zeros((self.buffer_capacity, self.critic_dimension))
        self.rewards = np.zeros((self.buffer_capacity, self.n_agents))
        self.next_states = np.zeros((self.buffer_capacity, self.critic_dimension))
        self.dones = np.zeros((self.buffer_capacity, self.n_agents), dtype=bool)

        self.list_actors_states = []
        self.list_actors_next_states = []
        self.list_actors_actions = []
        
        for n in range(self.n_agents):
            self.list_actors_states.append(np.zeros((self.buffer_capacity, self.list_actors_dimension[n])))
            self.list_actors_next_states.append(np.zeros((self.buffer_capacity, self.list_actors_dimension[n])))
            self.list_actors_actions.append(np.zeros((self.buffer_capacity, self.list_actor_n_actions[n])))
            
    def __len__(self):
        return self.buffer_counter
        
    def check_buffer_size(self):
        return self.buffer_counter >= self.batch_size and self.buffer_counter >= self.min_size_buffer
    
    def update_n_games(self):
        self.n_games += 1
          
    def add_record(self, actor_states, actor_next_states, actions, state, next_state, reward, done):
        
        index = self.buffer_counter % self.buffer_capacity

        for agent_index in range(self.n_agents):
            self.list_actors_states[agent_index][index] = actor_states[agent_index]
            self.list_actors_next_states[agent_index][index] = actor_next_states[agent_index]
            self.list_actors_actions[agent_index][index] = actions[agent_index]

        self.states[index] = state
        self.next_states[index] = next_state
        self.rewards[index] = reward
        self.dones[index] = done
            
        self.buffer_counter += 1
            
    def get_minibatch(self):
        # If the counter is less than the capacity we don't want to take zeros records, 
        # if the cunter is higher we don't access the record using the counter 
        # because older records are deleted to make space for new one
        buffer_range = min(self.buffer_counter, self.buffer_capacity)

        batch_index = np.random.choice(buffer_range, self.batch_size, replace=False)

        # Take indices
        state = self.states[batch_index]
        reward = self.rewards[batch_index]
        next_state = self.next_states[batch_index]
        done = self.dones[batch_index]
            
        actors_state = [self.list_actors_states[index][batch_index] for index in range(self.n_agents)]
        actors_new_state = [self.list_actors_next_states[index][batch_index] for index in range(self.n_agents)]
        actors_action = [self.list_actors_actions[index][batch_index] for index in range(self.n_agents)]

        return state, reward, next_state, done, actors_state, actors_new_state, actors_action
    
    def save(self, folder_path):
        """
        Save the replay buffer
        """
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        
        np.save(folder_path + '/states.npy', self.states)
        np.save(folder_path + '/rewards.npy', self.rewards)
        np.save(folder_path + '/next_states.npy', self.next_states)
        np.save(folder_path + '/dones.npy', self.dones)
        
        for index in range(self.n_agents):
            np.save(folder_path + '/states_actor_{}.npy'.format(index), self.list_actors_states[index])
            np.save(folder_path + '/next_states_actor_{}.npy'.format(index), self.list_actors_next_states[index])
            np.save(folder_path + '/actions_actor_{}.npy'.format(index), self.list_actors_actions[index])
            
        dict_info = {"buffer_counter": self.buffer_counter, "n_games": self.n_games}
        
        with open(folder_path + '/dict_info.json', 'w') as f:
            json.dump(dict_info, f)
            
    def load(self, folder_path):
        self.states = np.load(folder_path + '/states.npy')
        self.rewards = np.load(folder_path + '/rewards.npy')
        self.next_states = np.load(folder_path + '/next_states.npy')
        self.dones = np.load(folder_path + '/dones.npy')
        
        self.list_actors_states = [np.load(folder_path + '/states_actor_{}.npy'.format(index)) for index in range(self.n_agents)]
        self.list_actors_next_states = [np.load(folder_path + '/next_states_actor_{}.npy'.format(index)) for index in range(self.n_agents)]
        self.list_actors_actions = [np.load(folder_path + '/actions_actor_{}.npy'.format(index)) for index in range(self.n_agents)]
        
        with open(folder_path + '/dict_info.json', 'r') as f:
            dict_info = json.load(f)
        self.buffer_counter = dict_info["buffer_counter"]
        self.n_games = dict_info["n_games"]
