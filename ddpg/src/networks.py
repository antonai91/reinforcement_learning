import sys
sys.path.append("../src")
from config import *
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.initializers import random_normal

class Actor(tf.keras.Model):
    def __init__(self, action_dim, action_bound, hidden_0=ACTOR_HIDDEN_0, hidden_1=ACTOR_HIDDEN_1):
        super().__init__()
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.hidden_0 = hidden_0
        self.hidden_1 = hidden_1

        
        self.dense_0 = Dense(hidden_0,  kernel_initializer=random_normal(), activation='relu', name="dense_0")
        self.dense_1 = Dense(hidden_1,  kernel_initializer=random_normal(), activation='relu', name="dense_1")
        self.action = Dense(self.action_dim, activation='tanh', kernel_initializer=random_normal(), name="action")
        
    def __call__(self, inputs, **kwargs):
        x = tf.convert_to_tensor(inputs)
        x = self.dense_0(x)
        x = self.dense_1(x)
        return self.action(x) * self.action_bound
    
class Critic(tf.keras.Model):
    def __init__(self, action_dim, hidden_0=CRITIC_HIDDEN_0, hidden_1=CRITIC_HIDDEN_1, hidden_2=CRITIC_HIDDEN_2, hidden_3=CRITIC_HIDDEN_3):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_0 = hidden_0
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.hidden_3 = hidden_3
        
        self.dense_0 = Dense(hidden_0,  kernel_initializer=random_normal(), activation='relu', name="dense_0")
        self.dense_1 = Dense(hidden_1,  kernel_initializer=random_normal(), activation='relu', name="dense_1")

        self.dense_2 = Dense(hidden_2,  kernel_initializer=random_normal(), activation='relu', name="dense_2")
        
        self.concat = Concatenate()
        
        self.dense_3 = Dense(hidden_3,  kernel_initializer=random_normal(), activation='relu', name="dense_3")
        
        self.out = Dense(1, activation='linear', name='out')
        
    def __call__(self, inputs, actions):
        x = tf.convert_to_tensor(inputs)
        x = self.dense_0(x)
        x = self.dense_1(x)
        y = self.dense_2(actions)
        x = self.concat([x, y])
        x = self.dense_3(x)
        return self.out(x)
