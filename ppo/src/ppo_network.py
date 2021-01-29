import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input,
                                     Lambda, Subtract)
import tensorflow.keras.backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import sys
sys.path.append("../src/")
from config import *
from pong_wrapper import *

class PpoNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.normalize = Lambda(lambda layer: layer / 255, name="Normalize")    # normalize by 255
        self.conv1 = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)
        self.conv2 = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)
        self.conv3 = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)
        self.conv4 = Conv2D(HIDDEN, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)

        self.flatten = Flatten()

        self.value = Dense(1, kernel_initializer=VarianceScaling(scale=2.), name="value", activation=None)
        self.probs = Dense(NUM_ACTIONS, kernel_initializer=VarianceScaling(scale=2.), name='logits', activation='softmax')
    
    def call(self, inputs, **kwargs):
        # Inputs is a numpy array, convert to a tensor.
        x = tf.convert_to_tensor(inputs)
        # Separate hidden layers from the same input tensor.
        x = self.normalize(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        
        return self.probs(x), self.value(x)
