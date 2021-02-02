import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.initializers import VarianceScaling

class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        # Random distribution
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
    def __init__(self, num_actions, hidden):
        # Note: no tf.get_variable(), just simple Keras API!
        super().__init__('mlp_policy')
        self.normalize = kl.Lambda(lambda layer: layer / 255)    # normalize by 255
        self.conv1 = kl.Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)
        self.conv2 = kl.Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)
        self.conv3 = kl.Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)
        self.conv4 = kl.Conv2D(hidden, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)
        
        self.flatten = kl.Flatten()
        """
        self.d1 = kl.Dense(hidden, kernel_initializer=VarianceScaling(scale=2.), name="d1")
        self.d2 = kl.Dense(hidden / 2, kernel_initializer=VarianceScaling(scale=2.), name="d2")
        """
        self.value = kl.Dense(1, kernel_initializer=VarianceScaling(scale=2.), name="value")
        self.logits = kl.Dense(num_actions, kernel_initializer=VarianceScaling(scale=2.), name='policy_logits')
        
        self.dist = ProbabilityDistribution()

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
        """
        x = self.d1(x)
        x = self.d2(x)
        """
        return self.logits(x), self.value(x)
