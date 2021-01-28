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

def ppo_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        ratio = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(ratio * advantage, K.clip(ratio, min_value=1 - CLIP_RATIO, max_value=1 + CLIP_RATIO) * advantage) + \
                       ENTROPY_C * -(prob * K.log(K.clip(prob, K.epsilon(), 1-K.epsilon()))))
    return loss

def build_ppo_network(n_actions, learning_rate=LR, input_shape=INPUT_SHAPE, history_length=HISTORY_LENGHT, hidden=HIDDEN):
    """
    Builds a dueling DQN as a Keras model

    Arguments:
        n_actions: Number of possible actions
        learning_rate: Learning rate
        input_shape: Shape of the preprocessed image
        history_length: Number of historical frames to stack togheter
        hidden: Integer, Number of filters in the final convolutional layer. 

    Returns:
        A compiled Keras model
    """
    obs = Input(shape=(input_shape[0], input_shape[1], history_length))
    
    advantages = Input(shape=(1,))
    
    predictions = Input(shape=(n_actions,))
        
    
    x = Lambda(lambda layer: layer / 255)(obs)  # normalize by 255

    x = Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    x = Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    x = Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)
    x = Conv2D(hidden, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.), activation='relu', use_bias=False)(x)

    values_stream = Flatten()(x)
    values = Dense(1, kernel_initializer=VarianceScaling(scale=2.), name="values")(values_stream)

    probs_stream = Flatten()(x)
    probs = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.), activation='softmax', name="probs")(probs_stream)

    # Build model
    ppo_net = Model(inputs=[obs, advantages, predictions], outputs=[probs, values])
    ppo_net.compile(Adam(learning_rate), loss={'probs' : ppo_loss(advantages, predictions), 'values' : 'mean_squared_error'}, 
                 loss_weights={'probs': 1e-1, 'values': 1.})

    return ppo_net
