import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import RMSprop

def create_model(input_dim, hidden_nodes, output_dim, activation='sigmoid'):
    """
    Creates and compiles a neural network with one hidden layer using RProp.

    Parameters:
    - input_dim: Number of input features.
    - hidden_nodes: Number of nodes in the hidden layer.
    - output_dim: Number of output neurons.
    - activation: Activation function for the hidden layer.

    Returns:
    - A compiled Keras model.
    """
    model = Sequential([
        Dense(hidden_nodes, activation=activation, input_dim=input_dim, 
              kernel_initializer=RandomNormal(mean=0.0, stddev=0.05)),
        Dense(output_dim, activation='softmax', 
              kernel_initializer=RandomNormal(mean=0.0, stddev=0.05))
    ])

    model.compile(
        optimizer=RMSprop(learning_rate=0.001),  # RProp-like optimizer
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
