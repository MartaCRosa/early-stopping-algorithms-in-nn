import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam

def create_model(input_dim, hidden_nodes, output_dim, activation='sigmoid'):
    
    model = Sequential([
        Dense(hidden_nodes, activation=activation, input_dim=input_dim, 
              kernel_initializer=RandomNormal(mean=0.0, stddev=0.05)),
        Dense(output_dim, activation='softmax', 
              kernel_initializer=RandomNormal(mean=0.0, stddev=0.05))
    ])

    learning_rate = 0.001  # 0.001 0.0005

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),  # RProp optimizer
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall', 'mse']
    )
    return model

