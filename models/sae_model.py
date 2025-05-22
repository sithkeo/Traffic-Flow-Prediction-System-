# models/sae_model.py

"""
Stacked Autoencoder (SAE) model for traffic flow prediction using Keras.
This model uses a feedforward architecture with progressively reducing hidden layers,
mimicking the structure of autoencoders while optimizing for supervised regression tasks.
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

def build_model(input_shape):
    """
    Builds and returns a compiled Stacked Autoencoder (SAE)-inspired model
    for traffic prediction based on flattened input sequences.

    Architecture:
        - Flatten: Converts (SEQ_LEN, 1) sequence to a flat vector
        - Dense(64, 'sigmoid'): First hidden layer (encoder stage)
        - Dense(32, 'sigmoid'): Second hidden layer (deeper encoding)
        - Dense(16, 'sigmoid'): Bottleneck layer (deepest representation)
        - Dropout(0.2): Regularisation to prevent overfitting
        - Dense(1, 'linear'): Regression output (predicts traffic volume)

    Args:
        input_shape (tuple): Expected shape (e.g., (12, 1)) representing
                             (time steps, features)

    Returns:
        keras.Model: A compiled SAE model ready for training.

    """
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))         # Flatten time series input
    model.add(Dense(64, activation='sigmoid', name='hidden1'))  # Encoder layer 1
    model.add(Dense(32, activation='sigmoid', name='hidden2'))  # Encoder layer 2
    model.add(Dense(16, activation='sigmoid', name='hidden3'))  # Bottleneck layer
    model.add(Dropout(0.2))                             # Regularisation layer
    model.add(Dense(1, activation='linear'))            # Output: predicted traffic volume

    model.compile(optimizer='adam', loss='mse')         # Mean Squared Error for regression
    return model
