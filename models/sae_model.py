# models/sae_model.py

"""
Deep feedforward model for traffic flow prediction (EVA-style baseline).
This model follows a fully connected architecture as presented in supervisor documentation,
with 3 Dense layers of equal size and a final regression output layer.

Note: Unlike autoencoders, this version does not compress input; it uses large equal-sized layers.
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input

def build_model(input_shape):
    """
    Builds and returns a compiled feedforward model based on the supervisor's EVA diagram.

    Architecture:
        - Flatten: Converts (SEQ_LEN, 1) time series into a flat input vector
        - Dense(400, relu): Fully connected layer 1
        - Dense(400, relu): Fully connected layer 2
        - Dense(400, relu): Fully connected layer 3
        - Dropout(0.2): Regularisation to avoid overfitting
        - Dense(1): Final regression output

    Args:
        input_shape (tuple): Expected input shape (e.g., (12, 1))

    Returns:
        keras.Model: Compiled model ready for training.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))           # Explicit input declaration
    model.add(Flatten())                          # Flatten time-series input
    model.add(Dense(400, activation='relu'))      # Fully connected block 1
    model.add(Dense(400, activation='relu'))      # Fully connected block 2
    model.add(Dense(400, activation='relu'))      # Fully connected block 3
    model.add(Dropout(0.2))                       # Regularisation
    model.add(Dense(1))                           # Output: predict single value

    model.compile(optimizer='adam', loss='mse')   # Regression setup
    return model

# Documentation on Dense layer : https://keras.io/api/layers/core_layers/dense/
# Documentation on Dropout layer : https://keras.io/api/layers/core_layers/dropout/
# Documentation on Flatten layer : https://keras.io/api/layers/core_layers/flatten/
# Documentation on Sequential model : https://keras.io/api/models/sequential/