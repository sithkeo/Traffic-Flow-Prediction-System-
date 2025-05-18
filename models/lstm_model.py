# models/lstm_model.py

"""
Optimised LSTM model for traffic flow prediction using Keras.
This model includes stacked LSTM layers and dropout regularisation,
making it more robust than a basic single-layer LSTM.
"""

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def build_model(input_shape):
    """
    Builds and returns a compiled LSTM model with stacked architecture.

    Architecture:
        - LSTM(64) with return_sequences=True: captures full sequence
        - LSTM(32): reduces to summary vector
        - Dropout(0.2): prevents overfitting
        - Dense(1): predicts single output value (traffic volume)

    Args:
        input_shape (tuple): Input shape (e.g., (12, 1)) representing
                             (time steps, features)

    Returns:
        keras.Model: Compiled LSTM model ready for training.

    Note:
        This structure is more expressive than a single-layer LSTM,
        and follows best practices for sequence prediction tasks.
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))  # First LSTM layer returns sequence
    model.add(LSTM(32))                                                  # Second LSTM layer outputs summary vector
    model.add(Dropout(0.2))                                              # Regularisation
    model.add(Dense(1))                                                  # Output layer for regression

    model.compile(optimizer='adam', loss='mse')                          # Regression loss
    return model
