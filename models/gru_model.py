# models/gru_model.py

"""
GRU (Gated Recurrent Unit) model for traffic flow prediction using Keras.
This is an alternative to LSTM with fewer parameters and faster training.
"""

from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout

def build_model(input_shape):
    """
    Builds and returns a compiled GRU model with stacked architecture.

    Architecture:
        - GRU(64) with return_sequences=True: captures full input sequence
        - GRU(32): compresses to summary vector
        - Dropout(0.2): regularises model to prevent overfitting
        - Dense(1): outputs single predicted value (regression)

    Args:
        input_shape (tuple): Shape of the input sequence (e.g., (12, 1))

    Returns:
        keras.Model: Compiled GRU model ready for training.
    """
    model = Sequential()
    model.add(GRU(64, return_sequences=True, input_shape=input_shape))
    model.add(GRU(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Linear activation for continuous output

    model.compile(optimizer='adam', loss='mse')
    return model