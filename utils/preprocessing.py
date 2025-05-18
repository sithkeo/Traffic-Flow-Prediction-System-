# utils/preprocessing.py

"""
Utility for converting time-series data into supervised sequences.
"""

import numpy as np

def create_sequences(data, seq_len):
    """
    Convert 1D scaled array into supervised learning sequences.

    Args:
        data (np.ndarray): Normalised 1D array of traffic volume.
        seq_len (int): Number of timesteps per input sequence.

    Returns:
        X (np.ndarray): Sequences of shape (n_samples, seq_len, 1)
        y (np.ndarray): Targets of shape (n_samples,)
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)
