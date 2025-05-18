# utils/plotting.py

"""
Utility for plotting prediction vs actual results.
"""

import matplotlib.pyplot as plt

def plot_predictions(preds, actual, label="Model"):
    """
    Plot predicted vs. actual traffic volume.

    Args:
        preds (np.ndarray): Model predictions.
        actual (np.ndarray): True values.
        label (str): Title or model name.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(actual, label="Actual")
    plt.plot(preds, label="Predicted")
    plt.title(f"{label} Prediction vs Actual")
    plt.xlabel("Time Steps")
    plt.ylabel("Traffic Volume")
    plt.legend()
    plt.tight_layout()
    plt.show()
