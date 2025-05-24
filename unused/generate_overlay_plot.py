import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_model_predictions(predictions_dict, actual, timestamps=None,
                           output_path="output/predictions/prediction_overlay.png"):
    """
    Plots actual traffic flow vs predictions from multiple models.

    Args:
        predictions_dict (dict): Keys are model names, values are np.ndarrays of predicted values.
        actual (np.ndarray): Ground truth values.
        timestamps (list or np.ndarray): Optional time axis labels.
        output_path (str): Filepath to save the figure.
    """
    plt.figure(figsize=(12, 6))

    # Plot ground truth
    plt.plot(actual, label="True Data", linewidth=2, color='tab:blue')

    # Plot predictions
    for model, preds in predictions_dict.items():
        plt.plot(preds, label=model)

    if timestamps is not None and len(timestamps) == len(actual):
        plt.xticks(
            np.linspace(0, len(timestamps)-1, 10),
            [timestamps[int(i)] for i in np.linspace(0, len(timestamps)-1, 10)],
            rotation=30
        )
        plt.xlabel("Time of Day")
    else:
        plt.xlabel("Time Step")

    plt.ylabel("Flow")
    plt.title("Model Predictions vs True Traffic Flow")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()
    print(f"[INFO] Overlay plot saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Replace with actual file paths or logic to load your predictions and true values
    try:
        lstm = np.load("output/predictions/lstm_model_preds.npy")
        gru = np.load("output/predictions/gru_model_preds.npy")
        sae = np.load("output/predictions/sae_model_preds.npy")
        actual = np.load("output/predictions/actual_values.npy")

        plot_model_predictions(
            {
                "LSTM": lstm,
                "GRU": gru,
                "SAEs": sae
            },
            actual=actual
        )
    except Exception as e:
        print(f"[ERROR] Could not load prediction arrays: {e}")
