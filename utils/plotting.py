import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_predictions(preds: np.ndarray, actual: np.ndarray, label: str = "Model", save_dir: str = "output/predictions", show: bool = True) -> None:
    """
    Plot and save predicted vs actual traffic volume over time.
    """
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{label}_pred_vs_actual.png")

    plt.figure(figsize=(10, 4))
    plt.plot(actual, label="Actual", linewidth=2)
    plt.plot(preds, label="Predicted", linestyle="--", linewidth=2)
    plt.title(f"{label} Prediction vs Actual")
    plt.xlabel("Time Steps")
    plt.ylabel("Traffic Volume")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)

    if show:
        plt.show()
    else:
        plt.close()

    print(f"[INFO] Saved prediction plot to: {output_path}")


def plot_loss_curves(loss_dir: str = "output/losses", output_dir: str = "output/loss_curves") -> None:
    """
    Generates and saves training vs validation loss plots for each model loss file in `loss_dir`.
    Each CSV must contain 'loss' and 'val_loss' columns.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(loss_dir):
        if filename.endswith("_loss.csv"):
            model_name = filename.replace("_loss.csv", "")
            filepath = os.path.join(loss_dir, filename)

            try:
                df = pd.read_csv(filepath)
                if "loss" not in df.columns or "val_loss" not in df.columns:
                    print(f"[WARNING] Skipping {filename}: Missing 'loss' or 'val_loss' columns.")
                    continue

                plt.figure()
                plt.plot(df["loss"], label="Training Loss", linewidth=2)
                plt.plot(df["val_loss"], label="Validation Loss", linestyle="--", linewidth=2)
                plt.title(f"{model_name.upper()} Loss Curve")
                plt.xlabel("Epoch")
                plt.ylabel("Loss (MSE)")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                output_path = os.path.join(output_dir, f"{model_name}_loss_curve.png")
                plt.savefig(output_path)
                plt.close()
                print(f"[INFO] Saved loss curve for {model_name} to {output_path}")

            except Exception as e:
                print(f"[ERROR] Could not process {filename}: {e}")

def plot_residuals(preds: np.ndarray, actual: np.ndarray, label: str, save_dir: str = "output/predictions", show: bool = True) -> None:
    """
    Plot and save residuals (actual - predicted) over time for error analysis.

    Args:
        preds (np.ndarray): Model predictions.
        actual (np.ndarray): Ground truth values.
        label (str): Model name (used in filename and title).
        save_dir (str): Folder to save plot to.
        show (bool): Whether to display plot interactively.
    """
    os.makedirs(save_dir, exist_ok=True)
    residuals = actual.flatten() - preds.flatten()
    output_path = os.path.join(save_dir, f"{label}_residuals.png")

    plt.figure(figsize=(10, 4))
    plt.plot(residuals, label="Residuals", color="orange")
    plt.axhline(y=0, linestyle="--", color="gray")
    plt.title(f"{label} Residual Error Plot (Actual - Predicted)")
    plt.xlabel("Time Steps")
    plt.ylabel("Residual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)

    if show:
        plt.show()
    else:
        plt.close()

    print(f"[INFO] Residual plot saved to {output_path}")