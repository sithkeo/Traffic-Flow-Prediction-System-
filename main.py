# main.py

"""
Main entry point for training traffic prediction models.
Supports training by SCATS site or across all Boroondara sites.
Designed for instructional clarity for new students.
"""

import argparse
import importlib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Import custom training and utility functions
from train import train_model
from utils.preprocessing import create_sequences
from utils.plotting import plot_predictions, plot_loss_curves, plot_residuals

# Number of time steps in the input sequence (e.g., past 12 intervals)
SEQ_LEN = 12
# Fraction of data to be used as test set
TEST_SPLIT = 0.2

def load_data(csv_path):
    """
    Load the traffic volume dataset from CSV and prepare a Timestamp column.
    Ensures the correct columns are present and sorts the data.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")

    df = pd.read_csv(csv_path)
    if not {'SCATS', 'Date', 'Time', 'Volume'}.issubset(df.columns):
        raise ValueError("CSV is missing required columns: SCATS, Date, Time, Volume")

    # Combine Date and Time into a single datetime object
    df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    # Sort to ensure sequences are in order for each SCATS site
    df = df.sort_values(by=["SCATS", "Timestamp"])
    return df

def prepare_data(df, site_id):
    """
    Prepare training and testing sequences for a single SCATS site.
    Applies MinMax scaling and generates supervised learning pairs.
    """
    site_df = df[df["SCATS"].astype(str) == str(site_id)]
    if len(site_df) < SEQ_LEN + 1:
        raise ValueError("Not enough data for selected SCATS site.")

    volume = site_df["Volume"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(volume)

    # Create sliding window sequences
    X, y = create_sequences(scaled, SEQ_LEN)
    split = int(len(X) * (1 - TEST_SPLIT))
    return X[:split], y[:split], X[split:], y[split:], scaler

def prepare_data_all_sites(df):
    """
    Combine sequences from all SCATS sites into a single dataset.
    Filters out sites with too little data or inconsistent sequence shapes.
    """
    all_X, all_y = [], []
    for site_id, group in df.groupby("SCATS"):
        volume = group.sort_values("Timestamp")["Volume"].values.reshape(-1, 1)
        if len(volume) < SEQ_LEN + 1:
            print(f"[Skipping site {site_id}] Not enough data ({len(volume)} rows)")
            continue

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(volume)
        X, y = create_sequences(scaled, SEQ_LEN)

        if X.shape[1:] != (SEQ_LEN, 1):
            print(f"[Skipping site {site_id}] Inconsistent shape: {X.shape}")
            continue

        X = X.astype(np.float32)
        y = y.astype(np.float32)

        all_X.append(X)
        all_y.append(y)

    if not all_X:
        raise ValueError("No sites had sufficient data to build sequences.")

    try:
        X = np.vstack(all_X)
        y = np.vstack(all_y)
    except ValueError as ve:
        # Help debug by printing shapes before failure
        print("[DEBUG] Shapes of all_X:")
        for i, x in enumerate(all_X):
            print(f" - Site {i}: {x.shape}")
        raise RuntimeError(f"Shape mismatch when stacking: {ve}")

    split = int(len(X) * (1 - TEST_SPLIT))
    scaler = MinMaxScaler()
    scaler.fit(y)
    return X[:split], y[:split], X[split:], y[split:], scaler

# Ensure output directories exist
os.makedirs("output/losses", exist_ok=True)
os.makedirs("output/trained", exist_ok=True)
os.makedirs("output/metrics", exist_ok=True)

if __name__ == "__main__":
    # Set up command-line arguments for script usage
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to preprocessed CSV from data_parser")
    parser.add_argument("--site", help="SCATS site ID to model")
    parser.add_argument("--all-sites", action="store_true", help="Train on all Boroondara sites")
    parser.add_argument("--models", nargs='+', default=["gru_model", "lstm_model", "sae_model"], help="Model module names (e.g. lstm_model)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs. One epoch means the model sees the entire dataset once and updates itself based on errors.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    args = parser.parse_args()

    try:
        df = load_data(args.csv_path)
        print("Available SCATS site IDs:")
        print(df["SCATS"].value_counts())

        # Choose between one-site or all-sites training
        if args.all_sites:
            X_train, y_train, X_test, y_test, scaler = prepare_data_all_sites(df)
        elif args.site:
            X_train, y_train, X_test, y_test, scaler = prepare_data(df, args.site)
        else:
            raise ValueError("Must specify either --site or --all-sites")

        # Loop through each selected model
        for model_name in args.models:
            try:
                model_module = importlib.import_module(f"models.{model_name}")
                results = train_model(
                    model_module,
                    X_train, y_train,
                    X_test, y_test,
                    scaler,
                    batch_size=args.batch_size,
                    epochs=args.epochs
                )

                # Save loss history per epoch
                if "history" in results:
                    loss_path = f"output/losses/{model_name}_loss.csv"
                    pd.DataFrame(results["history"].history).to_csv(loss_path, index=False)
                    print(f"[INFO] Loss history saved to {loss_path}")

                # Debugging output before plotting
                print(f"[DEBUG] Plotting predictions for {model_name}")
                print(f"[DEBUG] Predictions shape: {results['predictions'].shape}")
                print(f"[DEBUG] Actuals shape: {results['actuals'].shape}")
                print(f"[DEBUG] First 5 predictions: {results['predictions'].flatten()[:5]}")
                print(f"[DEBUG] First 5 actuals: {results['actuals'].flatten()[:5]}")
                # Save prediction vs actual plot
                plot_predictions(results["predictions"], results["actuals"], model_name)
                plot_residuals(results["predictions"], results["actuals"], model_name)
                
                # Save predictions and actuals to .npy for GUI/web access
                os.makedirs("output/predictions", exist_ok=True)
                np.save(f"output/predictions/{model_name}_preds.npy", results["predictions"])

                # Save actual values once (if not already saved)
                actuals_path = "output/predictions/actual_values.npy"
                if not os.path.exists(actuals_path):
                    np.save(actuals_path, results["actuals"])

                # Save trained model to disk
                trained_model_path = f"output/trained/{model_name}_trained.keras"
                results["model"].save(trained_model_path)
                print(f"[INFO] Model saved to {trained_model_path}")

                # Append evaluation metrics to a summary CSV
                metrics_path = "output/metrics/metrics.csv"
                is_new_file = not os.path.exists(metrics_path)
                final_epoch = results["history"].epoch[-1]
                final_loss = results["history"].history["loss"][final_epoch]
                final_val_loss = results["history"].history["val_loss"][final_epoch]

                results_row = {
                    "Timestamp": datetime.now().isoformat(timespec='seconds'),
                    "Model": model_name,
                    "MAE": round(results["mae"], 2),
                    "RMSE": round(results["rmse"], 2),
                    "MAE_%": round(100 * results["mae"] / np.mean(results["actuals"]), 2),
                    "Site": args.site if args.site else "ALL",
                    "Epochs": args.epochs,
                    "BatchSize": args.batch_size,
                    "FinalLoss": round(final_loss, 6),
                    "FinalValLoss": round(final_val_loss, 6)
                }

                pd.DataFrame([results_row]).to_csv(
                    metrics_path, mode="a", header=is_new_file, index=False
                )
                print(f"[INFO] Metrics written to {metrics_path}")

            except Exception as model_err:
                print(f"[ERROR loading/training {model_name}] {model_err}")

        # After all models trained, generate loss curve plots for visual inspection
        plot_loss_curves()

        # Generate comparison chart across all models
        try:
            metrics_df = pd.read_csv("output/metrics/metrics.csv")
            if not metrics_df.empty:
                grouped = metrics_df.groupby("Model").tail(1)
                plt.figure(figsize=(8, 5))
                x = np.arange(len(grouped))
                width = 0.35
                plt.bar(x - width/2, grouped["MAE"], width, label="MAE")
                plt.bar(x + width/2, grouped["RMSE"], width, label="RMSE")
                plt.xticks(x, grouped["Model"])
                plt.ylabel("Error")
                plt.title("Model Comparison (Final MAE and RMSE)")
                plt.legend()
                plt.tight_layout()
                os.makedirs("output/metrics/", exist_ok=True)
                plt.savefig("output/metrics/model_comparison.png")
                plt.show()
                print("[INFO] Comparison chart saved to output/metrics/model_comparison.png")
        except Exception as compare_err:
            print(f"[WARNING] Could not generate comparison chart: {compare_err}")

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
