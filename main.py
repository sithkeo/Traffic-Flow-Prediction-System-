# main.py

"""
Main entry point for training traffic prediction models.
Supports training by SCATS site or across all Boroondara sites.
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
from utils.plotting import plot_predictions

SEQ_LEN = 12  # Number of previous timesteps to use as input
TEST_SPLIT = 0.2  # Fraction of data used for testing

def load_data(csv_path):
    """Load CSV file and prepare a timestamp column."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")

    df = pd.read_csv(csv_path)
    if not {'SCATS', 'Date', 'Time', 'Volume'}.issubset(df.columns):
        raise ValueError("CSV is missing required columns: SCATS, Date, Time, Volume")

    # Create a full timestamp for sorting
    df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df.sort_values(by=["SCATS", "Timestamp"])
    return df

def prepare_data(df, site_id):
    """Filter data by site and prepare input/output sequences."""
    site_df = df[df["SCATS"].astype(str) == str(site_id)]
    if len(site_df) < SEQ_LEN + 1:
        raise ValueError("Not enough data for selected SCATS site.")

    volume = site_df["Volume"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(volume)

    # Generate input/output pairs for supervised learning
    X, y = create_sequences(scaled, SEQ_LEN)
    # Split full dataset into training and testing sets
    split = int(len(X) * (1 - TEST_SPLIT))
    return X[:split], y[:split], X[split:], y[split:], scaler

def prepare_data_all_sites(df):
    # This function aggregates data from all SCATS sites
    # and creates a unified training set across all locations.
    """Prepare training/testing sequences from all SCATS sites combined."""
    all_X, all_y = [], []
    for site_id, group in df.groupby("SCATS"):
        volume = group.sort_values("Timestamp")["Volume"].values.reshape(-1, 1)
        if len(volume) < SEQ_LEN + 1:
            print(f"[Skipping site {site_id}] Not enough data ({len(volume)} rows)")
            continue

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(volume)
        X, y = create_sequences(scaled, SEQ_LEN)

        # Ensure consistent shape and dtype
        if X.shape[1:] != (SEQ_LEN, 1):
            print(f"[Skipping site {site_id}] Inconsistent shape: {X.shape}")
            continue

        X = X.astype(np.float32)  # Ensure consistent dtype for stacking
        y = y.astype(np.float32)  # Ensure consistent dtype for stacking

        all_X.append(X)
        all_y.append(y)

    if not all_X:
        raise ValueError("No sites had sufficient data to build sequences.")

    try:
        X = np.vstack(all_X)  # Stack inputs across all sites along first dimension
        y = np.vstack(all_y)  # Stack targets consistently to match X samples
    except ValueError as ve:
        print("[DEBUG] Shapes of all_X:")
        for i, x in enumerate(all_X):
            print(f" - Site {i}: {x.shape}")
        raise RuntimeError(f"Shape mismatch when stacking: {ve}")

    split = int(len(X) * (1 - TEST_SPLIT))
    scaler = MinMaxScaler()
    scaler.fit(y)  # Fit scaler to full target set
    return X[:split], y[:split], X[split:], y[split:], scaler


# Ensure output subfolders exist
os.makedirs("output/losses", exist_ok=True)
os.makedirs("output/trained", exist_ok=True)
os.makedirs("output/metrics", exist_ok=True)

if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to preprocessed CSV from data_parser")
    parser.add_argument("--site", help="SCATS site ID to model")
    parser.add_argument("--all-sites", action="store_true", help="Train on all Boroondara sites")
    parser.add_argument("--models", nargs='+', default=["lstm_model"], help="Model module names (e.g. lstm_model)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    args = parser.parse_args()

    try:
        df = load_data(args.csv_path)
        print("Available SCATS site IDs:")
        print(df["SCATS"].value_counts())

        # Choose data preparation mode
        if args.all_sites:
            X_train, y_train, X_test, y_test, scaler = prepare_data_all_sites(df)
        elif args.site:
            X_train, y_train, X_test, y_test, scaler = prepare_data(df, args.site)
        else:
            raise ValueError("Must specify either --site or --all-sites")

        # Loop over each model specified
        for model_name in args.models:
            try:
                # Dynamically import the model module
                model_module = importlib.import_module(f"models.{model_name}")
                # Train and evaluate the model
                results = train_model(
                    model_module,
                    X_train, y_train,
                    X_test, y_test,
                    scaler,
                    batch_size=args.batch_size,
                    epochs=args.epochs
                )

                # print(f"[DEBUG] History content for {model_name}: {results.get('history')}")
                # print(f"[DEBUG] History keys: {getattr(results.get('history'), 'history', {}).keys()}")

                # Save per-epoch loss history to CSV
                if "history" in results:
                    loss_path = f"output/losses/{model_name}_loss.csv"
                    pd.DataFrame(results["history"].history).to_csv(loss_path, index=False)
                    print(f"[INFO] Loss history saved to {loss_path}")
                # Plot predictions vs actual values
                plot_predictions(results["predictions"], results["actuals"], model_name)

                # Save trained model to file
                trained_model_path = f"output/trained/{model_name}_trained.keras"
                results["model"].save(trained_model_path)
                print(f"[INFO] Model saved to {trained_model_path}")

                # Append training metrics to CSV log
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

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
