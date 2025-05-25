# generate_predicted_site_volumes.py

"""
Generate average predicted traffic volumes for each SCATS site using a trained model.
Saves site_id → predicted_volume mapping as a CSV for use in routing systems.
"""

import argparse
import numpy as np
import pandas as pd
import os
from keras.models import load_model
from utils.preprocessing import create_sequences
from sklearn.preprocessing import MinMaxScaler

SEQ_LEN = 12

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df.sort_values(by=["SCATS", "Timestamp"])
    return df

def predict_site_volumes(df, model_path, output_csv):
    model = load_model(model_path)
    site_predictions = {}

    for site_id, group in df.groupby("SCATS"):
        volume = group.sort_values("Timestamp")["Volume"].values.reshape(-1, 1)
        if len(volume) < SEQ_LEN + 1:
            continue

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(volume)
        X, y = create_sequences(scaled, SEQ_LEN)

        if X.shape[0] == 0:
            continue

        preds = model.predict(X, verbose=0)
        preds_rescaled = scaler.inverse_transform(preds)
        site_predictions[site_id] = float(np.mean(preds_rescaled))

    # Save to CSV
    df_out = pd.DataFrame(list(site_predictions.items()), columns=["SCATS", "PredictedVolume"])
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"[INFO] Saved predicted site volumes to {output_csv}")

if __name__ == "__main__":
    print("=== Generate Predicted Site Volumes ===")
    default_csv = "output/Scats_Data_October_2006_parsed.csv"
    model_options = {
        "gru": "output/trained/gru_model_trained.keras",
        "lstm": "output/trained/lstm_model_trained.keras",
        "sae": "output/trained/sae_model_trained.keras"
    }

    print(f"\n[1] Use default CSV: {default_csv}")
    print("    → Predict for ALL sites using ALL models")
    print("[2] Predict for ALL sites using one selected model")
    print("[3] Predict for a SPECIFIC site using one selected model")
    print("[4] Custom CSV and model selection")

    choice = input("\nSelect an option [1-4]: ").strip()

    if choice == "1":
        csv_path = default_csv
        df = load_data(csv_path)

        for model_key, model_path in model_options.items():
            output_csv = f"output/predicted/{model_key}_site_predictions.csv"
            predict_site_volumes(df, model_path, output_csv)

    elif choice == "2":
        csv_path = default_csv
        df = load_data(csv_path)

        print("Available models: gru, lstm, sae")
        model_key = input("Select model to use: ").strip().lower()
        model_path = model_options.get(model_key, model_options["gru"])
        output_csv = f"output/predicted/{model_key}_site_predictions.csv"
        predict_site_volumes(df, model_path, output_csv)

    elif choice == "3":
        csv_path = default_csv
        df = load_data(csv_path)

        print("Available models: gru, lstm, sae")
        model_key = input("Select model to use: ").strip().lower()
        model_path = model_options.get(model_key, model_options["gru"])
        selected_site = input("Enter SCATS site ID to predict: ").strip()
        df_site = df[df["SCATS"].astype(str) == selected_site]

        if df_site.empty:
            print(f"[ERROR] No data found for site {selected_site}")
        else:
            output_csv = f"output/predicted/{model_key}_site_{selected_site}_prediction.csv"
            predict_site_volumes(df_site, model_path, output_csv)

    elif choice == "4":
        csv_path = input("Enter path to CSV: ").strip()
        df = load_data(csv_path)

        print("Available models: gru, lstm, sae")
        model_key = input("Select model to use: ").strip().lower()
        model_path = model_options.get(model_key, model_options["gru"])

        site_scope = input("Predict for [A]ll sites or [S]pecific site? [A/S]: ").strip().lower()
        if site_scope == "s":
            selected_site = input("Enter SCATS site ID to predict: ").strip()
            df = df[df["SCATS"].astype(str) == selected_site]
            output_csv = f"output/predicted/{model_key}_site_{selected_site}_prediction.csv"
        else:
            output_csv = f"output/predicted/{model_key}_site_predictions.csv"

        predict_site_volumes(df, model_path, output_csv)

    else:
        print("[ERROR] Invalid option. Please run again.")
