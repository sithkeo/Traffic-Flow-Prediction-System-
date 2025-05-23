import pandas as pd
import os

def summarise_final_metrics(path="output/metrics/metrics.csv", output_path="output/metrics/final_summary.csv"):
    if not os.path.exists(path):
        print("[ERROR] Metrics file not found.")
        return

    df = pd.read_csv(path)
    if df.empty:
        print("[WARNING] Metrics file is empty.")
        return

    # Get the latest row for each model
    summary = df.groupby("Model").tail(1).reset_index(drop=True)

    # Clean and reorder for export
    summary = summary[[
        "Model", "MAE", "RMSE", "MAE_%", "FinalLoss", "FinalValLoss",
        "Epochs", "BatchSize", "Site", "Timestamp"
    ]]

    # Print summary to terminal
    print("\n🧾 Final Model Performance Summary:\n")
    print(summary.to_string(index=False))

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"[INFO] Summary exported to: {output_path}")

if __name__ == "__main__":
    summarise_final_metrics()
