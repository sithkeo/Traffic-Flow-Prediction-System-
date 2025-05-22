import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_evaluation_chart(metrics_csv="output/metrics/metrics.csv",
                              output_path="output/metrics/eval_comparison_chart.png"):
    """
    Generates a comparison chart of MAE, RMSE, and Final Validation Loss per model+site.

    Args:
        metrics_csv (str): Path to the metrics CSV file containing model performance results.
        output_path (str): Path to save the evaluation comparison chart.
    """
    if not os.path.exists(metrics_csv):
        print(f"[ERROR] Metrics file not found: {metrics_csv}")
        return

    df = pd.read_csv(metrics_csv)

    if df.empty:
        print("[WARNING] Metrics file is empty.")
        return

    # Keep the last row for each (Model, Site) combo
    latest = df.groupby(["Model", "Site"]).tail(1).reset_index(drop=True)
    latest["Model-Site"] = latest["Model"] + " (Site " + latest["Site"].astype(str) + ")"
    latest.set_index("Model-Site", inplace=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.25
    x = np.arange(len(latest))

    ax.bar(x - bar_width, latest["MAE"], width=bar_width, label="MAE")
    ax.bar(x, latest["RMSE"], width=bar_width, label="RMSE")
    ax.bar(x + bar_width, latest["FinalValLoss"], width=bar_width, label="Val Loss")

    ax.set_xticks(x)
    ax.set_xticklabels(latest.index, rotation=30, ha='right')
    ax.set_ylabel("Error")
    ax.set_title("Model Evaluation by Site (MAE, RMSE, Val Loss)")
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"[INFO] Evaluation chart saved to: {output_path}")

if __name__ == "__main__":
    generate_evaluation_chart()
