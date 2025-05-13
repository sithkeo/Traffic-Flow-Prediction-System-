# data_parser.py
"""
Traffic Data Parser for SCATS Datasets
- Handles both legacy `.xlsx` format (2006) and new `.csv` format (2025+)
- Extracts 15-min interval volume data and reshapes it for time-series use
- Option to drop zero-volume records to exclude non-traffic periods or potential noise
- Warns user to manually convert `.xls` files (due to limited library support)
"""

import pandas as pd
from datetime import timedelta
import os


def parse_traffic_data(filepath, drop_zeros=False):
    """
    Parses traffic data from file (.xlsx or .csv) and returns reshaped time-series DataFrame.
    Supports both the 2006 Excel format and 2025 CSV format.

    Parameters:
        filepath (str): Path to the file.
        drop_zeros (bool): If True, removes rows where traffic volume is 0.

    Returns:
        pd.DataFrame: Time-series formatted traffic data.
    """
    if filepath.endswith(".xls"):
        raise ValueError(
            "The .xls format is not supported in this environment. "
            "Please convert the file to .xlsx manually (use Excel or other tools)"
        )
    elif filepath.endswith(".xlsx"):
        df = _parse_2006_excel(filepath)
    elif filepath.endswith(".csv"):
        df = _parse_2025_csv(filepath)
    else:
        raise ValueError("Unsupported file format. Only .xlsx and .csv are supported.")

    if drop_zeros:
        df = df[df["Volume"] != 0]

    return df


def _parse_2006_excel(filepath):
    """
    Parses the 2006-format Excel file with volume data recorded per SCATS site per day.
    """
    xls = pd.ExcelFile(filepath)
    df_raw = pd.read_excel(xls, sheet_name="Data", skiprows=1)

    metadata_cols = df_raw.columns[:10]  # SCATS site metadata
    time_cols = df_raw.columns[10:]      # V00 to V95 (15-min intervals)

    # Filter out incomplete rows and convert types
    df_clean = df_raw.dropna(subset=[metadata_cols[0], metadata_cols[9]])
    df_clean[metadata_cols[0]] = df_clean[metadata_cols[0]].astype(str)
    df_clean[metadata_cols[9]] = pd.to_datetime(df_clean[metadata_cols[9]], errors='coerce')

    # Expand each row into 96 rows (1 per time interval)
    all_rows = []
    for _, row in df_clean.iterrows():
        site_id = row[metadata_cols[0]]
        date = row[metadata_cols[9]]
        for i, col in enumerate(time_cols):
            time = (date + timedelta(minutes=15 * i))
            all_rows.append([site_id, time.date(), time.time(), row[col]])

    return pd.DataFrame(all_rows, columns=["SCATS", "Date", "Time", "Volume"])


def _parse_2025_csv(filepath):
    """
    Parses the newer 2025-format CSV file with per-detector traffic data.
    """
    df = pd.read_csv(filepath)

    # Confirm required columns are present
    required_cols = ["NB_SCATS_SITE", "QT_INTERVAL_COUNT"] + [f"V{i:02d}" for i in range(96)]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["QT_INTERVAL_COUNT"] = pd.to_datetime(df["QT_INTERVAL_COUNT"], errors="coerce")
    all_rows = []

    # Expand each detector-day row into 96 rows of timestamped volume data
    for _, row in df.iterrows():
        site_id = row["NB_SCATS_SITE"]
        date = row["QT_INTERVAL_COUNT"]
        for i in range(96):
            col = f"V{i:02d}"
            time = (date + timedelta(minutes=15 * i))
            all_rows.append([site_id, time.date(), time.time(), row[col]])

    return pd.DataFrame(all_rows, columns=["SCATS", "Date", "Time", "Volume"])


# Example usage (if running as script)
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python data_parser.py <file.xlsx|file.csv> [--drop-zeros]")
    else:
        file_path = sys.argv[1]
        drop = "--drop-zeros" in sys.argv
        try:
            parsed_df = parse_traffic_data(file_path, drop_zeros=drop)
            print(parsed_df.head())
        except ValueError as ve:
            print(f"Error: {ve}")
