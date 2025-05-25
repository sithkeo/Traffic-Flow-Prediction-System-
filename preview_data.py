# preview_data.py
"""
Preview script for manually inspecting parsed traffic data with coordinates.
This is not a unit test. Run this to load and display the first few rows of a dataset.
"""

import os
import pandas as pd
from data_parser import parse_traffic_data

# Default metadata paths for coordinate matching (used if applicable)
listing_path = "data/SCATSSiteListingSpreadsheet_VicRoads.csv"
gps_path = "data/Traffic_Count_Locations_with_LONG_LAT.csv"

def preview():
    files = sorted([f for f in os.listdir("data") if f.endswith(".csv") or f.endswith(".xlsx")])
    if not files:
        print("No compatible files found in the data folder.")
        return

    print("Select a file to preview:")
    for idx, f in enumerate(files):
        print(f"  [{idx}] {f}")

    choice = input("Enter file index: ").strip()
    if not choice.isdigit() or int(choice) >= len(files):
        print("Invalid selection.")
        return

    filename = files[int(choice)]
    full_path = os.path.join("data", filename)
    try:
        df = parse_traffic_data(full_path, drop_zeros=False, listing_path=listing_path, gps_path=gps_path, max_rows=10)
        print(f"\nPreviewing: {filename}\n")
        print(df.head(10))
    except ValueError as ve:
        print(f"Failed to parse {filename}: {ve}")

if __name__ == "__main__":
    preview()
