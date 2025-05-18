# data_parser.py
"""
Traffic Data Parser for SCATS Datasets
- Handles both legacy `.xlsx` format (2006) and new `.csv` format (2025+)
- Extracts 15-min interval volume data and reshapes it for time-series use
- Option to drop zero-volume records to exclude non-traffic periods or potential noise
- Warns user to manually convert `.xls` files (due to limited library support)
- Allows interactive selection of files from the `database/` folder for parsing and preview
- Adds coordinates from SCATS metadata and GPS dataset if available
"""

import pandas as pd
from datetime import timedelta
import os


def parse_traffic_data(filepath, drop_zeros=False, listing_path=None, gps_path=None, max_rows=None):
    """
    Parses traffic data from file (.xlsx or .csv) and returns reshaped time-series DataFrame.
    Supports both the 2006 Excel format and 2025 CSV format. Optionally merges metadata.



    Parameters:
        filepath (str): Path to the file.
        drop_zeros (bool): If True, removes rows where traffic volume is 0.
        listing_path (str): Optional path to SCATS site listing CSV.
        gps_path (str): Optional path to traffic location lat/lon CSV.
        max_rows (int): If provided, limits how many raw input rows are processed (for preview/testing).

    Returns:
        pd.DataFrame: Time-series formatted traffic data.
    """
    if filepath.endswith(".xls"):
        raise ValueError(
            "The .xls format is not supported in this environment. "
            "Please convert the file to .xlsx manually (use Excel or other tools)"
        )
    elif filepath.endswith(".xlsx"):
        df = _parse_2006_excel(filepath, max_rows=max_rows)
    elif filepath.endswith(".csv"):
        df = _parse_2025_csv(filepath, max_rows=max_rows)
    else:
        raise ValueError("Unsupported file format. Only .xlsx and .csv are supported.")

    if drop_zeros:
        df = df[df["Volume"] != 0]

    if listing_path and gps_path:
        df = add_coordinates(df, listing_path, gps_path)

    return df



def _parse_2006_excel(filepath, max_rows=None):
    """
    Parses the 2006-format Excel file with volume data recorded per SCATS site per day.
    """
    xls = pd.ExcelFile(filepath)
    df_raw = pd.read_excel(xls, sheet_name="Data", skiprows=1, nrows=max_rows)

    metadata_cols = df_raw.columns[:10]  # SCATS site metadata
    time_cols = df_raw.columns[10:]      # V00 to V95 (15-min intervals)

        # Remove rows missing site ID or date information
    df_clean = df_raw.dropna(subset=[metadata_cols[0], metadata_cols[9]])
        # Ensure site ID is treated as string
    df_clean[metadata_cols[0]] = df_clean[metadata_cols[0]].astype(str)
        # Convert the date column to datetime object
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



def _parse_2025_csv(filepath, max_rows=None):
    """
    Parses the newer 2025-format CSV file with per-detector traffic data.
    """
    df = pd.read_csv(filepath, nrows=max_rows)
        # Expected 96 volume columns for each 15-minute interval of a day
    required_cols = ["NB_SCATS_SITE", "QT_INTERVAL_COUNT"] + [f"V{i:02d}" for i in range(96)]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

        # Convert date column to datetime to support time expansion
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



def add_coordinates(df, listing_path, gps_path):
    """
    Adds latitude and longitude columns to a parsed DataFrame using SCATS location info and GPS data.
    Optimised for performance using dictionary-based keyword lookup.
    """
      # Load SCATS listing metadata and GPS reference dataset
      # Dynamically determine whether to skip rows by checking if expected header is present
    with open(listing_path, 'r', encoding='utf-8') as f:
        header_line = f.readline()

    if "Site Number" in header_line and "Location Description" in header_line:
        listing = pd.read_csv(listing_path)
    else:
        listing = pd.read_csv(listing_path, skiprows=9)
    gps = pd.read_csv(gps_path)

      # Prepare and clean SCATS listing
    listing = listing.rename(columns={"Site Number": "SCATS", "Location Description": "Location"})
    listing["SCATS"] = pd.to_numeric(listing["SCATS"], errors="coerce")
    df = df.merge(listing[["SCATS", "Location"]], on="SCATS", how="left")

      # Prepare uppercase strings for keyword matching
    df["Location_UPPER"] = df["Location"].astype(str).str.upper()
    gps["SITE_DESC_UPPER"] = gps["SITE_DESC"].astype(str).str.upper()

      # Build lookup dictionary using first keyword from GPS site descriptions
    gps_lookup = {}
    for _, row in gps.iterrows():
        keyword = row["SITE_DESC_UPPER"].split()[0]
        if keyword not in gps_lookup:
            gps_lookup[keyword] = (row["X"], row["Y"])

      # Vectorised coordinate assignment using keyword map
    def resolve_coords(location):
        if not isinstance(location, str) or len(location.strip()) == 0:
            return pd.Series([None, None])
        keyword = location.split()[0]
        return gps_lookup.get(keyword, (None, None))

      # Assign the matched coordinates to new Longitude and Latitude columns
    coords = df["Location_UPPER"].map(lambda loc: resolve_coords(loc))
    df[["Longitude", "Latitude"]] = pd.DataFrame(coords.tolist(), index=df.index)
    return df
    

def step_through_directory(directory_path="database", drop_zeros=False, listing_path=None, gps_path=None):
    """
    Prompts the user to step through files in a directory and select which ones to parse.
    Each selected file is parsed using parse_traffic_data().
    """
    files = sorted([f for f in os.listdir(directory_path) if f.endswith(".csv") or f.endswith(".xlsx")])
    if not files:
        print("No compatible files found in directory.")
        return

    selected = []
    print("Available files:")
    for idx, f in enumerate(files):
        print(f"  [{idx}] {f}")

    while True:
        choice = input("Enter the index of a file to parse (or 'done' to finish): ").strip()
        if choice.lower() == 'done':
            break
        if not choice.isdigit() or int(choice) >= len(files):
            print("Invalid choice. Try again.")
            continue

        filename = files[int(choice)]
        full_path = os.path.join(directory_path, filename)
        try:
            df = parse_traffic_data(full_path, drop_zeros=drop_zeros, listing_path=listing_path, gps_path=gps_path)
            print(f"Successfully parsed {filename}. Showing preview:")
            print(df.head())
            selected.append((filename, df))
        except ValueError as e:
            print(f"Failed to parse {filename}: {e}")

    print(f"Parsed {len(selected)} file(s).")
    return selected

# Entry point: if script is run directly, handle command-line arguments and start processing
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python data_parser.py <file|folder> [--drop-zeros] [--listing path] [--gps path] [--max-rows N]")
    else:
            # First argument is expected to be a file or folder path
        path = sys.argv[1]
            # Optional flag to exclude zero-volume rows
        drop = "--drop-zeros" in sys.argv
            # Default metadata file paths for cross-referencing (used if --listing/--gps not provided)
        default_listing = "database/SCATSSiteListingSpreadsheet_VicRoads.csv"
        listing_path = next((sys.argv[i + 1] for i, x in enumerate(sys.argv) if x == "--listing"), default_listing)
        default_gps = "database/Traffic_Count_Locations_with_LONG_LAT.csv"  # Default GPS dataset path
        gps_path = next((sys.argv[i + 1] for i, x in enumerate(sys.argv) if x == "--gps"), default_gps)
        max_rows = next((int(sys.argv[i + 1]) for i, x in enumerate(sys.argv) if x == "--max-rows"), None)

        try:
                # If a folder is given, prompt the user to pick which files to process
            if os.path.isdir(path):
                step_through_directory(path, drop_zeros=drop, listing_path=listing_path, gps_path=gps_path)
                # Otherwise parse a single file directly
            else:
                df = parse_traffic_data(path, drop_zeros=drop, listing_path=listing_path, gps_path=gps_path, max_rows=max_rows)
            print(df.head())
        except ValueError as ve:
            print(f"Error: {ve}")
