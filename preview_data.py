# preview_data.py
"""
Preview script for manually inspecting parsed traffic data with coordinates.
This is not a unit test. Run this to load and display the first few rows of a dataset.
"""

import pandas as pd
from data_parser import parse_traffic_data, add_coordinates

# Define file paths
input_file = "database/VSDATA_20250501.csv"
listing_path = "database/SCATSSiteListingSpreadsheet_VicRoads.csv"
gps_path = "database/Traffic_Count_Locations_with_LONG_LAT.csv"

# Parse and enrich the data
df = parse_traffic_data(input_file, drop_zeros=False, max_rows=1)
df = add_coordinates(df, listing_path, gps_path)

# Preview the first 10 rows
print("Previewing first 10 rows of enriched traffic data:")
print(df.head(10))