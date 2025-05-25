# test_data_parser.py
"""
Unit tests for SCATS traffic data parser.
Ensures that valid and invalid files are handled as expected,
and that optional behaviours (like drop_zeros) work correctly.
"""
import unittest
import os
import pandas as pd
from data_parser import parse_traffic_data, add_coordinates

class TestTrafficParser(unittest.TestCase):

        # Ensure that a valid CSV is parsed into a proper DataFrame with expected columns.
    def test_parse_valid_csv(self):
        # Check that the file exists before attempting to parse
        df = parse_traffic_data("data/VSDATA_20250501.csv")
        # Check that the result is a DataFrame.
        self.assertIsInstance(df, pd.DataFrame)
        # Check that the DataFrame has at least 4 columns, as expected.
        self.assertGreaterEqual(len(df.columns), 4)
        # Ensure that the columns exists and contains numeric data.
        self.assertIn("SCATS", df.columns)
        self.assertIn("Volume", df.columns)

        # Ensure that a valid XLSX file (2006 format) is parsed properly.
    def test_parse_valid_xlsx(self):
        df = parse_traffic_data("data/Scats Data October 2006.xlsx")
        self.assertIsInstance(df, pd.DataFrame)

        # Check expected columns for legacy format with GPS enrichment
        expected_columns = {"SCATS", "Location", "Date", "Time", "Volume", "Longitude", "Latitude"}
        self.assertTrue(expected_columns.issubset(set(df.columns)),
                        f"Expected columns missing: {expected_columns - set(df.columns)}")

        self.assertGreater(len(df), 0, "Parsed DataFrame should not be empty")

        # Check that rows with zero volume are correctly dropped when drop_zeros=True.
    def test_drop_zeros(self):
        df = parse_traffic_data("data/VSDATA_20250501.csv", drop_zeros=True)
        self.assertTrue((df["Volume"] != 0).all())

        # Test that unsupported file types raise a ValueError.
    def test_reject_invalid_filetype(self):
        with self.assertRaises(ValueError) as context:
            parse_traffic_data("test_data/invalid_file.txt")
        self.assertIn("Unsupported file format", str(context.exception))

        # Test that .xls files are explicitly rejected with conversion warning.
    def test_reject_xls_warning(self):
        with self.assertRaises(ValueError) as context:
            parse_traffic_data("data/Scats Data October 2006.xls")
        self.assertIn(".xls format is not supported", str(context.exception))

        # Test coordinate augmentation for a simple SCATS dataframe.
    def test_add_coordinates_integration(self):
        df = pd.DataFrame({
            "SCATS": [100],
            "Date": ["2025-05-01"],
            "Time": ["08:00:00"],
            "Volume": [100]
        })
        enriched = add_coordinates(
            df,
            "data/SCATSSiteListingSpreadsheet_VicRoads.csv",
            "data/Traffic_Count_Locations_with_LONG_LAT.csv"
        )
        self.assertIn("Longitude", enriched.columns)
        self.assertIn("Latitude", enriched.columns)
        self.assertFalse(enriched["Longitude"].isna().any())
        self.assertFalse(enriched["Latitude"].isna().any())

        # Limit how many rows are processed from input
    def test_max_rows_limit(self):
        file_path = "data/vsdata_20250501.csv"
        self.assertTrue(os.path.exists(file_path), f"Missing test file: {file_path}")
        df = parse_traffic_data(file_path, max_rows=3)
        # Each row expands to 96 intervals
        self.assertLessEqual(len(df), 3 * 96, "Too many rows returned for max_rows=3")

        # Ensure missing required columns trigger parsing errors
    def test_missing_required_column_raises_error(self):
        bad_df = pd.DataFrame({
            "NB_SCATS_SITE": ["101"],
            "V00": [10], "V01": [20], "V02": [15]
        })
        temp_path = "data/temp_missing_column.csv"
        bad_df.to_csv(temp_path, index=False)
        # Attempt to parse the invalid CSV
        with self.assertRaises(ValueError) as context:
            parse_traffic_data(temp_path)
        # Check for specific error messages indicating missing columns or format issues
        self.assertTrue(
            any(msg in str(context.exception) for msg in [
                "Missing required column",
                "Unrecognised CSV format"
            ]),
            "Expected a missing column or format error"
        )
        # Clean up temporary file
        os.remove(temp_path)

        # Verify that missing GPS file causes failure
    def test_missing_gps_file_raises(self):
        file_path = "data/vsdata_20250501.csv"
        self.assertTrue(os.path.exists(file_path), f"Missing test file: {file_path}")
        with self.assertRaises(FileNotFoundError):
            parse_traffic_data(
                file_path,
                listing_path="data/SCATSSiteListingSpreadsheet_VicRoads.csv",
                gps_path="data/does_not_exist.csv"
            )


if __name__ == '__main__':
        # Run all unit tests
    unittest.main()
