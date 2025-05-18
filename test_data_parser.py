# test_data_parser.py
"""
Unit tests for SCATS traffic data parser.
Ensures that valid and invalid files are handled as expected,
and that optional behaviours (like drop_zeros) work correctly.
"""

import unittest
import warnings
import pandas as pd
from data_parser import parse_traffic_data, add_coordinates

    # Suppress openpyxl header/footer warnings during Excel reads
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

class TestTrafficParser(unittest.TestCase):

        # Ensure that a valid CSV is parsed into a proper DataFrame with expected columns.
    def test_parse_valid_csv(self):
        df = parse_traffic_data("database/VSDATA_20250501.csv")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("SCATS", df.columns)
        self.assertEqual(len(df.columns), 4)

        # Ensure that a valid XLSX file (2006 format) is parsed properly.
    def test_parse_valid_xlsx(self):
        df = parse_traffic_data("database/Scats Data October 2006.xlsx")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df.columns), 4)

        # Check that rows with zero volume are correctly dropped when drop_zeros=True.
    def test_drop_zeros(self):
        df = parse_traffic_data("database/VSDATA_20250501.csv", drop_zeros=True)
        self.assertTrue((df["Volume"] != 0).all())

        # Test that unsupported file types raise a ValueError.
    def test_reject_invalid_filetype(self):
        with self.assertRaises(ValueError) as context:
            parse_traffic_data("test_data/invalid_file.txt")
        self.assertIn("Unsupported file format", str(context.exception))

        # Test that .xls files are explicitly rejected with conversion warning.
    def test_reject_xls_warning(self):
        with self.assertRaises(ValueError) as context:
            parse_traffic_data("database/Scats Data October 2006.xls")
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
            "database/SCATSSiteListingSpreadsheet_VicRoads.csv",
            "database/Traffic_Count_Locations_with_LONG_LAT.csv"
        )
        self.assertIn("Longitude", enriched.columns)
        self.assertIn("Latitude", enriched.columns)
        self.assertFalse(enriched["Longitude"].isna().any())
        self.assertFalse(enriched["Latitude"].isna().any())


if __name__ == '__main__':
        # Run all unit tests
    unittest.main()
