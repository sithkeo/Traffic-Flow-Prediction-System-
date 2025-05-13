import unittest
import pandas as pd
import os
from data_parser import parse_traffic_data

class TestTrafficParser(unittest.TestCase):
    # This test checks that a valid 2025-format CSV can be parsed into a time-series DataFrame
    # def test_parse_valid_csv(self):
    #     df = parse_traffic_data("test_data/VSDATA_20250501.csv")
    #     self.assertIsInstance(df, pd.DataFrame)
    #     self.assertIn("SCATS", df.columns)
    #     self.assertEqual(len(df.columns), 4)  # Expected columns: SCATS, Date, Time, Volume

    # This test ensures the parser can successfully handle a 2006-format Excel file (.xlsx)
    def test_parse_valid_xlsx(self):
        df = parse_traffic_data("test_data/Scats Data October 2006.xlsx")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df.columns), 4)

    # This test confirms that rows with zero vehicle counts are excluded when drop_zeros=True
    # def test_drop_zeros(self):
    #    df = parse_traffic_data("test_data/VSDATA_20250501.csv", drop_zeros=True)
    #    self.assertTrue((df["Volume"] != 0).all())

    # This test ensures unsupported file types raise a clear ValueError
    def test_reject_invalid_filetype(self):
        with self.assertRaises(ValueError) as context:
            parse_traffic_data("test_data/invalid_file.txt")
        self.assertIn("Unsupported file format", str(context.exception))

    # This test confirms the program warns users to convert .xls files to .xlsx manually
    def test_reject_xls_warning(self):
        with self.assertRaises(ValueError) as context:
            parse_traffic_data("test_data/Scats Data October 2006.xls")
        self.assertIn(".xls format is not supported", str(context.exception))

if __name__ == '__main__':
    unittest.main()
# This test suite is designed to validate the functionality of the traffic data parser.