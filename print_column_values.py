import argparse
import sys

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Print all values for a column in a CSV file.")
    parser.add_argument("csv_path", help="Path to the CSV file")
    parser.add_argument("column", help="Column name to print")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    if args.column not in df.columns:
        print("Column not found. Available columns:")
        for name in df.columns:
            print(name)
        return 1

    for value in df[args.column].tolist():
        print(value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
