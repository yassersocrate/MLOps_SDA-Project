import pandas as pd
import argparse


def count_defaults(csv_file, target_column):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(csv_file)

    # Check if the target column exists
    if target_column not in data.columns:
        print(f"Column '{target_column}' not found in the CSV file.")
        return

    # Count the number of 0s and 1s in the target column
    default_counts = data[target_column].value_counts()

    # Display the counts
    print(f"Count of defaults in '{target_column}' column:")
    print(f"0: {default_counts.get(0, 0)}")
    print(f"1: {default_counts.get(1, 0)}")


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(
        description="Count loan default occurrences (0 and 1) in a CSV file."
    )
    parser.add_argument("csv_file", help="Path to the input CSV file.")
    parser.add_argument(
        "--target_column",
        default="default",
        help="Column name containing the default values (0 or 1).",
    )

    args = parser.parse_args()

    # Call the count_defaults function
    count_defaults(args.csv_file, args.target_column)
