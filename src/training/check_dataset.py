
from pathlib import Path
import pandas as pd

# Project folder containing the processed CSV files
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_ROOT / "data" / "raw" / "processed"

files = [
    "train_processed.csv",
    "val_processed.csv"
]

for filename in files:
    csv_path = DATA_FOLDER / filename

    print("\n" + "=" * 50)
    print("Checking:", filename)

    if not csv_path.exists():
        print("ERROR: CSV file not found:", csv_path)
        continue

    df = pd.read_csv(csv_path)

    print("Number of rows:", len(df))
    print("Columns:", df.columns.tolist())

    # Check required columns
    required_columns = ["image_path", "text", "label"]

    if not all(column in df.columns for column in required_columns):
        print("ERROR: Required columns are missing.")
        continue

    # Validate labels
    labels = pd.to_numeric(df["label"], errors="coerce")

    invalid_labels = (
        labels.isna()
        | (labels < 0)
        | (labels >= 21)
        | (labels % 1 != 0)
    )

    print("Minimum label:", labels.min())
    print("Maximum label:", labels.max())
    print("Invalid labels:", int(invalid_labels.sum()))

    # Check image paths
    image_paths = df["image_path"].dropna().astype(str)

    missing_images = sum(
        not Path(path).is_file()
        for path in image_paths
    )

    print("Missing image files:", missing_images)

    # Show number of samples per class
    print("\nClass counts:")
    print(labels.value_counts().sort_index().to_string())

print("\nDataset validation finished.")
