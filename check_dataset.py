
from pathlib import Path
import pandas as pd

# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "processed"
)

FILES = [
    "train_processed.csv",
    "val_processed.csv"
]

NUM_CLASSES = 21

# ==========================================
# 2. CHECK BOTH DATASETS
# ==========================================

for filename in FILES:

    print("\n" + "=" * 60)
    print(f"CHECKING DATASET: {filename}")
    print("=" * 60)

    csv_path = DATA_FOLDER / filename

    # Check whether CSV exists
    if not csv_path.exists():
        print("ERROR: CSV file not found!")
        print("Expected path:", csv_path)
        continue

    # Load CSV
    df = pd.read_csv(csv_path)

    print("Dataset loaded successfully.")
    print("Number of rows:", len(df))
    print("Number of columns:", len(df.columns))
    print("Columns:", df.columns.tolist())

    # ==========================================
    # 3. CHECK REQUIRED COLUMNS
    # ==========================================

    required_columns = [
        "image_path",
        "text",
        "label"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        print("ERROR: Missing columns:", missing_columns)
        continue

    print("\nRequired columns check: PASSED")

    # ==========================================
    # 4. CHECK MISSING VALUES
    # ==========================================

    print("\nMissing values:")
    print(df[required_columns].isnull().sum())

    # ==========================================
    # 5. CHECK LABELS
    # ==========================================

    labels = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    invalid_labels = (
        labels.isna()
        | (labels < 0)
        | (labels >= NUM_CLASSES)
        | (labels % 1 != 0)
    )

    print("\nMinimum label:", labels.min())
    print("Maximum label:", labels.max())
    print("Invalid labels:", int(invalid_labels.sum()))

    if invalid_labels.sum() == 0:
        print("Label validation: PASSED")
    else:
        print("Label validation: FAILED")

    # ==========================================
    # 6. CHECK IMAGE PATHS
    # ==========================================

    image_paths = df["image_path"]

    missing_path_values = int(image_paths.isnull().sum())

    existing_images = 0
    missing_images = 0

    for path in image_paths.dropna().astype(str):

        image_path = Path(path)

        if image_path.is_file():
            existing_images += 1
        else:
            missing_images += 1

    print("\nImage path values missing:", missing_path_values)
    print("Existing image files:", existing_images)
    print("Missing image files:", missing_images)

    if missing_path_values == 0 and missing_images == 0:
        print("Image validation: PASSED")
    else:
        print("Image validation: FAILED")

    # ==========================================
    # 7. CHECK TEXT VALUES
    # ==========================================

    empty_text = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    print("\nEmpty text rows:", int(empty_text.sum()))

    # ==========================================
    # 8. CLASS DISTRIBUTION
    # ==========================================

    print("\nClass distribution:")
    print(
        labels.value_counts()
        .sort_index()
        .to_string()
    )

    print("\nDataset check completed:", filename)

# ==========================================
# 9. FINISHED
# ==========================================

print("\n" + "=" * 60)
print("ALL DATASET CHECKS FINISHED")
print("=" * 60)
