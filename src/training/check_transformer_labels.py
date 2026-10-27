
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT / "data" / "raw" / "processed"
    / "train_processed.csv"
)
VAL_PATH = (
    PROJECT_ROOT / "data" / "raw" / "processed"
    / "val_processed.csv"
)
MAPPING_PATH = (
    PROJECT_ROOT / "data" / "raw" / "processed"
    / "label_mapping.csv"
)

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
mapping_df = pd.read_csv(MAPPING_PATH)

print("=" * 65)
print("TRANSFORMER LABEL MAPPING DIAGNOSTIC")
print("=" * 65)

print("\nMapping file columns:", mapping_df.columns.tolist())
print("\nMapping file preview:")
print(mapping_df.head(25).to_string(index=False))

# Check that all datasets contain the required columns.
required = {"label", "categories"}

for name, df in [
    ("Training", train_df),
    ("Validation", val_df),
]:
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} dataset is missing columns: {missing}"
        )

# Check label ranges and category counts.
for name, df in [
    ("Training", train_df),
    ("Validation", val_df),
]:
    print(f"\n{name} dataset")
    print("Rows:", len(df))
    print("Unique labels:", df["label"].nunique())
    print("Label range:", df["label"].min(), "to", df["label"].max())
    print("Missing labels:", df["label"].isna().sum())

    print("\nCategory counts:")
    print(
        df.groupby(["label", "categories"])
        .size()
        .to_string()
    )

# Check whether each numeric label identifies one category.
train_pairs = train_df[["label", "categories"]].drop_duplicates()
val_pairs = val_df[["label", "categories"]].drop_duplicates()

print("\nTraining label/category pairs:", len(train_pairs))
print("Validation label/category pairs:", len(val_pairs))

print("\nChecking label-to-category consistency...")

train_mapping = dict(
    train_pairs[["label", "categories"]].itertuples(
        index=False, name=None
    )
)
val_mapping = dict(
    val_pairs[["label", "categories"]].itertuples(
        index=False, name=None
    )
)

if train_mapping == val_mapping:
    print("PASS: Training and validation mappings match.")
else:
    print("WARNING: Training and validation mappings differ.")

    for label in sorted(set(train_mapping) | set(val_mapping)):
        train_name = train_mapping.get(label)
        val_name = val_mapping.get(label)

        if train_name != val_name:
            print(
                f"Label {label}: "
                f"train={train_name!r}, "
                f"validation={val_name!r}"
            )

expected_labels = set(range(21))

print("\nChecking expected labels 0 through 20...")

for name, df in [
    ("Training", train_df),
    ("Validation", val_df),
]:
    actual_labels = set(df["label"].dropna().astype(int).unique())
    missing_labels = expected_labels - actual_labels
    unexpected_labels = actual_labels - expected_labels

    print(f"{name} missing labels:", sorted(missing_labels))
    print(f"{name} unexpected labels:", sorted(unexpected_labels))

print("\nDIAGNOSTIC COMPLETE")
