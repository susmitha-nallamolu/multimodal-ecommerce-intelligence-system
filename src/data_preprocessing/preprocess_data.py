
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ============================================================
# STEP 1: DEFINE PROJECT FOLDERS
# ============================================================

# This script is currently located in the project root folder.
PROJECT_ROOT = Path(__file__).resolve().parent

# Folder containing the original CSV files and images.
DATA_FOLDER = PROJECT_ROOT / "data" / "raw"

# Folder containing training product images.
IMAGE_FOLDER = DATA_FOLDER / "train"

# Folder where processed CSV files will be saved.
OUTPUT_FOLDER = DATA_FOLDER / "processed"

# Create the processed folder if it does not exist.
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PROJECT PATH:", PROJECT_ROOT)
print("DATA PATH:", DATA_FOLDER)
print("IMAGE PATH:", IMAGE_FOLDER)
print("OUTPUT PATH:", OUTPUT_FOLDER)
print("=" * 60)


# ============================================================
# STEP 2: CHECK THAT INPUT FILES AND FOLDERS EXIST
# ============================================================

CSV_FILE = DATA_FOLDER / "train.csv"

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Training CSV not found: {CSV_FILE}"
    )

if not IMAGE_FOLDER.exists():
    raise FileNotFoundError(
        f"Training image folder not found: {IMAGE_FOLDER}"
    )

print("\nInput files and folders found successfully.")


# ============================================================
# STEP 3: LOAD THE DATASET
# ============================================================

df = pd.read_csv(CSV_FILE)

print("\nOriginal dataset shape:", df.shape)
print("Columns:", df.columns.tolist())


# ============================================================
# STEP 4: CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "ImgId",
    "title",
    "description",
    "categories"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )


# ============================================================
# STEP 5: HANDLE MISSING TEXT VALUES
# ============================================================

df["title"] = (
    df["title"]
    .fillna("Unknown product")
    .astype(str)
)

df["description"] = (
    df["description"]
    .fillna("")
    .astype(str)
)

# Combine title and description into one text column.
df["text"] = (
    df["title"].str.strip()
    + " "
    + df["description"].str.strip()
).str.strip()

print("\nMissing values handled.")
print("Combined title and description into 'text'.")


# ============================================================
# STEP 6: FIND AVAILABLE TRAINING IMAGES
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}

image_paths = {}

for image_path in IMAGE_FOLDER.rglob("*"):
    if (
        image_path.is_file()
        and image_path.suffix.lower() in image_extensions
    ):
        # The image filename without its extension.
        image_id = image_path.stem

        # Store its absolute path.
        image_paths[image_id] = str(image_path.resolve())

print("\nNumber of training images found:", len(image_paths))


# ============================================================
# STEP 7: MATCH DATASET RECORDS WITH IMAGES
# ============================================================

df["ImgId"] = df["ImgId"].astype(str).str.strip()

# Keep only records whose images actually exist.
df = df[df["ImgId"].isin(image_paths)].copy()

# Add the corresponding image file path to each record.
df["image_path"] = df["ImgId"].map(image_paths)

print("Records with matching images:", len(df))
print("Records without matching images were excluded.")


# ============================================================
# STEP 8: CHECK DATA BEFORE ENCODING
# ============================================================

if df.empty:
    raise ValueError(
        "No matching images found. Check image filenames and ImgId values."
    )

# Remove records that have no category label.
df = df.dropna(subset=["categories"]).copy()

if df["categories"].nunique() < 2:
    raise ValueError(
        "At least two categories are required for this split."
    )

df["categories"] = df["categories"].astype(str)

print("\nNumber of categories:", df["categories"].nunique())
print("\nProducts per category:")
print(df["categories"].value_counts())


# ============================================================
# STEP 9: CONVERT CATEGORY NAMES INTO NUMERIC LABELS
# ============================================================

label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(df["categories"])

print("\nCategory-to-label mapping:")
for category, label in zip(
    label_encoder.classes_,
    label_encoder.transform(label_encoder.classes_)
):
    print(f"{category} -> {label}")


# ============================================================
# STEP 10: SPLIT INTO TRAINING AND VALIDATION DATA
# ============================================================

train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

print("\nTraining records:", len(train_df))
print("Validation records:", len(val_df))


# ============================================================
# STEP 11: SAVE PROCESSED DATASETS
# ============================================================

train_file = OUTPUT_FOLDER / "train_processed.csv"
val_file = OUTPUT_FOLDER / "val_processed.csv"
mapping_file = OUTPUT_FOLDER / "label_mapping.csv"

train_df.to_csv(train_file, index=False)
val_df.to_csv(val_file, index=False)

label_mapping = pd.DataFrame({
    "category": label_encoder.classes_,
    "label": range(len(label_encoder.classes_))
})

label_mapping.to_csv(mapping_file, index=False)


# ============================================================
# STEP 12: VERIFY THAT FILES WERE SAVED
# ============================================================

expected_files = [
    train_file,
    val_file,
    mapping_file
]

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

for file_path in expected_files:
    if file_path.exists():
        print(
            f"SUCCESS: {file_path.name} "
            f"({file_path.stat().st_size:,} bytes)"
        )
    else:
        print(f"ERROR: File was not created: {file_path}")

print("\nFinal output folder:", OUTPUT_FOLDER.resolve())
print("Files currently in output folder:")

for file_path in OUTPUT_FOLDER.iterdir():
    print(" -", file_path.name)


