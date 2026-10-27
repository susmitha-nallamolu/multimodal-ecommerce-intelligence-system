
from pathlib import Path
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "processed"
    / "train_processed.csv"
)

df = pd.read_csv(CSV_PATH)

print("Total training records:", len(df))

# Check whether image paths exist
missing_images = []
invalid_images = []

for _, row in df.head(200).iterrows():
    image_path = Path(row["image_path"])

    if not image_path.exists():
        missing_images.append(str(image_path))
        continue

    try:
        with Image.open(image_path) as image:
            image.verify()
    except Exception:
        invalid_images.append(str(image_path))

print("Images checked:", min(200, len(df)))
print("Missing image files:", len(missing_images))
print("Invalid image files:", len(invalid_images))

if missing_images:
    print("\nExample missing path:", missing_images[0])

if invalid_images:
    print("\nExample invalid path:", invalid_images[0])

# Inspect a product record
row = df.iloc[0]

print("\nFirst product details:")
print("Product ID:", row["ImgId"])
print("Title:", row["title"])
print("Category:", row["categories"])
print("Label:", row["label"])
print("Image path:", row["image_path"])
print("Text matches title:", str(row["title"]) in str(row["text"]))
