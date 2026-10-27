
from pathlib import Path
import pandas as pd

# Dataset paths
data_folder = Path("data/raw")
train_csv = data_folder / "train.csv"
train_images_folder = data_folder / "train"

# Load training metadata
df = pd.read_csv(train_csv)

# Collect image filenames
image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

image_files = [
    path for path in train_images_folder.rglob("*")
    if path.is_file()
    and path.suffix.lower() in image_extensions
]

# Create a set of image filenames without extensions
image_ids = {path.stem for path in image_files}

# Check which products have matching images
df["has_image"] = df["ImgId"].astype(str).isin(image_ids)

print("Total training records:", len(df))
print("Total image files:", len(image_files))
print("Records with matching images:", df["has_image"].sum())
print("Records without matching images:", (~df["has_image"]).sum())

print("\nExample matching records:")
print(df.loc[df["has_image"], ["ImgId", "title"]].head(5).to_string(index=False))

print("\nNumber of categories:", df["categories"].nunique())
print("\nTop 10 categories:")
print(df["categories"].value_counts().head(10))
