
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# 1. Define dataset paths
data_folder = Path("data/raw")
csv_path = data_folder / "train.csv"
image_folder = data_folder / "train"

# 2. Load the training dataset
df = pd.read_csv(csv_path)

# 3. Find available product images
image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

image_paths = {
    path.stem: path
    for path in image_folder.rglob("*")
    if path.is_file()
    and path.suffix.lower() in image_extensions
}

# 4. Keep only records with matching images
df["ImgId"] = df["ImgId"].astype(str)
df = df[df["ImgId"].isin(image_paths)].copy()

# 5. Handle missing text
df["title"] = df["title"].fillna("Unknown product")
df["description"] = df["description"].fillna("")

print("Records available for image-text training:", len(df))
print("Number of categories:", df["categories"].nunique())

# 6. Display the category distribution
print("\nProducts per category:")
print(df["categories"].value_counts())

# 7. Select six products to visualize
samples = df.sample(n=6, random_state=42)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for ax, (_, row) in zip(axes, samples.iterrows()):
    image_path = image_paths[row["ImgId"]]

    with Image.open(image_path) as image:
        ax.imshow(image.convert("RGB"))

    ax.set_title(
        f'{row["categories"]}\n{row["title"][:55]}',
        fontsize=9
    )
    ax.axis("off")

plt.tight_layout()
plt.show()
