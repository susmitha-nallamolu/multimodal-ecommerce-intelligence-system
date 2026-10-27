
from pathlib import Path
import pandas as pd

data_folder = Path("data/raw")

# Inspect both CSV files
for filename in ["train.csv", "test.csv"]:
    file_path = data_folder / filename

    print("\n" + "=" * 50)
    print("FILE:", filename)

    df = pd.read_csv(file_path)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

# Count image files in both folders
image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

for folder_name in ["train", "test"]:
    folder_path = data_folder / folder_name

    images = [
        path for path in folder_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in image_extensions
    ]

    print(f"\nImages in {folder_name}: {len(images)}")
    if images:
        print("Example image:", images[0].name)
