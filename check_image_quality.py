
from pathlib import Path
from PIL import Image

image_dir = Path(
    r"C:\Users\susmi\multimodal_ecommerce_intelligence\data\raw\train\train"
)

image_files = list(image_dir.glob("*.jpg"))[:20]

print("Checking image dimensions...\n")

for image_path in image_files:
    with Image.open(image_path) as image:
        print(image_path.name, "->", image.size)

print("\nTotal images checked:", len(image_files))
