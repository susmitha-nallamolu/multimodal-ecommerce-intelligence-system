
from pathlib import Path
from torch.utils.data import DataLoader

from data_preprocessing.product_dataset import ProductDataset


# Find the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Find the processed training CSV
TRAIN_CSV = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "processed"
    / "train_processed.csv"
)

print("Starting dataset test...")

# Create the dataset object
dataset = ProductDataset(TRAIN_CSV)

# Check dataset size
print("\nDataset length:", len(dataset))

# Load one product
sample = dataset[0]

print("\n--- First Product ---")
print("Image shape:", sample["image"].shape)
print("Text shape:", sample["text"].shape)
print("Label:", sample["label"].item())

# Create a batch of four products
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

batch = next(iter(loader))

print("\n--- DataLoader Batch ---")
print("Image batch shape:", batch["image"].shape)
print("Text batch shape:", batch["text"].shape)
print("Label batch shape:", batch["label"].shape)

print("\nDataset test completed successfully!")
