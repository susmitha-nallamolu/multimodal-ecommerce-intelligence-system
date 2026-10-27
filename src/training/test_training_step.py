
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_preprocessing.product_dataset import (
    ProductDataset,
    image_transform
)
from src.models.multimodal_model import MultimodalModel


# 1. Find the project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

csv_path = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "processed"
    / "train_processed.csv"
)

# 2. Load the real training dataset
train_dataset = ProductDataset(
    csv_file=str(csv_path),
    transform=image_transform
)

# 3. Use a small batch for the first training test
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

# 4. Create the complete multimodal model
model = MultimodalModel(num_classes=21)

# 5. Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# 6. Get one batch of real products
batch = next(iter(train_loader))

images = batch["image"]
texts = batch["text"]
labels = batch["label"]

# 7. Train on this batch
model.train()

optimizer.zero_grad()

outputs = model(images, texts)

loss = criterion(outputs, labels)

loss.backward()

optimizer.step()

# 8. Display results
predictions = outputs.argmax(dim=1)
correct = (predictions == labels).sum().item()

print("\nFirst training step successful!")
print("Images shape:", images.shape)
print("Texts shape:", texts.shape)
print("Labels shape:", labels.shape)
print("Output shape:", outputs.shape)
print("Training loss:", loss.item())
print("Correct predictions before update:", correct, "out of", len(labels))
