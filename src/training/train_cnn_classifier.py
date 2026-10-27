
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data_preprocessing.product_dataset import ProductDataset
from src.models.cnn_classifier import CNNImageClassifier


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CSV = PROJECT_ROOT / "data" / "raw" / "processed" / "train_processed.csv"
VAL_CSV = PROJECT_ROOT / "data" / "raw" / "processed" / "val_processed.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "cnn_image_only.pth"

NUM_CLASSES = 21
MAX_TRAIN_SAMPLES = 1050
MAX_VAL_SAMPLES = 210

BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 0.001
SEED = 42

torch.set_num_threads(4)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")


# --------------------------------------------------
# 2. Stratified sampling
# --------------------------------------------------

def stratified_indices(dataframe, sample_limit, seed):

    rng = np.random.default_rng(seed)
    labels = dataframe["label"].to_numpy(copy=True)
    classes = np.unique(labels)

    selected = []

    per_class = sample_limit // len(classes)

    for class_id in classes:
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)

        count = min(per_class, len(class_indices))
        selected.extend(class_indices[:count].tolist())

    # Fill any remaining slots if necessary.
    if len(selected) < sample_limit:
        selected_set = set(selected)
        remaining = [
            i for i in range(len(labels))
            if i not in selected_set
        ]
        rng.shuffle(remaining)

        needed = sample_limit - len(selected)
        selected.extend(remaining[:needed])

    rng.shuffle(selected)
    return selected


# --------------------------------------------------
# 3. Load datasets
# --------------------------------------------------

print("Loading datasets...")

train_dataset = ProductDataset(TRAIN_CSV)
val_dataset = ProductDataset(VAL_CSV)

train_indices = stratified_indices(
    train_dataset.data,
    min(MAX_TRAIN_SAMPLES, len(train_dataset)),
    SEED
)

val_indices = stratified_indices(
    val_dataset.data,
    min(MAX_VAL_SAMPLES, len(val_dataset)),
    SEED + 1
)

train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(val_dataset, val_indices)

print("Training subset size:", len(train_subset))
print("Validation subset size:", len(val_subset))

print(
    "Training class counts:",
    Counter(
        train_dataset.data.iloc[train_indices]["label"].tolist()
    )
)

print(
    "Validation class counts:",
    Counter(
        val_dataset.data.iloc[val_indices]["label"].tolist()
    )
)


# --------------------------------------------------
# 4. Create data loaders
# --------------------------------------------------

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# 5. Initialize the CNN classifier
# --------------------------------------------------

model = CNNImageClassifier(num_classes=NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)

best_val_loss = float("inf")

print("Training device:", DEVICE)
print("Starting CNN image-only training...")


# --------------------------------------------------
# 6. Training and validation
# --------------------------------------------------

for epoch in range(EPOCHS):

    model.train()

    train_loss_total = 0.0
    train_correct = 0
    train_total = 0

    for batch in train_loader:

        images = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss_total += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_loss = train_loss_total / train_total
    train_accuracy = train_correct / train_total

    # Validation
    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for batch in val_loader:

            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_total += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss_total / val_total
    val_accuracy = val_correct / val_total

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Training loss: {train_loss:.4f}")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")

    # Save the checkpoint with the lowest validation loss.
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_classes": NUM_CLASSES,
                "epoch": epoch + 1,
                "validation_loss": val_loss,
                "validation_accuracy": val_accuracy,
            },
            MODEL_PATH
        )

        print("Best CNN checkpoint saved.")


print("\nCNN image-only training completed.")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Model saved at: {MODEL_PATH}")
