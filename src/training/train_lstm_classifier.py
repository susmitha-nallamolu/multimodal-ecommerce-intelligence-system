
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data_preprocessing.product_dataset import ProductDataset
from src.models.lstm_classifier import LSTMTextClassifier


# Configuration
PROJECT_ROOT = Path(
    r"C:\Users\susmi\multimodal_ecommerce_intelligence"
)

TRAIN_CSV = PROJECT_ROOT / "data" / "raw" / "processed" / "train_processed.csv"
VAL_CSV = PROJECT_ROOT / "data" / "raw" / "processed" / "val_processed.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "lstm_text_only.pth"

NUM_CLASSES = 21
SAMPLES_PER_CLASS = 50
VAL_SAMPLES_PER_CLASS = 10
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 0.001
SEED = 42

DEVICE = torch.device("cpu")
torch.set_num_threads(4)


def stratified_indices(dataset, samples_per_class, seed):
    labels = dataset.data["label"].to_numpy()
    rng = np.random.default_rng(seed)
    selected = []

    for label in sorted(np.unique(labels)):
        class_indices = np.where(labels == label)[0]
        count = min(samples_per_class, len(class_indices))

        selected.extend(
            rng.choice(
                class_indices,
                size=count,
                replace=False
            ).tolist()
        )

    return sorted(selected)


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            texts = batch["text"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(texts)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

    return total_loss / total, correct / total


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading datasets...")

    train_dataset = ProductDataset(str(TRAIN_CSV))
    val_dataset = ProductDataset(str(VAL_CSV))

    train_indices = stratified_indices(
        train_dataset, SAMPLES_PER_CLASS, SEED
    )

    val_indices = stratified_indices(
        val_dataset, VAL_SAMPLES_PER_CLASS, SEED + 1
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Training samples:", len(train_indices))
    print("Validation samples:", len(val_indices))
    print("Device:", DEVICE)

    model = LSTMTextClassifier(
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, criterion
        )

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Train accuracy: {train_acc:.2%}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_acc:.2%}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            MODEL_PATH.parent.mkdir(
                parents=True, exist_ok=True
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": NUM_CLASSES,
                    "epoch": epoch + 1,
                    "validation_loss": val_loss,
                    "validation_accuracy": val_acc,
                },
                MODEL_PATH
            )

            print("Best checkpoint saved.")

    print("\nText-only LSTM training completed.")
    print("Model path:", MODEL_PATH)


if __name__ == "__main__":
    main()
