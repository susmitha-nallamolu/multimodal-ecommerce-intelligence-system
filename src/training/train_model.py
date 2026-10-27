
from pathlib import Path
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

from data_preprocessing.product_dataset import ProductDataset
from models.multimodal_model import MultimodalModel


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_FOLDER = (
    PROJECT_ROOT / "data" / "raw" / "processed"
)

TRAIN_CSV = PROCESSED_FOLDER / "train_processed.csv"
VAL_CSV = PROCESSED_FOLDER / "val_processed.csv"

MODEL_FOLDER = PROJECT_ROOT / "models"
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

# Keep the original checkpoint safe.
CHECKPOINT_PATH = (
    MODEL_FOLDER / "multimodal_model_fasttrack.pth"
)

NUM_CLASSES = 21

# Faster experiment with more training data.
MAX_TRAIN_SAMPLES = 8400
MAX_VAL_SAMPLES = 2100

BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0001

SEED = 42
PATIENCE = 2


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Use GPU if available; otherwise use CPU.
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("MULTIMODAL E-COMMERCE INTELLIGENCE SYSTEM")
print("=" * 60)
print("Project root:", PROJECT_ROOT)
print("Training device:", device)
print("Training CSV:", TRAIN_CSV)
print("Validation CSV:", VAL_CSV)
print("Checkpoint:", CHECKPOINT_PATH)


# ============================================================
# 3. CHECK INPUT FILES
# ============================================================

if not TRAIN_CSV.is_file():
    raise FileNotFoundError(
        f"Training CSV not found: {TRAIN_CSV}"
    )

if not VAL_CSV.is_file():
    raise FileNotFoundError(
        f"Validation CSV not found: {VAL_CSV}"
    )


# ============================================================
# 4. LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

train_dataset = ProductDataset(TRAIN_CSV)
val_dataset = ProductDataset(VAL_CSV)

if "label" not in train_dataset.data.columns:
    raise ValueError(
        "Training CSV must contain a 'label' column."
    )

if "label" not in val_dataset.data.columns:
    raise ValueError(
        "Validation CSV must contain a 'label' column."
    )

train_labels = set(
    train_dataset.data["label"].astype(int).unique()
)

val_labels = set(
    val_dataset.data["label"].astype(int).unique()
)

expected_labels = set(range(NUM_CLASSES))

if train_labels != expected_labels:
    raise ValueError(
        f"Training labels must be 0 to {NUM_CLASSES - 1}. "
        f"Found: {sorted(train_labels)}"
    )

if not val_labels.issubset(expected_labels):
    raise ValueError(
        "Validation CSV contains labels outside the expected range."
    )

print("Full training dataset:", len(train_dataset))
print("Full validation dataset:", len(val_dataset))


# ============================================================
# 5. CREATE STRATIFIED SUBSETS
# ============================================================

def stratified_indices(dataframe, sample_limit, seed):
    """
    Select a representative subset while preserving
    approximately the original class proportions.
    """

    rng = np.random.default_rng(seed)

    if sample_limit >= len(dataframe):
        return list(range(len(dataframe)))

    labels = dataframe["label"].astype(int).to_numpy()
    class_indices = {}

    for label in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        class_indices[label] = indices

    # Allocate samples proportionally to class sizes.
    exact_quotas = {
        label: sample_limit * len(indices) / len(dataframe)
        for label, indices in class_indices.items()
    }

    quotas = {
        label: int(quota)
        for label, quota in exact_quotas.items()
    }

    # Distribute remaining samples by largest fractional
    # remainder, without exceeding a class's available size.
    remaining_count = sample_limit - sum(quotas.values())

    labels_by_remainder = sorted(
        class_indices,
        key=lambda label: (
            exact_quotas[label] - quotas[label]
        ),
        reverse=True
    )

    for label in labels_by_remainder:
        if remaining_count == 0:
            break

        if quotas[label] < len(class_indices[label]):
            quotas[label] += 1
            remaining_count -= 1

    selected = []

    for label, indices in class_indices.items():
        selected.extend(indices[:quotas[label]].tolist())

    rng.shuffle(selected)

    if len(selected) != sample_limit:
        raise RuntimeError(
            f"Expected {sample_limit} samples, "
            f"but selected {len(selected)}."
        )

    return selected


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

print("\nTraining subset size:", len(train_subset))
print("Validation subset size:", len(val_subset))

print(
    "Training classes represented:",
    train_dataset.data.iloc[train_indices]["label"].nunique()
)

print(
    "Validation classes represented:",
    val_dataset.data.iloc[val_indices]["label"].nunique()
)


# ============================================================
# 6. CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)


# ============================================================
# 7. INITIALIZE MODEL, LOSS AND OPTIMIZER
# ============================================================

model = MultimodalModel(
    num_classes=NUM_CLASSES
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# Reduce the learning rate when validation loss plateaus.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=1
)


# ============================================================
# 8. TRAINING CONFIGURATION
# ============================================================

best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0

history = []


# ============================================================
# 9. TRAINING AND VALIDATION LOOP
# ============================================================

for epoch in range(EPOCHS):

    print("\n" + "=" * 60)
    print(f"EPOCH {epoch + 1}/{EPOCHS}")
    print("=" * 60)

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    train_loss_total = 0.0
    train_correct = 0
    train_total = 0

    for batch in train_loader:

        images = batch["image"].to(device)
        texts = batch["text"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images, texts)
        loss = criterion(outputs, labels)

        loss.backward()

        # Prevent unusually large gradients.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        current_batch_size = labels.size(0)

        train_loss_total += (
            loss.item() * current_batch_size
        )

        predictions = outputs.argmax(dim=1)

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += current_batch_size

    train_loss = train_loss_total / train_total
    train_accuracy = train_correct / train_total


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    val_true_labels = []
    val_pred_labels = []

    with torch.no_grad():

        for batch in val_loader:

            images = batch["image"].to(device)
            texts = batch["text"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images, texts)
            loss = criterion(outputs, labels)

            current_batch_size = labels.size(0)

            val_loss_total += (
                loss.item() * current_batch_size
            )

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_true_labels.extend(
                labels.cpu().tolist()
            )

            val_pred_labels.extend(
                predictions.cpu().tolist()
            )

            val_total += current_batch_size

    val_loss = val_loss_total / val_total
    val_accuracy = val_correct / val_total

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(f"Training loss:       {train_loss:.4f}")
    print(f"Training accuracy:   {train_accuracy:.2%}")
    print(f"Validation loss:     {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.2%}")
    print(f"Learning rate:       {current_lr:.6f}")

    print(
        "Actual validation class counts:",
        Counter(val_true_labels)
    )

    print(
        "Predicted validation class counts:",
        Counter(val_pred_labels)
    )

    print(
        "Number of predicted classes:",
        len(set(val_pred_labels))
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "learning_rate": current_lr
    })


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch + 1
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            CHECKPOINT_PATH
        )

        print("Best validation checkpoint saved.")

    else:

        epochs_without_improvement += 1

        print(
            "No validation-loss improvement:",
            epochs_without_improvement,
            "/",
            PATIENCE
        )


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print(
            "\nEarly stopping: validation loss "
            "did not improve."
        )
        break


# ============================================================
# 10. SAVE TRAINING HISTORY
# ============================================================

history_file = MODEL_FOLDER / "training_history.csv"

pd.DataFrame(history).to_csv(
    history_file,
    index=False
)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING EXPERIMENT COMPLETED")
print("=" * 60)

print("Best epoch:", best_epoch)
print("Best validation loss:", round(best_val_loss, 4))
print("Checkpoint saved at:", CHECKPOINT_PATH)
print("Training history saved at:", history_file)

if CHECKPOINT_PATH.is_file():
    print(
        "Checkpoint size:",
        f"{CHECKPOINT_PATH.stat().st_size:,} bytes"
    )
else:
    print("WARNING: Checkpoint file was not found.")

print("\nNext step: evaluate the saved checkpoint on the")
print("full validation dataset before finalizing the model.")
