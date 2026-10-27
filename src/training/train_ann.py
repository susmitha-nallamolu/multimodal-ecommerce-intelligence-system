
from pathlib import Path
import json
import re
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from src.models.ann_model import ANNProductClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT / "data" / "raw" / "processed"
    / "train_processed.csv"
)
VAL_PATH = (
    PROJECT_ROOT / "data" / "raw" / "processed"
    / "val_processed.csv"
)
LABEL_PATH = (
    PROJECT_ROOT / "data" / "raw" / "processed"
    / "label_mapping.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODEL_DIR / "ann_product_classifier.pth"
SCALER_PATH = MODEL_DIR / "ann_feature_scaler.pkl"
METRICS_PATH = REPORT_DIR / "ann_classification_report.txt"
PREDICTIONS_PATH = REPORT_DIR / "ann_validation_predictions.csv"
HISTORY_PATH = REPORT_DIR / "ann_training_history.csv"

NUM_CLASSES = 21
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001
RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 60)
print("ANN PRODUCT CLASSIFIER - TRAINING")
print("=" * 60)
print("Device:", DEVICE)


# ============================================================
# 2. LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
label_df = pd.read_csv(LABEL_PATH)

required_columns = {"title", "description", "label"}

for dataset_name, dataset in [
    ("Training", train_df),
    ("Validation", val_df),
]:
    missing = required_columns - set(dataset.columns)

    if missing:
        raise ValueError(
            f"{dataset_name} dataset is missing columns: {missing}"
        )

if not set(train_df["label"].unique()).issubset(
    set(label_df["label"])
):
    raise ValueError("Training labels do not match label_mapping.csv")

if not set(val_df["label"].unique()).issubset(
    set(label_df["label"])
):
    raise ValueError("Validation labels do not match label_mapping.csv")

if (
    train_df["label"].min() < 0
    or train_df["label"].max() >= NUM_CLASSES
    or val_df["label"].min() < 0
    or val_df["label"].max() >= NUM_CLASSES
):
    raise ValueError("Labels must be integers from 0 to 20.")

print("Training rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Number of categories:", label_df["label"].nunique())


# ============================================================
# 3. EXTRACT 10 NUMERICAL FEATURES
# ============================================================

FEATURE_NAMES = [
    "title_character_count",
    "title_word_count",
    "description_character_count",
    "description_word_count",
    "combined_character_count",
    "combined_word_count",
    "unique_word_ratio",
    "average_word_length",
    "digit_count",
    "punctuation_count",
]


def extract_features(title, description):
    """
    Convert product text into 10 numerical features.

    These features describe the text; they do not capture
    the full semantic meaning of the product.
    """

    title = str(title) if pd.notna(title) else ""
    description = (
        str(description) if pd.notna(description) else ""
    )

    combined_text = f"{title} {description}".strip()

    title_words = re.findall(r"\b\w+\b", title.lower())
    description_words = re.findall(
        r"\b\w+\b", description.lower()
    )
    combined_words = re.findall(
        r"\b\w+\b", combined_text.lower()
    )

    unique_ratio = (
        len(set(combined_words)) / len(combined_words)
        if combined_words else 0.0
    )

    average_word_length = (
        np.mean([len(word) for word in combined_words])
        if combined_words else 0.0
    )

    digit_count = sum(
        character.isdigit()
        for character in combined_text
    )

    punctuation_count = sum(
        not character.isalnum() and not character.isspace()
        for character in combined_text
    )

    return [
        len(title),
        len(title_words),
        len(description),
        len(description_words),
        len(combined_text),
        len(combined_words),
        unique_ratio,
        average_word_length,
        digit_count,
        punctuation_count,
    ]


def build_feature_matrix(dataset):
    features = [
        extract_features(row.title, row.description)
        for row in dataset.itertuples(index=False)
    ]

    return np.asarray(features, dtype=np.float32)


X_train_raw = build_feature_matrix(train_df)
X_val_raw = build_feature_matrix(val_df)

y_train = train_df["label"].to_numpy(dtype=np.int64)
y_val = val_df["label"].to_numpy(dtype=np.int64)

print("\nFeature matrix shapes:")
print("Training:", X_train_raw.shape)
print("Validation:", X_val_raw.shape)


# ============================================================
# 4. SCALE NUMERICAL FEATURES
# ============================================================

scaler = StandardScaler()

# Fit ONLY on training data to avoid validation leakage.
X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
X_val = scaler.transform(X_val_raw).astype(np.float32)

with open(SCALER_PATH, "wb") as file:
    pickle.dump(scaler, file)

print("Feature scaler saved:", SCALER_PATH.name)


# ============================================================
# 5. CREATE PYTORCH DATASETS
# ============================================================

X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train)

X_val_tensor = torch.from_numpy(X_val)
y_val_tensor = torch.from_numpy(y_val)

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 6. INITIALIZE ANN
# ============================================================

model = ANNProductClassifier(
    input_dim=len(FEATURE_NAMES),
    num_classes=NUM_CLASSES
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("\nANN initialized successfully.")
print("Input features:", len(FEATURE_NAMES))
print("Output classes:", NUM_CLASSES)


# ============================================================
# 7. TRAIN AND VALIDATE
# ============================================================

history = []
best_val_loss = float("inf")
best_epoch = 0

for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0.0
    train_correct = 0
    train_total = 0

    for features, labels in train_loader:

        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_train_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)
        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += batch_size

    train_loss = total_train_loss / train_total
    train_accuracy = train_correct / train_total

    # ---------------- VALIDATION ----------------

    model.eval()

    total_val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for features, labels in val_loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(features)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_val_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += batch_size

    val_loss = total_val_loss / val_total
    val_accuracy = val_correct / val_total

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    })

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy:.4f} "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Acc: {val_accuracy:.4f}"
    )

    # Save the best model according to validation loss.
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch + 1

        torch.save({
            "model_state_dict": model.state_dict(),
            "input_dim": len(FEATURE_NAMES),
            "num_classes": NUM_CLASSES,
            "feature_names": FEATURE_NAMES,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        }, CHECKPOINT_PATH)


# ============================================================
# 8. SAVE TRAINING HISTORY
# ============================================================

pd.DataFrame(history).to_csv(
    HISTORY_PATH,
    index=False
)

print("\nTraining history saved:", HISTORY_PATH.name)
print("Best epoch:", best_epoch)
print(f"Best validation loss: {best_val_loss:.4f}")


# ============================================================
# 9. EVALUATE THE BEST CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_predictions = []

with torch.no_grad():

    for features, _ in val_loader:

        features = features.to(DEVICE)

        outputs = model(features)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

accuracy = accuracy_score(y_val, all_predictions)

macro_f1 = f1_score(
    y_val,
    all_predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_val,
    all_predictions,
    average="weighted",
    zero_division=0
)

target_names = [
    str(label_to_category)
    for label_to_category in label_df.sort_values("label")[
        "category"
    ].tolist()
]

report = classification_report(
    y_val,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=target_names,
    zero_division=0
)

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    file.write("ANN PRODUCT CLASSIFIER EVALUATION\n")
    file.write("=" * 50 + "\n")
    file.write(f"Validation accuracy: {accuracy:.4f}\n")
    file.write(f"Macro F1: {macro_f1:.4f}\n")
    file.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
    file.write(report)

prediction_df = val_df[
    ["title", "description", "label"]
].copy()

prediction_df["predicted_label"] = all_predictions

prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)

print("\n" + "=" * 60)
print("ANN EVALUATION RESULTS")
print("=" * 60)
print(f"Validation accuracy: {accuracy:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")
print("\nClassification report:")
print(report)

print("\nSaved files:")
print("Model:", CHECKPOINT_PATH)
print("Scaler:", SCALER_PATH)
print("History:", HISTORY_PATH)
print("Metrics:", METRICS_PATH)
print("Predictions:", PREDICTIONS_PATH)
