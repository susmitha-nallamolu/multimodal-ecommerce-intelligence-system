
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "processed" / "train_processed.csv"
VAL_PATH = PROJECT_ROOT / "data" / "raw" / "processed" / "val_processed.csv"

OUTPUT_DIR = PROJECT_ROOT / "models" / "transformer_distilbert_3epoch"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_NAME = "distilbert/distilbert-base-uncased"
NUM_CLASSES = 21
MAX_LENGTH = 128
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 3
LEARNING_RATE = 2e-5

TRAIN_SAMPLES_PER_CLASS = 100
VAL_SAMPLES_PER_CLASS = 40
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cpu")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("DISTILBERT PRODUCT CLASSIFIER - CPU FINE-TUNING")
print("=" * 65)
print("Device:", DEVICE)

# --------------------------------------------------
# 2. Load a balanced subset of the dataset
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

required_columns = {"text", "label"}
for name, df in [("training", train_df), ("validation", val_df)]:
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{name} dataset is missing columns: {missing}")

def select_balanced_subset(df, samples_per_class):
    selected = (
        df.groupby("label", group_keys=False)
        .sample(
            n=samples_per_class,
            random_state=SEED,
            replace=False,
        )
        .reset_index(drop=True)
    )
    return selected

train_df = select_balanced_subset(train_df, TRAIN_SAMPLES_PER_CLASS)
val_df = select_balanced_subset(val_df, VAL_SAMPLES_PER_CLASS)

train_df["text"] = train_df["text"].fillna("").astype(str)
val_df["text"] = val_df["text"].fillna("").astype(str)

train_labels = train_df["label"].astype(int).to_numpy()
val_labels = val_df["label"].astype(int).to_numpy()

print("Training subset:", len(train_df))
print("Validation subset:", len(val_df))
print("Categories:", train_df["label"].nunique())

# --------------------------------------------------
# 3. Load tokenizer
# --------------------------------------------------

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --------------------------------------------------
# 4. Dataset class
# --------------------------------------------------

class ProductTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoded.items()
        }
        item["labels"] = torch.tensor(
            int(self.labels[index]),
            dtype=torch.long,
        )
        return item

train_dataset = ProductTextDataset(
    train_df["text"].tolist(),
    train_labels,
    tokenizer,
    MAX_LENGTH,
)
val_dataset = ProductTextDataset(
    val_df["text"].tolist(),
    val_labels,
    tokenizer,
    MAX_LENGTH,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# --------------------------------------------------
# 5. Load pretrained DistilBERT
# --------------------------------------------------

print("\nLoading pretrained DistilBERT...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
)
model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

total_steps = (
    len(train_loader) + GRADIENT_ACCUMULATION_STEPS - 1
) // GRADIENT_ACCUMULATION_STEPS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps * EPOCHS,
)

# --------------------------------------------------
# 6. Fine-tune model
# --------------------------------------------------

history = []
best_val_loss = float("inf")
best_checkpoint_path = OUTPUT_DIR / "best_model.pth"

for epoch in range(EPOCHS):
    start_time = time.time()
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(train_loader):
        batch = {
            key: value.to(DEVICE)
            for key, value in batch.items()
        }

        outputs = model(**batch)
        loss = outputs.loss
        (loss / GRADIENT_ACCUMULATION_STEPS).backward()

        if (
            (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0
            or (step + 1) == len(train_loader)
        ):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()
        predictions = outputs.logits.argmax(dim=1)
        correct += (predictions == batch["labels"]).sum().item()
        total += batch["labels"].size(0)

        if (step + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {step + 1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total

    # ----------------------------------------------
    # Validation
    # ----------------------------------------------

    model.eval()
    val_running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            val_running_loss += outputs.loss.item()

            predictions = outputs.logits.argmax(dim=1)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    val_loss = val_running_loss / len(val_loader)
    val_accuracy = accuracy_score(all_labels, all_predictions)
    val_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )
    elapsed = time.time() - start_time

    print("\n" + "-" * 65)
    print(f"Epoch: {epoch + 1}/{EPOCHS}")
    print(f"Training loss: {train_loss:.4f}")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Validation macro F1: {val_macro_f1:.4f}")
    print(f"Time: {elapsed / 60:.1f} minutes")
    print("-" * 65)

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "time_seconds": elapsed,
    })

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_name": MODEL_NAME,
                "num_classes": NUM_CLASSES,
                "max_length": MAX_LENGTH,
                "best_val_loss": best_val_loss,
                "label_mapping": {
                    str(int(label)): str(category)
                    for label, category in (
                        train_df[["label", "categories"]]
                        .drop_duplicates()
                        .itertuples(index=False, name=None)
                    )
                },
            },
            best_checkpoint_path,
        )

        tokenizer.save_pretrained(OUTPUT_DIR)
        print("Best checkpoint saved:", best_checkpoint_path)

# --------------------------------------------------
# 7. Save metrics and validation report
# --------------------------------------------------

history_path = REPORT_DIR / "transformer_training_history.csv"
pd.DataFrame(history).to_csv(history_path, index=False)

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=[
        str(
            train_df.loc[
                train_df["label"] == label, "categories"
            ].iloc[0]
        )
        for label in range(NUM_CLASSES)
    ],
    zero_division=0,
)

report_path = REPORT_DIR / "transformer_classification_report.txt"
report_path.write_text(report, encoding="utf-8")

predictions_path = REPORT_DIR / "transformer_validation_predictions.csv"
pd.DataFrame({
    "actual_label": all_labels,
    "predicted_label": all_predictions,
}).to_csv(predictions_path, index=False)

print("\nTRAINING COMPLETE")
print("History:", history_path)
print("Classification report:", report_path)
print("Predictions:", predictions_path)
print("Best checkpoint:", best_checkpoint_path)
