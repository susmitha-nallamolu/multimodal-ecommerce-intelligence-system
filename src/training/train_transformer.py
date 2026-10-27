
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "processed" / "train_processed.csv"

MODEL_NAME = "distilbert/distilbert-base-uncased"
NUM_CLASSES = 21
MAX_LENGTH = 128

print("=" * 60)
print("TRANSFORMER MODEL SETUP CHECK")
print("=" * 60)

device = torch.device("cpu")
print("Device:", device)

df = pd.read_csv(DATA_PATH)
print("Dataset rows:", len(df))
print("Available columns:", df.columns.tolist())

texts = df["text"].fillna("").astype(str).head(2).tolist()
labels = df["label"].head(2).astype(int).tolist()

print("\nDownloading/loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Downloading/loading classification model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES
)
model.to(device)
model.eval()

inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=MAX_LENGTH,
    return_tensors="pt"
)

labels_tensor = torch.tensor(labels, dtype=torch.long)

with torch.no_grad():
    outputs = model(**inputs)

print("\nTokenizer and model test successful!")
print("Input IDs shape:", inputs["input_ids"].shape)
print("Labels shape:", labels_tensor.shape)
print("Logits shape:", outputs.logits.shape)
print("Expected logits shape: [2, 21]")
print("\nNote: This is a setup check, not trained classification.")

