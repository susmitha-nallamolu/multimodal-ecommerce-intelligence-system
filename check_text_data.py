
from pathlib import Path
from collections import Counter

import pandas as pd
import json
import re

PROJECT_ROOT = Path(__file__).resolve().parent

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "processed"
    / "train_processed.csv"
)

VOCAB_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "processed"
    / "vocabulary.json"
)

df = pd.read_csv(CSV_PATH)

with open(VOCAB_PATH, "r", encoding="utf-8") as file:
    vocabulary = json.load(file)

print("Dataset shape:", df.shape)
print("Vocabulary size:", len(vocabulary))
print("Missing text:", df["text"].isna().sum())

unknown_counts = Counter()
total_tokens = 0

for text in df["text"].dropna().astype(str).head(1000):
    tokens = re.findall(r"\b\w+\b", text.lower())

    total_tokens += len(tokens)

    for token in tokens:
        if token not in vocabulary:
            unknown_counts[token] += 1

unknown_total = sum(unknown_counts.values())

print("Tokens checked:", total_tokens)
print("Unknown tokens:", unknown_total)

if total_tokens:
    print(
        "Unknown-token rate:",
        round(unknown_total / total_tokens * 100, 2),
        "%"
    )

print("\nExample product text:")
print(df["text"].iloc[0][:300])

print("\nVocabulary examples:")
print(list(vocabulary.items())[:15])
