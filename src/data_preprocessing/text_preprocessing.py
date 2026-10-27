
from collections import Counter
from pathlib import Path
import re
import json

import pandas as pd


# ============================================================
# STEP 1: DEFINE FILE PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FOLDER = PROJECT_ROOT / "data" / "raw" / "processed"

TRAIN_CSV = DATA_FOLDER / "train_processed.csv"

VOCAB_FILE = DATA_FOLDER / "vocabulary.json"


# ============================================================
# STEP 2: DEFINE SPECIAL TOKENS
# ============================================================

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

PAD_ID = 0
UNK_ID = 1


# ============================================================
# STEP 3: TOKENIZE TEXT
# ============================================================

def tokenize(text):
    text = str(text).lower()

    # Keep words and numbers; ignore punctuation.
    tokens = re.findall(r"\b\w+\b", text)

    return tokens


# ============================================================
# STEP 4: LOAD TRAINING TEXT
# ============================================================

df = pd.read_csv(TRAIN_CSV)

texts = df["text"].fillna("").astype(str)

print("Training text records:", len(texts))


# ============================================================
# STEP 5: COUNT WORD FREQUENCIES
# ============================================================

word_counts = Counter()

for text in texts:
    tokens = tokenize(text)
    word_counts.update(tokens)

print("Unique tokens before filtering:", len(word_counts))


# ============================================================
# STEP 6: BUILD THE VOCABULARY
# ============================================================

MIN_FREQUENCY = 2

vocabulary = {
    PAD_TOKEN: PAD_ID,
    UNK_TOKEN: UNK_ID
}

for word, count in word_counts.most_common():
    if count >= MIN_FREQUENCY and word not in vocabulary:
        vocabulary[word] = len(vocabulary)


# ============================================================
# STEP 7: SAVE THE VOCABULARY
# ============================================================

with open(VOCAB_FILE, "w", encoding="utf-8") as file:
    json.dump(vocabulary, file, ensure_ascii=False, indent=2)


# ============================================================
# STEP 8: VERIFY THE RESULT
# ============================================================

print("\nVocabulary created successfully.")
print("Vocabulary size:", len(vocabulary))
print("PAD token ID:", vocabulary[PAD_TOKEN])
print("UNK token ID:", vocabulary[UNK_TOKEN])
print("Saved vocabulary to:", VOCAB_FILE)

example_text = "Whirlpool Cabrio Gas Dryer White"
example_tokens = tokenize(example_text)

example_ids = [
    vocabulary.get(token, vocabulary[UNK_TOKEN])
    for token in example_tokens
]

print("\nExample text:", example_text)
print("Tokens:", example_tokens)
print("Token IDs:", example_ids)
