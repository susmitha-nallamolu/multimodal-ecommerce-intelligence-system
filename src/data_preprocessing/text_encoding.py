
from pathlib import Path
import json
import re

import torch


# ============================================================
# STEP 1: DEFINE PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_FOLDER = (
    PROJECT_ROOT / "data" / "raw" / "processed"
)

VOCAB_FILE = PROCESSED_FOLDER / "vocabulary.json"

MAX_LENGTH = 100


# ============================================================
# STEP 2: LOAD THE SAVED VOCABULARY
# ============================================================

with open(VOCAB_FILE, "r", encoding="utf-8") as file:
    vocabulary = json.load(file)

PAD_ID = vocabulary["<PAD>"]
UNK_ID = vocabulary["<UNK>"]

print("Vocabulary loaded:", len(vocabulary))
print("Maximum sequence length:", MAX_LENGTH)


# ============================================================
# STEP 3: TOKENIZE AND ENCODE TEXT
# ============================================================

def encode_text(text):

    # Convert text to lowercase.
    text = str(text).lower()

    # Split text into words and numbers.
    tokens = re.findall(r"\b\w+\b", text)

    # Convert each token into its vocabulary ID.
    token_ids = [
        vocabulary.get(token, UNK_ID)
        for token in tokens
    ]

    # Keep at most MAX_LENGTH tokens.
    token_ids = token_ids[:MAX_LENGTH]

    # Add padding if the sequence is too short.
    if len(token_ids) < MAX_LENGTH:
        padding_length = MAX_LENGTH - len(token_ids)

        token_ids.extend([PAD_ID] * padding_length)

    # Convert the list into a PyTorch tensor.
    return torch.tensor(token_ids, dtype=torch.long)


# ============================================================
# STEP 4: TEST THE TEXT ENCODER
# ============================================================

if __name__ == "__main__":

    sample_text = (
        "Whirlpool Cabrio Gas Dryer White"
    )

    encoded = encode_text(sample_text)

    print("\nOriginal text:", sample_text)
    print("Encoded sequence:", encoded.tolist())
    print("Sequence shape:", encoded.shape)
    print("Tensor data type:", encoded.dtype)

    print(
        "Non-padding tokens:",
        (encoded != PAD_ID).sum().item()
    )
