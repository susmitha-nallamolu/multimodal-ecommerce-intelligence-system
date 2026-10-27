from pathlib import Path
import json
import re

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms



# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

# Project root:
# multimodal_ecommerce_intelligence/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_FOLDER = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "processed"
)

VOCAB_FILE = PROCESSED_FOLDER / "vocabulary.json"

MAX_LENGTH = 100
PAD_ID = 0
UNK_ID = 1


# --------------------------------------------------
# 2. IMAGE TRANSFORMATION
# --------------------------------------------------

# Converts every product image into a tensor
# with shape [3, 224, 224].
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# --------------------------------------------------
# 3. PRODUCT DATASET CLASS
# --------------------------------------------------

class ProductDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        # Load processed product information
        self.data = pd.read_csv(csv_file)

        # Use the default image transformation if none
        # is supplied by the caller.
        self.transform = (
            transform if transform is not None
            else image_transform
        )

        # Load the word-to-ID vocabulary
        with open(VOCAB_FILE, "r", encoding="utf-8") as file:
            self.vocabulary = json.load(file)

        # Validate the vocabulary structure
        if not isinstance(self.vocabulary, dict):
            raise TypeError(
                "vocabulary.json must contain a word-to-ID dictionary."
            )

        print(f"Dataset loaded: {csv_file}")
        print(f"Number of products: {len(self.data)}")
        print(f"Vocabulary size: {len(self.vocabulary)}")


        # Validate the vocabulary structure
        if not isinstance(self.vocabulary, dict):
            raise TypeError(
                "vocabulary.json must contain a word-to-ID dictionary."
            )

        # Validate special-token IDs
        if self.vocabulary.get("<PAD>") != PAD_ID:
            raise ValueError(
                "Vocabulary must map '<PAD>' to PAD_ID = 0."
            )

        if self.vocabulary.get("<UNK>") != UNK_ID:
            raise ValueError(
                "Vocabulary must map '<UNK>' to UNK_ID = 1."
            )


    # --------------------------------------------------
    # 4. TEXT ENCODING
    # --------------------------------------------------

    def encode_text(self, text):

        # Handle missing or non-string text
        if not isinstance(text, str):
            text = ""

        # Convert text into lowercase word tokens
        tokens = re.findall(
            r"\b\w+\b",
            text.lower()
        )

        # Convert words to vocabulary IDs.
        # Unknown words use UNK_ID.
        token_ids = [
            self.vocabulary.get(token, UNK_ID)
            for token in tokens
        ]

        # Truncate text if it exceeds MAX_LENGTH
        token_ids = token_ids[:MAX_LENGTH]

        # Pad shorter sequences to MAX_LENGTH
        padding_length = MAX_LENGTH - len(token_ids)
        token_ids.extend([PAD_ID] * padding_length)

        # Return integer tensor for the embedding layer
        return torch.tensor(
            token_ids,
            dtype=torch.long
        )

    # --------------------------------------------------
    # 5. LOAD ONE PRODUCT
    # --------------------------------------------------

    def __getitem__(self, index):

        # Retrieve one row from the CSV
        row = self.data.iloc[index]

        # Get image path
        image_path = Path(row["image_path"])

        # Fail clearly if an image is missing
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Product image not found: {image_path}"
            )

        # Open image and convert to RGB
        with Image.open(image_path) as image:
            image = image.convert("RGB")

            # Convert image into a tensor
            image = self.transform(image)

        # Encode product title and description text
        text = self.encode_text(row["text"])

        # Convert category label into an integer tensor
        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long
        )

        # Return all three inputs/targets
        return {
            "image": image,
            "text": text,
            "label": label
        }

    # --------------------------------------------------
    # 6. DATASET LENGTH
    # --------------------------------------------------

    def __len__(self):
        return len(self.data)

